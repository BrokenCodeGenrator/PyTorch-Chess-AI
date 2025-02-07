#!/usr/bin/env python3
#the comment above makes it posible to easily use whit a chess gui, for example cutechess
import chess
import sys
import torch
from Model import ChessNet  
import Encoder  

# Global variables
board = chess.Board()
cuda = torch.cuda.is_available()


# Model parameters
num_blocks = 15
num_filters = 128
model_path = './Newest_model_4.pt'  

def load_model():
    """Load the AI model for evaluation."""
    model = ChessNet(num_blocks, num_filters)
    if cuda:
        model = model.cuda()
        checkpoint = torch.load(model_path, map_location='cuda')
    else:
        checkpoint = torch.load(model_path, map_location='cpu')

    model.load_state_dict(checkpoint)
    model.eval()
    return model

ai_model = load_model()  # Load the model 

node_count = 0

def reset_node_count():
    global node_count
    node_count = 0

def increment_node_count():
    global node_count
    node_count += 1

def get_node_count():
    global node_count
    return node_count

def evaluate_position_with_ai(board):
 
    if board.is_checkmate():
        return -1 
    
    if board.is_stalemate() or board.is_insufficient_material() or  board.is_fifty_moves() or board.can_claim_threefold_repetition():
        return 0


    position = Encoder.encodePositionForInference(board)  
    position_tensor = torch.from_numpy(position).float().unsqueeze(0) 
    if cuda:
        position_tensor = position_tensor.cuda()

    with torch.no_grad():
        value = ai_model(position_tensor).item()  

    return value

nodes_visited = 0

def increment_node_visit():
    global nodes_visited
    nodes_visited += 1

MVV_LVA = [
    [0, 0, 0, 0, 0, 0, 0],  # None 
    [0,15 ,14 ,13 ,12 ,11 ,14],  # Pawn victim
    [0,25 ,24 ,23 ,22 ,21 ,24],  # Knight victim
    [0,35 ,34 ,33 ,32 ,31 ,34],  # Bishop victim
    [0,45 ,44 ,43 ,42 ,41 ,44],  # Rook victim
    [0,55 ,54 ,53 ,52 ,51 ,54], # Queen as victim
]  


def order_moves(moves):
    global MVV_LVA
   
    move_values = []

    # Assign values to each move
    for move in moves:
        value = 0
        attacker = board.piece_at(move.from_square)
        victim = board.piece_at(move.to_square)

        # MVV-LVA scoring for captures
        if board.is_capture(move):
            attacker_type = attacker.piece_type if attacker else None
            victim_type = victim.piece_type if victim else None
            if victim_type and attacker_type:
                value = MVV_LVA[victim_type][attacker_type]

        # Bonus for giving check
        if board.gives_check(move):
            value += 40



        # Penalize if the attacker moves to a square attacked by an enemy piece
        if attacker:
            move_square = move.to_square
            if any(board.is_attacked_by(color, move_square) for color in [not board.turn]):
                value -= attacker.piece_type * 2  

        # Bonus for promoting a pawn
        if move.promotion:
            promotion_values = {
                chess.QUEEN: 56,  
                chess.ROOK: 43,
                chess.BISHOP: 30,
                chess.KNIGHT: 30,
            }
            value += promotion_values.get(move.promotion)

        move_values.append((move, value))


    move_values.sort(key=lambda x: x[1], reverse=True)


    ordered_moves = [mv[0] for mv in move_values]
    return ordered_moves

def Quiescen(alpha, beta, nodes_left):
    global nodes_visited
    increment_node_visit() 
    increment_node_count()
    if nodes_visited >= nodes_left:
        return evaluate_position_with_ai(board)  # Return static evaluation when node limit is reached
    
    # Standpat evaluation
    stand_pat = evaluate_position_with_ai(board)
    best_value = stand_pat

    # Beta cutoff
    if stand_pat >= beta:
        return stand_pat

    # Update alpha
    if alpha < stand_pat:
        alpha = stand_pat

    # If the board is in check, consider all moves
    if board.is_check():
        loud_moves = board.legal_moves
    else:
        loud_moves = [
            move for move in board.legal_moves
            if board.is_capture(move) or board.gives_check(move)
        ]
    loud_moves = order_moves(loud_moves)

    for move in loud_moves:
        board.push(move)
        
        
        score = -Quiescen(-alpha, -beta, nodes_left)  
        board.pop()


        if score >= beta:
            return score

        # Update best value and alpha
        if score > best_value:
            best_value = score
        if score > alpha:
            alpha = score
        if best_value == 1:
            return best_value

    return best_value


def negamax(alpha, beta, depth):
    global node_count
    global nodes_visited
    increment_node_count()  # Increment the counter for every node visited
    if board.is_checkmate():
        return -1 #max output of ai is 1 and -1 so this will always be prefered
    # Check for draw
    if board.is_stalemate() or board.is_insufficient_material() or  board.is_fifty_moves() or board.can_claim_threefold_repetition():
        return 0
    
    if depth == 0:
        nodes_visited = 0
        return Quiescen(alpha, beta, nodes_left=300)

    best_value = -float('inf')
    moves = order_moves(board.legal_moves)
    for move in moves:
        board.push(move)
        score = -negamax(-beta, -alpha, depth - 1)
        board.pop()

        if score > best_value:
            best_value = score

        if score > alpha:
            alpha = score
        if alpha >= beta:
            break  
        if best_value == 1:
            return best_value

    return best_value


def get_best_move():
    best_score = -float('inf')
    best_move = None
    alpha = -float('inf')
    beta = float('inf')
    moves = order_moves(board.legal_moves)
    reset_node_count()  # Reset once for the entire search
    for move in moves:
        board.push(move)
        score = -negamax(-beta, -alpha, 1)
        board.pop()

        if score > best_score:
            best_score = score
            best_move = move

        if score > alpha:
            alpha = score

        print(f"info currmove {move.uci()} currscore {score:.4f} nodes {get_node_count()}")
        if best_score == 1:
            break

    adjusted_score = best_score * 20 * 100
    print(f"info score cp {int(adjusted_score)} pv {best_move}", flush=True)

    return best_move.uci()

def get_best_move_fast():
    best_score = -float('inf')
    best_move = None
    alpha = -float('inf')
    beta = float('inf')
    moves = order_moves(board.legal_moves)
    reset_node_count()
    for move in moves:
        board.push(move)
        score = -Quiescen(-beta, -alpha,400)  
        board.pop()

        if score > best_score:
            best_score = score
            best_move = move

        
        if score > alpha:
            alpha = score

        print(f"info currmove {move.uci()} currscore {score:.4f} nodes {get_node_count()}")
        if best_score == 1:
            break

    adjusted_score = best_score * 20 *100


    print(f"info score cp {int(adjusted_score)} pv {best_move}", flush=True)

    return best_move.uci()

def has_time(line, min_time=15):

    tokens = line.split(" ")

    wtime = btime = movetime = None

    # Extract the time parameters from the line
    for i in range(len(tokens)):
        if tokens[i] == "movetime":
            movetime = int(tokens[i + 1])
        elif tokens[i] == "wtime":
            wtime = int(tokens[i + 1])
        elif tokens[i] == "btime":
            btime = int(tokens[i + 1])
        elif tokens[i] == "infinite":
            return True

    
    if movetime is not None:
        return movetime / 1000.0 > min_time

    
    if wtime is not None and wtime / 1000.0 > min_time and board.turn == chess.WHITE:
        return True
    if btime is not None and btime / 1000.0 > min_time and board.turn == chess.BLACK:
        return True

    return False

def uci_loop():
    while True:
        try:
            line = input().strip()
            if line == "uci":
                print("id name AI-PyTorch")
                print("id author BrokenCodeGenrator")
                print("uciok", flush=True)
            elif line == "isready":
                print("readyok", flush=True)
            elif line.startswith("position"):
                tokens = line.split(" ")
                if tokens[1] == "startpos":
                    moves_index = tokens.index("moves") if "moves" in tokens else len(tokens)
                    board.reset()
                    moves = tokens[moves_index + 1:] if moves_index < len(tokens) else []
                    for move in moves:
                        board.push_uci(move)
                elif tokens[1] == "fen":
                    fen = " ".join(tokens[2:tokens.index("moves")] if "moves" in tokens else tokens[2:])
                    board.set_fen(fen)
                    moves = tokens[tokens.index("moves") + 1:] if "moves" in tokens else []
                    for move in moves:
                        board.push_uci(move)
            elif line.startswith("go"):
                if has_time(line):
                    best_move = get_best_move()
                else:
                    best_move = get_best_move_fast()
                print(f"bestmove {best_move}", flush=True)
            elif line == "quit":
                sys.exit()
            elif line =="fen_":
                print(board.fen)
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr, flush=True)

if __name__ == "__main__":
    uci_loop()

