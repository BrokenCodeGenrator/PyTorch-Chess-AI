import chess
import numpy as np
import torch

cuda = torch.cuda.is_available()


def encodePosition(board):
    """
    Optimized encoding of a chess position as a vector.
    The first 12 planes represent piece positions, the next 4 represent castling rights,
    and the final 12 encode attack maps for each piece type and color.
    """
    planes = np.zeros((28, 8, 8), dtype=np.float32)

    # Precompute rank and file arrays for indexing
    square_ranks = np.array([chess.square_rank(sq) for sq in chess.SQUARES])
    square_files = np.array([chess.square_file(sq) for sq in chess.SQUARES])

    # Piece positions (12 planes)
    for i, (piece, color) in enumerate([
        (chess.PAWN, chess.WHITE), (chess.PAWN, chess.BLACK),
        (chess.ROOK, chess.WHITE), (chess.ROOK, chess.BLACK),
        (chess.BISHOP, chess.WHITE), (chess.BISHOP, chess.BLACK),
        (chess.KNIGHT, chess.WHITE), (chess.KNIGHT, chess.BLACK),
        (chess.QUEEN, chess.WHITE), (chess.QUEEN, chess.BLACK),
        (chess.KING, chess.WHITE), (chess.KING, chess.BLACK),
    ]):
        positions = np.array(list(board.pieces(piece, color)))
        if positions.size > 0:  # Skip empty positions
            ranks = square_ranks[positions]
            files = square_files[positions]
            planes[i, ranks, files] = 1.0

    # Castling rights (4 planes)
    planes[12, :, :] = board.has_kingside_castling_rights(chess.WHITE)
    planes[13, :, :] = board.has_kingside_castling_rights(chess.BLACK)
    planes[14, :, :] = board.has_queenside_castling_rights(chess.WHITE)
    planes[15, :, :] = board.has_queenside_castling_rights(chess.BLACK)

    # Attack planes (12 planes)
    for i, (piece, color) in enumerate([
        (chess.PAWN, chess.WHITE), (chess.PAWN, chess.BLACK),
        (chess.ROOK, chess.WHITE), (chess.ROOK, chess.BLACK),
        (chess.BISHOP, chess.WHITE), (chess.BISHOP, chess.BLACK),
        (chess.KNIGHT, chess.WHITE), (chess.KNIGHT, chess.BLACK),
        (chess.QUEEN, chess.WHITE), (chess.QUEEN, chess.BLACK),
        (chess.KING, chess.WHITE), (chess.KING, chess.BLACK),
    ]):
        positions = np.array(list(board.pieces(piece, color)))
        if positions.size > 0:  # Skip empty positions
            for sq in positions:
                attacks = np.array(list(board.attacks(sq)))
                ranks = square_ranks[attacks]
                files = square_files[attacks]
                planes[16 + i, ranks, files] = 1.0

    return planes

def encodeTrainingPointWithoutPolicy(board, winner):

    # Flip everything if it's black's turn
    if not board.turn:
        board = board.mirror()
        winner *= -1

    positionPlanes = encodePosition(board)

    return positionPlanes, float(winner)

def encodePositionForInference(board):

    # Flip if black's turn
    if not board.turn:
        board = board.mirror()

    positionPlanes = encodePosition(board)

    return positionPlanes

def callNeuralNetworkForValue(board, neuralNetwork):

    position = encodePositionForInference(board)

    position = torch.from_numpy(position)[None, ...]

    if cuda:
        position = position.cuda()

    value, _ = neuralNetwork(position)  # No policy head, only value head

    value = value.cpu().numpy()[0, 0]

    return value





if __name__ == '__main__':
    def debugFenToTensor(fen):
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)

        board = chess.Board(fen)
        tensor = encodePositionForInference(board)

        print("FEN:", fen)
        print("Encoded Tensor Shape:", tensor.shape)
        print("Encoded Tensor:\n", tensor)

    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

    debugFenToTensor(fen)
