import os
import json
import random
import torch
from torch.utils.data import Dataset
import chess
import Encoder  


def normalize(eval,eval_type):

    if eval_type == 'eval':  # handle type eval
        eval = eval / 20
        if eval > 0.9:
            return 0.9
        if eval < -0.9:
            return -0.9
        return eval

    mate_eval = 1 - (abs(eval)/100)
    if eval < 0:
        return -mate_eval
    return mate_eval


class ChessJSONLDataset(Dataset):
    """
    PyTorch Dataset for loading chess positions from multiple smaller JSONL files.
    """

    def __init__(self, data_dir):
        """
        Args:
            data_dir (str): Path to directory containing JSONL files.
        """
        self.data_dir = data_dir
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jsonl')]

    def __len__(self):
        """
        Total number of files in the dataset.
        """
        return len(self.files)

    def __getitem__(self, idx):
        """
        Extract a random position and its evaluation from the file.

        Args:
            idx (int): Index of the file to fetch.

        Returns:
            dict: Encoded position, evaluation, etc.
        """
        file_path = self.files[idx]
        try:
            with open(file_path, 'r') as file:
                lines = file.readlines()
                while True:  # Retry until valid data is found
                    line = random.choice(lines)  # Randomly pick one entry
                    data = json.loads(line)

                    eval_value = data.get('eval')
                    eval_type = data.get('type')
                    fen = data.get('fen')

                    if eval_type == "mate" and eval_value == 0:
                        continue  # Skip this line and pick another
                    elif eval_value is not None and fen:
                        normalized_eval = normalize(eval_value, eval_type)
                        board = chess.Board(fen)
                        position, value = Encoder.encodeTrainingPointWithoutPolicy(board, normalized_eval)

                        return {
                            'position': torch.from_numpy(position).float(),
                            'value': torch.tensor([value], dtype=torch.float32),
                        }
        except Exception as e:
            raise ValueError(f"Error processing file {file_path}: {e}")

        raise IndexError(f"Invalid index {idx}")

class ChessJSONLValDataset(Dataset):
    """
    PyTorch Dataset for loading all valid chess positions from multiple JSONL files.
    This is specifically for validation purposes, with no randomness or variation.
    """

    def __init__(self, data_dir):
        """
        Args:
            data_dir (str): Path to directory containing JSONL files.
        """
        self.data_dir = data_dir
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.jsonl')]
        self.data = self._load_all_data()

    def _load_all_data(self):
        """
        Load all valid positions from JSONL files into memory.

        Returns:
            list: A list of dictionaries containing positions and evaluations.
        """
        all_data = []

        for file_path in self.files:
            try:
                with open(file_path, 'r') as file:
                    for line in file:
                        data = json.loads(line)

                        eval_value = data.get('eval')
                        eval_type = data.get('type')
                        fen = data.get('fen')

                        if eval_type == "mate" and eval_value == 0:
                            continue  # Skip mate 0 positions

                        if eval_value is not None and fen:
                            normalized_eval = normalize(eval_value, eval_type)
                            board = chess.Board(fen)
                            position, value = Encoder.encodeTrainingPointWithoutPolicy(board, normalized_eval)

                            all_data.append({
                                'position': torch.from_numpy(position).float(),
                                'value': torch.tensor([value], dtype=torch.float32),
                            })

            except Exception as e:
                raise ValueError(f"Error processing file {file_path}: {e}")

        return all_data

    def __len__(self):
        """
        Total number of valid positions in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve the position and its evaluation at the specified index.

        Args:
            idx (int): Index of the position to fetch.

        Returns:
            dict: Encoded position, evaluation, etc.
        """
        if idx < 0 or idx >= len(self.data):
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self.data)}")

        return self.data[idx]
