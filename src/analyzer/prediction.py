"""Encode chess positions and display predictions in input order."""

import sys

from .chess.chessencoder import ChessEncoder
from .chess.fen import FEN
from .neural_network import NeuralNetwork


def _print_accuracy(correct: int, labeled: int, total: int) -> None:
    """Report accuracy on labeled records only, separately from predictions."""
    if labeled:
        print(f"Prediction accuracy: {correct}/{labeled} correct "
              f"({100 * correct / labeled:.2f}%); "
              f"{total - labeled} unlabeled position(s) excluded.", file=sys.stderr)


def run_prediction(network: NeuralNetwork, positions: list[FEN]) -> None:
    """Predict without labels as input, then score against available labels."""
    correct = labeled = 0
    for index, position in enumerate(positions):
        encoded = ChessEncoder(position).encode()
        prediction, confidence = network.predict(encoded)
        print(f"FEN game {index}: Prediction: {prediction}, Confidence: {confidence}")
        if position.label:
            labeled += 1
            correct += prediction == position.label[0]
    _print_accuracy(correct, labeled, len(positions))
