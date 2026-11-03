"""Encode chess positions and display predictions in input order."""

import sys

from .chess.chessencoder import ChessEncoder
from .chess.fen import FEN
from .neural_network import NeuralNetwork

def run_prediction(network: NeuralNetwork, positions: list[FEN]) -> None:
    """Predict without labels as input, then score against available labels."""
    correct = labeled = 0
    for index, position in enumerate(positions):
        encoded = ChessEncoder(position).encode()
        prediction, confidence = network.predict(encoded)
        print(f"{prediction}")
        if position.label:
            label = position.label[0]
            if len(position.label) != 1:
                label = " ".join(position.label)
            labeled += 1
            correct += prediction == label
