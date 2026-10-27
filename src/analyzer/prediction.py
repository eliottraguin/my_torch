"""Encode chess positions and display predictions in input order."""

from .chess.chessencoder import ChessEncoder
from .chess.fen import FEN
from .neural_network import NeuralNetwork


def run_prediction(network: NeuralNetwork, positions: list[FEN]) -> None:
    """Predict each position without using its optional expected label."""
    for index, position in enumerate(positions):
        encoded = ChessEncoder(position).encode()
        prediction, confidence = network.predict(encoded)
        print(f"FEN game {index}: Prediction: {prediction}, Confidence: {confidence}")
