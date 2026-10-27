"""Dedicated entry point for the future supervised training workflow."""

from .chess.fen import FEN
from .neural_network import NeuralNetwork


def run_training(network: NeuralNetwork, positions: list[FEN], save_path: str) -> None:
    """Receive training inputs and destination; learning is not implemented yet."""
    raise NotImplementedError("Training is not implemented yet; use --predict.")
