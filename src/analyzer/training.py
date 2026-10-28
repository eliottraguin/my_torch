"""Supervised training via backpropagation."""

import random
import sys

import numpy as np

from .chess.chessencoder import ChessEncoder
from .chess.fen import FEN
from .neural_network import NeuralNetwork

_EPOCHS = 200
_REPORT_EVERY = 10


def _make_target(label: str, network: NeuralNetwork) -> np.ndarray:
    label_to_index = {lbl: i for i, lbl in enumerate(network.labels)}
    if label not in label_to_index:
        raise ValueError(f"Unknown label '{label}', expected one of {network.labels}")
    target = np.zeros(network.output_size, dtype=np.float32)
    target[label_to_index[label]] = 1.0
    return target


def _backprop_step(network: NeuralNetwork, encoded: np.ndarray, target: np.ndarray) -> float:
    activations = network.forward_pass(encoded)
    output = activations[-1]

    loss = 0.5 * float(np.sum((output - target) ** 2))

    delta = (output - target) * output * (1.0 - output)

    for i in range(len(network.weights) - 1, -1, -1):
        dW = np.outer(delta, activations[i])
        db = delta.copy()
        if i > 0:
            delta_next = (
                np.dot(network.weights[i].T, delta)
                * activations[i]
                * (1.0 - activations[i])
            )
        network.weights[i] -= network.learning_rate * dW
        network.biases[i] -= network.learning_rate * db
        if i > 0:
            delta = delta_next

    return loss


def run_training(network: NeuralNetwork, positions: list[FEN], save_path: str) -> None:
    labeled = [
        (p, p.label[0]) for p in positions if p.label is not None
    ]
    if not labeled:
        raise ValueError("No labeled positions found in training file.")

    encoded_data = []
    for position, label in labeled:
        try:
            encoded = ChessEncoder(position).encode()
            target = _make_target(label, network)
            encoded_data.append((encoded, target))
        except ValueError as e:
            print(f"Skipping position: {e}", file=sys.stderr)

    if not encoded_data:
        raise ValueError("No usable training examples after encoding.")

    print(f"Training on {len(encoded_data)} examples for {_EPOCHS} epochs.", file=sys.stderr)

    for epoch in range(1, _EPOCHS + 1):
        random.shuffle(encoded_data)
        total_loss = sum(
            _backprop_step(network, enc, tgt) for enc, tgt in encoded_data
        )
        if epoch % _REPORT_EVERY == 0:
            print(
                f"Epoch {epoch}/{_EPOCHS}  loss={total_loss / len(encoded_data):.6f}",
                file=sys.stderr,
            )

    network.save(save_path)
    print(f"Network saved to {save_path}", file=sys.stderr)
