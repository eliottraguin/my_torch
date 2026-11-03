"""Supervised training via backpropagation."""

import random
import sys
from hashlib import blake2b

import numpy as np

from .chess.chessencoder import ChessEncoder
from .chess.fen import FEN
from .neural_network import NeuralNetwork
from .early_stopping import EarlyStopping

_EPOCHS = 200
_REPORT_EVERY = 10


def _make_target(label: str, network: NeuralNetwork) -> np.ndarray:
    label_to_index = {lbl: i for i, lbl in enumerate(network.labels)}
    if label not in label_to_index:
        raise ValueError(f"Unknown label '{label}', expected one of {network.labels}")
    target = np.zeros(network.output_size, dtype=np.float32)
    target[label_to_index[label]] = 1.0
    return target

def _activation_backward(activation, z, gradient_output):
    derivative = activation.derivative(z)

    if activation.gradient:
        return derivative.T @ gradient_output

    return gradient_output * derivative

def _backprop_step(network: NeuralNetwork, encoded: np.ndarray, target: np.ndarray) -> float:
    activations, preactivations = network.forward_pass(encoded, return_preactivations=True)
    output = activations[-1]

    loss = 0.5 * float(np.sum((output - target) ** 2))

    gradient_output = output - target
    delta = _activation_backward(
        network.activation_for_layer(len(network.weights) - 1),
        preactivations[-1],
        gradient_output
    )

    for i in range(len(network.weights) - 1, -1, -1):
        dW = np.outer(delta, activations[i])
        db = delta.copy()
        if i > 0:
            gradient_previous = network.weights[i].T @ delta

            delta_next = _activation_backward(
                network.activation_for_layer(i - 1),
                preactivations[i - 1],
                gradient_previous,
            )
        network.weights[i] -= network.learning_rate * dW
        network.biases[i] -= network.learning_rate * db
        if i > 0:
            delta = delta_next

    return loss

def train_network(network: NeuralNetwork,
                  encoded_data: list[tuple[np.ndarray, np.ndarray]],
                  epoch: int, *, rng=None) -> float:
    """Train one epoch and return its mean online loss."""
    if not encoded_data:
        raise ValueError("Cannot train on an empty dataset")
    (rng if rng is not None else random).shuffle(encoded_data)
    total_loss = sum(_backprop_step(network, enc, tgt) for enc, tgt in encoded_data)
    if not np.isfinite(total_loss) or not all(
        np.isfinite(p).all() for p in network.weights + network.biases
    ):
        raise ValueError("Non-finite training result; previous checkpoint preserved")
    return total_loss / len(encoded_data)


def evaluate_loss(network, examples):
    """Evaluate mean quadratic loss without modifying the network."""
    if not examples:
        raise ValueError("Cannot evaluate an empty dataset")
    total = sum(
        0.5 * float(np.sum((network.forward_pass(encoded)[-1] - target) ** 2))
        for encoded, target in examples
    )
    loss = total / len(examples)
    if not np.isfinite(loss):
        raise ValueError("Non-finite evaluation loss; previous checkpoint preserved")
    return loss


def _prepare_examples(network, positions):
    labeled = []
    for p in positions:
        if p.label:
            labels = p.label[0]
            if len(p.label) != 1:
                labels = " ".join(p.label)
            labeled.append((p, labels))
    if not labeled:
        raise ValueError("No labeled positions found in training file.")
    examples = []
    for position, label in labeled:
        try:
            examples.append((ChessEncoder(position).encode(), _make_target(label, network)))
        except ValueError as error:
            print(f"Skipping position: {error}", file=sys.stderr)
    if not examples:
        raise ValueError("No usable training examples after encoding.")
    return examples


def _split_dataset(examples, fraction, rng):
    """Keep identical inputs together and split distinct groups within each class."""
    groups = {}
    for encoded, target in examples:
        key = blake2b(encoded.tobytes(), digest_size=32).digest()
        label = int(np.argmax(target))
        if key in groups and groups[key][0] != label:
            raise ValueError("Identical encoded positions have conflicting labels")
        groups.setdefault(key, (label, []))[1].append((encoded, target))
    by_label = {}
    for label, group in groups.values():
        by_label.setdefault(label, []).append(group)
    train, validation = [], []
    for label_groups in by_label.values():
        rng.shuffle(label_groups)
        count = min(len(label_groups) - 1, max(1, round(len(label_groups) * fraction)))
        for index, group in enumerate(label_groups):
            (validation if index < count else train).extend(group)
    return train, validation


def _snapshot(network):
    return ([w.copy() for w in network.weights], [b.copy() for b in network.biases])


def _report_epoch(epoch, train_loss, monitored_loss, metric):
    print(f"Epoch {epoch}/{_EPOCHS}  loss={train_loss:.8e}  "
          f"{metric}={monitored_loss:.8e}", file=sys.stderr)


def _run_epochs(network, train, validation, save_path, stopping, rng):
    """Manage complete-epoch checkpoints and restore the best model on normal exit."""
    monitored = validation if validation else train
    metric = "validation_loss" if validation else "training_loss"
    network.save(save_path)
    checkpoint = _snapshot(network)
    completed_epoch = 0
    try:
        stopping.update(evaluate_loss(network, monitored), network, epoch=0)
        for epoch in range(1, _EPOCHS + 1):
            train_loss = train_network(network, train, epoch, rng=rng)
            monitored_loss = evaluate_loss(network, monitored)
            should_stop = stopping.update(monitored_loss, network, epoch)
            network.save(save_path)
            checkpoint = _snapshot(network)
            completed_epoch = epoch
            if epoch == 1 or epoch % _REPORT_EVERY == 0 or should_stop:
                _report_epoch(epoch, train_loss, monitored_loss, metric)
            if should_stop:
                print(f"Early stopping: no improvement greater than {stopping.min_delta:g} "
                      f"for {stopping.patience} epochs.", file=sys.stderr)
                break
        stopping.restore(network)
        network.save(save_path)
    except KeyboardInterrupt:
        network.weights, network.biases = checkpoint
        network.save(save_path)
        print(f"Training interrupted; epoch {completed_epoch} saved to {save_path}",
              file=sys.stderr)
        raise SystemExit(130) from None
    print(f"Restored epoch {stopping.best_epoch}: "
          f"{metric}={stopping.best_loss:.8e}", file=sys.stderr)
    print(f"Network saved to {save_path}", file=sys.stderr)


def run_training(network: NeuralNetwork, positions: list[FEN], save_path: str,
                 *, patience=10, min_delta=1e-6, validation_fraction=0.2,
                 seed=42) -> None:
    """Prepare data, monitor held-out loss and save training checkpoints."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between zero and one")
    stopping = EarlyStopping(patience, min_delta)
    examples = _prepare_examples(network, positions)
    rng = random.Random(seed)
    train, validation = _split_dataset(examples, validation_fraction, rng)
    if len({int(np.argmax(target)) for _, target in examples}) < 2:
        print("Warning: only one class is present; low loss does not demonstrate "
              "classification across all labels.", file=sys.stderr)
    if not validation:
        print("Warning: too few distinct positions per class for validation; "
              "monitoring training loss only, without a generalization estimate.",
              file=sys.stderr)
    print(f"Training on {len(train)} examples, validating on {len(validation)}, "
          f"for at most {_EPOCHS} epochs (patience={patience}).", file=sys.stderr)
    _run_epochs(network, train, validation, save_path, stopping, rng)
