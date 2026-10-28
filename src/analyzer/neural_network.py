"""Represents a neural network loaded from a file."""

import json
import numpy as np


def sigmoid(x):
    """Apply sigmoid without overflow, preserving floating-point precision."""
    values = np.asarray(x)
    values = values.astype(np.result_type(values.dtype, np.float32), copy=False)
    with np.errstate(under="ignore"):
        exp_value = np.exp(-np.abs(values))
    return np.where(values >= 0, 1 / (1 + exp_value), exp_value / (1 + exp_value))

class NeuralNetwork:
    """A class to represent a neural network loaded from a file."""
    def __init__(self, filepath):
        self.filepath = filepath
        self.input_size = None
        self.output_size = None
        self.hidden_layers = None
        self.learning_rate = None
        self.labels = None
        self.weights = None
        self.biases = None

    @classmethod
    def load_from_file(cls, filepath):
        """Create a NeuralNetwork instance and load it from a file."""
        nn = cls(filepath)
        nn.load()
        return nn

    def load(self):
        """Load the neural network from a file."""
        try:
            with open(self.filepath, 'r') as file:
                data = json.load(file)
                self.input_size = data['layers'][0]
                self.output_size = data['layers'][-1]
                self.hidden_layers = data['layers'][1:-1]
                self.learning_rate = data['learning_rate']
                self.labels = data['labels']
                self.weights = [np.array(matrix) for matrix in data['weights']]
                self.biases = [np.array(vector) for vector in data['biases']]
        except (FileNotFoundError, KeyError, ValueError) as e:
            raise ValueError(f"Error loading neural network from {self.filepath}: {e}") from e

    def forward_pass(self, input_data):
        """Forward pass returning activations at every layer (including input)."""
        activations = [np.array(input_data, dtype=np.float32)]
        for weights, biases in zip(self.weights, self.biases):
            activations.append(sigmoid(np.dot(weights, activations[-1]) + biases))
        return activations

    def predict(self, input_data):
        """Predict the output for the given input data."""
        if len(input_data) != self.input_size:
            raise ValueError(f"Input data must have size {self.input_size}, got {len(input_data)}")
        activations = self.forward_pass(input_data)
        output = activations[-1]
        index_label = np.argmax(output)
        return self.labels[index_label], output[index_label]

    def save(self, filepath):
        """Save the neural network to a JSON file."""
        layers = [self.input_size] + list(self.hidden_layers) + [self.output_size]
        data = {
            "layers": layers,
            "learning_rate": self.learning_rate,
            "labels": self.labels,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, allow_nan=False)
            f.write('\n')