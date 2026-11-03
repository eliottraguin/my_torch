"""Represents a neural network loaded from a file."""

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
import numpy as np

from .activation import ActivationFunction

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
        self.activation_function = ActivationFunction("sigmoid")
        self.output_activation_function = ActivationFunction("sigmoid")

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
                self.activation_function = ActivationFunction(data.get("hidden_activation", "sigmoid"))
                self.output_activation_function = ActivationFunction(data.get("output_activation", "sigmoid"))
        except (FileNotFoundError, KeyError, ValueError) as e:
            raise ValueError(f"Error loading neural network from {self.filepath}: {e}") from e

    def activation_for_layer(self, index):
        """Select the output activation for the last layer, otherwise the hidden one."""
        if index == len(self.weights) - 1:
            return self.output_activation_function
        return self.activation_function

    def forward_pass(self, input_data, *, return_preactivations=False):
        """Return activations, optionally with z values required by backpropagation."""
        activations = [np.array(input_data, dtype=np.float32)]
        preactivations = []
        for index, (weights, biases) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(weights, activations[-1]) + biases
            preactivations.append(z)
            activations.append(self.activation_for_layer(index).func(z))
        if return_preactivations:
            return activations, preactivations
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
        """Replace the model atomically after writing a complete JSON file."""
        layers = [self.input_size] + list(self.hidden_layers) + [self.output_size]
        data = {
            "layers": layers,
            "learning_rate": self.learning_rate,
            "labels": self.labels,
            "hidden_activation": self.activation_function.func_name,
            "output_activation": self.output_activation_function.func_name,
            "weights": [w.tolist() for w in self.weights],
            "biases": [b.tolist() for b in self.biases],
        }
        destination = Path(filepath)
        temporary = None
        try:
            with NamedTemporaryFile(mode='w', encoding='utf-8',
                                    dir=destination.parent,
                                    prefix=f'.{destination.name}.',
                                    suffix='.tmp', delete=False) as f:
                temporary = Path(f.name)
                json.dump(data, f, indent=2, allow_nan=False)
                f.write('\n')
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
