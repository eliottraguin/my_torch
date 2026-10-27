"""Create JSON neural network files from a generator configuration."""

import json
import sys
from typing import Any

import numpy as np

from generator.configfile import ConfigFile


class NeuralFile:
    """Initialize networks and save each one to a distinct file."""

    def __init__(self, config_file: ConfigFile):
        self.config_data: dict[str, Any] = config_file.get_config()
        self.neural_data = self.load_neural()
        self._next_index = 1

    def get_data(self, seed=None):
        if seed is None:
            seed = self.config_data["seed"]
        weights, biases = self.get_weights_bias(
            seed=seed,
            init=self.config_data["init"],
            layers=self.config_data["layers"],
        )
        return {
            "layers": self.config_data["layers"],
            "learning_rate": self.config_data["learning_rate"],
            "labels": self.config_data["labels"],
            "seed": seed,
            "weights": [matrix.tolist() for matrix in weights],
            "biases": [vector.tolist() for vector in biases],
        }

    def get_weights_bias(self, seed, init, layers):
        rng = np.random.default_rng(seed)
        weights = []
        biases = []
        for input_size, output_size in zip(layers[:-1], layers[1:]):
            matrix = rng.uniform(
                -init, init, size=(output_size, input_size),
            ).astype(np.float32)
            weights.append(matrix)
            biases.append(np.zeros(output_size, dtype=np.float32))
        return weights, biases

    def load_neural(self):
        """Reserved for a future model-loading implementation."""
        return None

    def create_file(self):
        try:
            filename = f"{self.config_data['name']}_{self._next_index}.nn"
            seed = self.config_data["seed"] + self._next_index - 1
            content = json.dumps(self.get_data(seed=seed), indent=2, allow_nan=False)
            with open(filename, "x", encoding="utf-8") as file:
                file.write(content + "\n")
        except (OSError, TypeError, ValueError, KeyError) as error:
            print(f"Error creating neural network file: {error}", file=sys.stderr)
            raise SystemExit(84) from error
        self._next_index += 1
        return filename
