"""Regression checks for network initialization and JSON persistence."""

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from generator.configfile import ConfigFile
from generator.neuralfile import NeuralFile


class NeuralFileTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory(prefix="mytorch neuralfile ")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.config = {
            "name": str(self.directory / "network"),
            "layers": [4, 5, 3], "seed": 42, "init": 0.1,
            "learning_rate": 0.01,
            "labels": ["Checkmate", "Check", "Nothing"],
        }

    def network(self):
        path = self.directory / "network.conf"
        path.write_text(json.dumps(self.config))
        return NeuralFile(ConfigFile(path))

    def test_data_is_json_serializable_with_all_parameters(self):
        data = json.loads(json.dumps(self.network().get_data(), allow_nan=False))
        self.assertEqual(data["layers"], [4, 5, 3])
        self.assertEqual(data["labels"], self.config["labels"])
        self.assertEqual(data["learning_rate"], 0.01)
        self.assertEqual([np.shape(w) for w in data["weights"]], [(5, 4), (3, 5)])
        self.assertEqual([np.shape(b) for b in data["biases"]], [(5,), (3,)])

    def test_each_layer_has_float32_weights_and_zero_biases(self):
        weights, biases = self.network().get_weights_bias(42, 0.1, [4, 5, 3])
        self.assertEqual(len(weights), 2)
        self.assertEqual(len(biases), 2)
        for matrix in weights:
            self.assertEqual(matrix.dtype, np.float32)
            self.assertTrue(np.isfinite(matrix).all())
            self.assertTrue((np.abs(matrix) <= 0.1).all())
        for vector, size in zip(biases, [5, 3]):
            self.assertEqual(vector.dtype, np.float32)
            np.testing.assert_array_equal(vector, np.zeros(size))

    def test_fixed_seed_is_reproducible(self):
        self.assertEqual(self.network().get_data(), self.network().get_data())

    def test_three_calls_create_three_distinct_networks(self):
        network = self.network()
        paths = [Path(network.create_file()) for _ in range(3)]
        self.assertEqual([path.name for path in paths],
                         ["network_1.nn", "network_2.nn", "network_3.nn"])
        models = [json.loads(path.read_text()) for path in paths]
        self.assertEqual([model["seed"] for model in models], [42, 43, 44])
        for index, model in enumerate(models):
            self.assertEqual(model, network.get_data(seed=42 + index))
        self.assertNotEqual(models[0]["weights"], models[1]["weights"])
        self.assertNotEqual(models[1]["weights"], models[2]["weights"])

    def assert_save_failure(self, network):
        stdout, stderr = StringIO(), StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as error:
                network.create_file()
        self.assertEqual(error.exception.code, 84)
        self.assertEqual(stdout.getvalue(), "")
        self.assertIn("Error creating neural network file:", stderr.getvalue())

    def test_existing_model_is_preserved(self):
        path = self.directory / "network_1.nn"
        path.write_text("existing model\n")
        self.assert_save_failure(self.network())
        self.assertEqual(path.read_text(), "existing model\n")

    def test_nonfinite_data_creates_no_partial_file(self):
        self.config["learning_rate"] = float("nan")
        self.assert_save_failure(self.network())
        self.assertFalse((self.directory / "network_1.nn").exists())

    def test_invalid_initialization_creates_no_file(self):
        self.config["init"] = -0.1
        self.assert_save_failure(self.network())
        self.assertFalse((self.directory / "network_1.nn").exists())

    def test_unwritable_destination_reports_error(self):
        self.config["name"] = str(self.directory / "missing" / "network")
        self.assert_save_failure(self.network())
