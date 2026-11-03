"""Interrupted training and atomic model persistence."""

from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from analyzer.neural_network import NeuralNetwork
from analyzer.training import run_training
from tests.test_activation import small_network


class CheckpointTests(unittest.TestCase):
    def test_ctrl_c_restores_and_saves_last_completed_epoch(self):
        net = small_network()
        positions = [SimpleNamespace(label=['first'])]
        initial = net.weights[0].copy()
        steps = 0

        def train_step(network, encoded, target):
            nonlocal steps
            steps += 1
            network.weights[0] += 1
            if steps == 2:
                raise KeyboardInterrupt
            return 0.1

        with TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.nn'
            with patch('analyzer.training.ChessEncoder') as encoder, \
                 patch('analyzer.training._backprop_step', side_effect=train_step), \
                 redirect_stderr(StringIO()) as stderr:
                encoder.return_value.encode.return_value = np.zeros(2)
                with self.assertRaises(SystemExit) as error:
                    run_training(net, positions, path)
            self.assertEqual(error.exception.code, 130)
            restored = NeuralNetwork.load_from_file(path)
            np.testing.assert_array_equal(restored.weights[0], initial + 1)
            np.testing.assert_array_equal(net.weights[0], initial + 1)
            self.assertIn('epoch 1 saved', stderr.getvalue())

    def test_failure_keeps_checkpoint_and_partial_serialization_preserves_file(self):
        net = small_network()
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.nn'
            net.save(path)
            original = path.read_bytes()
            net.weights[0][0, 0] = np.nan
            with self.assertRaises(ValueError):
                net.save(path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_interruption_during_serialization_preserves_file_and_cleans_temp(self):
        net = small_network()
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.nn'
            net.save(path)
            original = path.read_bytes()
            with patch('analyzer.neural_network.json.dump', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    net.save(path)
            self.assertEqual(path.read_bytes(), original)
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_training_error_leaves_initial_checkpoint_readable(self):
        net = small_network()
        positions = [SimpleNamespace(label=['first'])]
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.nn'
            with patch('analyzer.training.ChessEncoder') as encoder, \
                 patch('analyzer.training._backprop_step', side_effect=ValueError('failed')), \
                 redirect_stderr(StringIO()):
                encoder.return_value.encode.return_value = np.zeros(2)
                with self.assertRaisesRegex(ValueError, 'failed'):
                    run_training(net, positions, path)
            restored = NeuralNetwork.load_from_file(path)
            np.testing.assert_array_equal(restored.weights[0], net.weights[0])
