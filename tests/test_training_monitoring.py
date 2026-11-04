"""Observer metrics and cooperative stopping preserve CLI training behavior."""

from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch

import numpy as np

from analyzer.chess.fen import FEN
from analyzer.metrics import classification_metrics, evaluate_metrics
from analyzer.neural_network import NeuralNetwork
from analyzer.training import run_training
from tests.support import write_test_network
from tests.test_activation import small_network


class MonitoringTests(unittest.TestCase):
    def test_metrics_use_true_rows_and_predicted_columns(self):
        result = classification_metrics([[3, 1], [2, 4]], ['a', 'b'])
        self.assertEqual(result['count'], 10)
        self.assertAlmostEqual(result['accuracy'], 0.7)
        self.assertAlmostEqual(result['per_class'][0]['precision'], 0.6)
        self.assertAlmostEqual(result['per_class'][0]['recall'], 0.75)
        empty = classification_metrics([[0, 0], [0, 0]], ['a', 'b'])
        self.assertIsNone(empty['accuracy'])

    def test_evaluation_does_not_update_parameters(self):
        net = small_network()
        before = [p.copy() for p in net.weights + net.biases]
        result = evaluate_metrics(net, [(np.array([.7, -.2]), np.array([1., 0.]))])
        self.assertEqual(result['count'], 1)
        for old, new in zip(before, net.weights + net.biases):
            np.testing.assert_array_equal(old, new)

    def test_observer_gets_every_epoch_and_final_restoration(self):
        with TemporaryDirectory() as directory, redirect_stderr(StringIO()):
            initial, output = Path(directory) / 'initial.nn', Path(directory) / 'trained.nn'
            write_test_network(initial)
            network = NeuralNetwork.load_from_file(initial)
            positions = [FEN('8/8/8/8/8/8/8/K6k w - - 0 1 Nothing').parse()]
            records = []
            run_training(network, positions, output, max_epochs=2, observer=records.append)
            epochs = [r for r in records if r['event'] == 'epoch']
            self.assertEqual([r['epoch'] for r in epochs], [0, 1, 2])
            self.assertEqual(records[-1]['event'], 'completed')
            self.assertTrue(all(r['validation'] is None for r in epochs))
            best = min(r['train']['loss'] for r in epochs)
            loaded = NeuralNetwork.load_from_file(output)
            from analyzer.chess.chessencoder import ChessEncoder
            target = np.array([0., 0., 1.])
            measured = evaluate_metrics(loaded, [(ChessEncoder(positions[0]).encode(), target)])
            self.assertAlmostEqual(measured['loss'], best)

    def test_cooperative_stop_retains_last_complete_epoch(self):
        with TemporaryDirectory() as directory, redirect_stderr(StringIO()):
            initial, output = Path(directory) / 'initial.nn', Path(directory) / 'trained.nn'
            write_test_network(initial)
            network = NeuralNetwork.load_from_file(initial)
            positions = [FEN('8/8/8/8/8/8/8/K6k w - - 0 1 Nothing').parse()]
            stop = threading.Event()
            records = []
            def observe(record):
                records.append(record)
                if record['event'] == 'epoch' and record['epoch'] == 1:
                    stop.set()
            run_training(network, positions, output, max_epochs=10,
                         observer=observe, should_stop=stop.is_set)
            self.assertEqual(records[-1]['event'], 'stopped')
            self.assertEqual(records[-1]['epoch'], 1)
            self.assertTrue(output.is_file())

    def test_external_validation_overlap_is_rejected_before_saving(self):
        with TemporaryDirectory() as directory:
            initial, output = Path(directory) / 'initial.nn', Path(directory) / 'trained.nn'
            write_test_network(initial)
            network = NeuralNetwork.load_from_file(initial)
            positions = [FEN('8/8/8/8/8/8/8/K6k w - - 0 1 Nothing').parse()]
            with self.assertRaisesRegex(ValueError, 'identical encoded positions'):
                run_training(network, positions, output, validation_positions=positions)
            self.assertFalse(output.exists())

    def test_default_cli_path_does_not_compute_extra_accuracy_metrics(self):
        with TemporaryDirectory() as directory, redirect_stderr(StringIO()):
            initial = Path(directory) / 'initial.nn'
            write_test_network(initial)
            positions = [FEN('8/8/8/8/8/8/8/K6k w - - 0 1 Nothing').parse()]
            with patch('analyzer.training.evaluate_metrics') as metrics:
                run_training(NeuralNetwork.load_from_file(initial), positions,
                             Path(directory) / 'out.nn', max_epochs=1)
            metrics.assert_not_called()
