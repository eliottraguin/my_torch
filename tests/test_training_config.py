"""Training settings survive generation, loading, saving and actual training."""

from contextlib import redirect_stderr
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from analyzer.chess.fen import FEN
from analyzer.neural_network import NeuralNetwork
from analyzer.training import run_training
from generator.neuralfile import NeuralFile
from training_config import training_settings


class TrainingConfigTests(unittest.TestCase):
    def data(self, **settings):
        config = dict(name='test', layers=[838, 3], seed=42, init=.1,
                      learning_rate=.01, labels=['Checkmate', 'Check', 'Nothing'], **settings)
        return NeuralFile(SimpleNamespace(get_config=lambda: config)).get_data()

    def load(self, directory, data):
        path = Path(directory) / 'initial.nn'
        path.write_text(json.dumps(data))
        return NeuralNetwork.load_from_file(path)

    def test_generation_and_round_trip_preserve_custom_values(self):
        with TemporaryDirectory() as directory:
            data = self.data(max_epochs=37, patience=4, min_delta=.0002)
            network = self.load(directory, data)
            self.assertEqual((network.max_epochs, network.patience, network.min_delta), (37, 4, .0002))
            path = Path(directory) / 'saved.nn'
            network.save(path)
            saved = json.loads(path.read_text())
            self.assertEqual(training_settings(saved), training_settings(data))

    def test_legacy_models_keep_previous_defaults(self):
        with TemporaryDirectory() as directory:
            data = self.data()
            for field in ('max_epochs', 'patience', 'min_delta'):
                data.pop(field)
            network = self.load(directory, data)
            self.assertEqual((network.max_epochs, network.patience, network.min_delta), (200, 10, 1e-6))

    def test_invalid_values_are_rejected_by_generator_and_loader(self):
        cases = [('max_epochs', 0), ('max_epochs', -1), ('max_epochs', True), ('max_epochs', 2.5),
                 ('patience', 0), ('patience', False), ('patience', '2'),
                 ('min_delta', -1), ('min_delta', float('nan')), ('min_delta', float('inf')),
                 ('min_delta', '0.1'), ('min_delta', True)]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(ValueError, field):
                    self.data(**{field: value})
                with TemporaryDirectory() as directory:
                    data = self.data(); data[field] = value
                    with self.assertRaisesRegex(ValueError, field):
                        self.load(directory, data)
        self.assertEqual(training_settings({'min_delta': 0})['min_delta'], 0.)

    def test_model_max_epochs_is_used_without_api_override(self):
        with TemporaryDirectory() as directory, redirect_stderr(StringIO()):
            network = self.load(directory, self.data(max_epochs=2))
            positions = [FEN('8/8/8/8/8/8/8/K6k w - - 0 1 Nothing').parse()]
            records = []
            run_training(network, positions, Path(directory) / 'trained.nn', observer=records.append)
            self.assertEqual([r['epoch'] for r in records if r['event'] == 'epoch'], [0, 1, 2])

    def test_model_patience_and_min_delta_control_stopping(self):
        with TemporaryDirectory() as directory, redirect_stderr(StringIO()):
            network = self.load(directory, self.data(max_epochs=20, patience=2, min_delta=.2))
            positions = [FEN('8/8/8/8/8/8/8/K6k w - - 0 1 Nothing').parse()]
            with patch('analyzer.training.evaluate_loss', side_effect=[1., .9, .85]), \
                 patch('analyzer.training.train_network', return_value=.5) as train:
                run_training(network, positions, Path(directory) / 'trained.nn')
            self.assertEqual(train.call_count, 2)

    def test_explicit_api_overrides_are_saved_in_output(self):
        with TemporaryDirectory() as directory, redirect_stderr(StringIO()):
            network = self.load(directory, self.data(max_epochs=20, patience=2, min_delta=.2))
            positions = [FEN('8/8/8/8/8/8/8/K6k w - - 0 1 Nothing').parse()]
            output = Path(directory) / 'trained.nn'
            run_training(network, positions, output, max_epochs=1, patience=3, min_delta=.01)
            self.assertEqual(training_settings(json.loads(output.read_text())),
                             {'max_epochs': 1, 'patience': 3, 'min_delta': .01})
