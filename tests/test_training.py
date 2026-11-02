"""Stopping decisions, held-out data and restoration of saved parameters."""

from contextlib import redirect_stderr
from io import StringIO
from types import SimpleNamespace
import random
import unittest
from unittest.mock import patch

import numpy as np

from analyzer.early_stopping import EarlyStopping
from analyzer.training import _split_dataset, evaluate_loss, run_training, train_network
from tests.test_activation import small_network


class EarlyStoppingTests(unittest.TestCase):
    def test_plateau_stops_and_restores_independent_best_checkpoint(self):
        net = small_network()
        stopping = EarlyStopping(patience=2)
        original = net.weights[0].copy()
        self.assertFalse(stopping.update(0.2, net, 0))
        net.weights[0][:] = 9
        self.assertFalse(stopping.update(0.3, net, 1))
        self.assertTrue(stopping.update(0.4, net, 2))
        stopping.restore(net)
        np.testing.assert_array_equal(net.weights[0], original)
        self.assertEqual(stopping.best_epoch, 0)

    def test_small_improvements_accumulate_and_best_is_always_retained(self):
        net = small_network()
        stopping = EarlyStopping(patience=2, min_delta=0.1)
        self.assertFalse(stopping.update(1., net, 0))
        self.assertFalse(stopping.update(0.95, net, 1))
        self.assertFalse(stopping.update(0.85, net, 2))
        self.assertFalse(stopping.update(0.84, net, 3))
        self.assertTrue(stopping.update(0.83, net, 4))
        self.assertEqual(stopping.best_epoch, 4)

    def test_rejects_nonfinite_loss(self):
        for value in (float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'Non-finite'):
                EarlyStopping().update(value, small_network(), 0)


class TrainingTests(unittest.TestCase):
    def test_evaluation_matches_loss_without_changing_parameters(self):
        net = small_network()
        x, target = np.array([0.7, -0.2]), np.array([0., 1.])
        original = [p.copy() for p in net.weights + net.biases]
        expected = 0.5 * np.sum((net.forward_pass(x)[-1] - target) ** 2)
        self.assertAlmostEqual(evaluate_loss(net, [(x, target)] * 2), expected)
        for before, after in zip(original, net.weights + net.biases):
            np.testing.assert_array_equal(before, after)

    def test_split_is_reproducible_and_keeps_duplicates_together(self):
        examples = [(np.array([float(i)]), np.eye(2)[i % 2])
                    for i in range(10)] * 2
        train, valid = _split_dataset(examples, 0.2, random.Random(42))
        train_ids = {x.tobytes() for x, _ in train}
        valid_ids = {x.tobytes() for x, _ in valid}
        self.assertFalse(train_ids & valid_ids)
        self.assertEqual(len(train) + len(valid), len(examples))
        self.assertEqual({int(y.argmax()) for _, y in valid}, {0, 1})
        again = _split_dataset(examples, 0.2, random.Random(42))[1]
        self.assertEqual([x.tobytes() for x, _ in valid], [x.tobytes() for x, _ in again])

    def test_split_rejects_contradictory_labels(self):
        examples = [(np.array([1.]), y) for y in np.eye(2)]
        with self.assertRaisesRegex(ValueError, 'conflicting labels'):
            _split_dataset(examples, 0.2, random.Random(42))

    def test_training_stops_saves_best_and_excludes_validation_from_updates(self):
        net = small_network()
        original = net.weights[0].copy()
        examples = [(np.array([float(i), 0.]), np.array([1., 0.])) for i in range(5)]
        positions = [SimpleNamespace(label=['first']) for _ in examples]
        stderr = StringIO()

        def step(network, encoded, target):
            network.weights[0][:] = 99
            return 0.5

        with patch('analyzer.training.ChessEncoder') as encoder, \
             patch('analyzer.training.evaluate_loss', side_effect=[0.1, 0.2, 0.3]) as evaluate, \
             patch('analyzer.training._backprop_step', side_effect=step) as backprop, \
             patch.object(net, 'save') as save, redirect_stderr(stderr):
            encoder.return_value.encode.side_effect = [x for x, _ in examples]
            run_training(net, positions, 'unused.nn', patience=2)
        self.assertEqual(backprop.call_count, 8)
        validation_ids = {x.tobytes() for x, _ in evaluate.call_args.args[1]}
        self.assertTrue(all(call.args[1].tobytes() not in validation_ids
                            for call in backprop.call_args_list))
        np.testing.assert_array_equal(net.weights[0], original)
        self.assertEqual(save.call_count, 4)  # Initial, two complete epochs, best model.
        self.assertTrue(all(call.args == ('unused.nn',) for call in save.call_args_list))
        self.assertIn('Early stopping', stderr.getvalue())
        self.assertIn('Restored epoch 0', stderr.getvalue())

    def test_nonfinite_evaluation_keeps_only_initial_checkpoint(self):
        net = small_network()
        positions = [SimpleNamespace(label=['first'])]
        with patch('analyzer.training.ChessEncoder') as encoder, \
             patch('analyzer.training.evaluate_loss', side_effect=ValueError('Non-finite')), \
             patch.object(net, 'save') as save, redirect_stderr(StringIO()):
            encoder.return_value.encode.return_value = np.zeros(2)
            with self.assertRaisesRegex(ValueError, 'Non-finite'):
                run_training(net, positions, 'unused.nn')
        save.assert_called_once_with('unused.nn')

    def test_train_network_returns_mean_loss_above_one_without_stopping(self):
        net = small_network()
        examples = [(np.zeros(2), np.array([1., 0.]))] * 3
        with patch('analyzer.training._backprop_step', side_effect=[2., 3., 4.]):
            self.assertEqual(train_network(net, examples, 1), 3.)

    def test_max_epochs_saves_best_model_even_after_a_temporary_increase(self):
        net = small_network()
        initial = net.weights[0].copy()
        positions = [SimpleNamespace(label=['first'])]
        saved = []

        def step(network, examples, epoch, **kwargs):
            network.weights[0] += 1
            return float(epoch)

        with patch('analyzer.training.ChessEncoder') as encoder, \
             patch('analyzer.training._EPOCHS', 2), \
             patch('analyzer.training.train_network', side_effect=step) as train, \
             patch('analyzer.training.evaluate_loss', side_effect=[2., 0.5, 0.6]), \
             patch.object(net, 'save', side_effect=lambda path: saved.append(net.weights[0].copy())), \
             redirect_stderr(StringIO()):
            encoder.return_value.encode.return_value = np.zeros(2)
            run_training(net, positions, 'unused.nn')
        self.assertEqual(train.call_count, 2)
        self.assertEqual(len(saved), 4)
        np.testing.assert_array_equal(saved[2], initial + 2)
        np.testing.assert_array_equal(saved[-1], initial + 1)
        np.testing.assert_array_equal(net.weights[0], initial + 1)
