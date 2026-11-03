"""Activation values, independent gradient checks and persistence contracts."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import numpy as np

from analyzer.activation import ActivationFunction
from analyzer.neural_network import NeuralNetwork
from analyzer.training import _backprop_step
from generator.neuralfile import NeuralFile


NAMES = ("sigmoid", "relu", "silu")


def small_network(hidden="sigmoid", output="sigmoid"):
    network = NeuralNetwork("unused.nn")
    network.input_size, network.output_size = 2, 2
    network.hidden_layers = [3]
    network.learning_rate = 0.01
    network.labels = ["first", "second"]
    network.weights = [
        np.array([[0.2, -0.3], [-0.4, 0.1], [0.1, 0.2]], dtype=np.float64),
        np.array([[0.3, -0.2, 0.4], [-0.1, 0.2, 0.5]], dtype=np.float64),
    ]
    network.biases = [np.array([0.1, -0.3, 0.05]), np.array([0.2, 0.3])]
    network.activation_function = ActivationFunction(hidden)
    network.output_activation_function = ActivationFunction(output)
    return network


class ActivationTests(unittest.TestCase):
    def test_registry_exposes_function_and_derivative(self):
        for name in NAMES:
            with self.subTest(name=name):
                activation = ActivationFunction(name)
                self.assertEqual(activation.func_name, name)
                self.assertTrue(callable(activation.func))
                self.assertTrue(callable(activation.derivative))

    def test_unknown_names_are_rejected(self):
        for name in ("", "missing", "ReLU", None, [], 42):
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "Unknown activation"):
                    ActivationFunction(name)

    def test_relu_values_and_derivative_at_zero(self):
        activation = ActivationFunction("relu")
        np.testing.assert_array_equal(activation.func([-2, 0, 2]), [0, 0, 2])
        np.testing.assert_array_equal(activation.derivative([-2, 0, 2]), [0, 0, 1])

    def test_silu_known_values(self):
        activation = ActivationFunction("silu")
        np.testing.assert_allclose(activation.func([-2., 0., 2.]),
                                   [-0.2384058440442351, 0., 1.7615941559557646])
        self.assertEqual(float(activation.derivative(0.)), 0.5)

    def test_derivatives_match_finite_differences(self):
        # Avoid ReLU's nondifferentiable origin.
        values = np.array([-3., -0.5, 0.4, 2.3])
        epsilon = 1e-6
        for name in NAMES:
            with self.subTest(name=name):
                activation = ActivationFunction(name)
                numerical = (activation.func(values + epsilon)
                             - activation.func(values - epsilon)) / (2 * epsilon)
                np.testing.assert_allclose(activation.derivative(values), numerical,
                                           rtol=1e-6, atol=1e-8)

    def test_shape_dtype_and_input_are_preserved(self):
        for name in NAMES:
            for dtype in (np.float32, np.float64):
                with self.subTest(name=name, dtype=dtype):
                    activation = ActivationFunction(name)
                    values = np.array([[-2, -1], [0, 2]], dtype=dtype)
                    original = values.copy()
                    for operation in (activation.func, activation.derivative):
                        result = operation(values)
                        self.assertEqual(result.shape, values.shape)
                        self.assertEqual(result.dtype, dtype)
                        self.assertTrue(np.isfinite(result).all())
                    np.testing.assert_array_equal(values, original)

    def test_finite_extremes_do_not_overflow(self):
        for name in NAMES:
            for dtype in (np.float32, np.float64):
                with self.subTest(name=name, dtype=dtype):
                    maximum = np.finfo(dtype).max
                    values = np.array([-maximum, -1000, 0, 1000, maximum], dtype=dtype)
                    activation = ActivationFunction(name)
                    with np.errstate(all="raise"):
                        for operation in (activation.func, activation.derivative):
                            self.assertTrue(np.isfinite(operation(values)).all())


class NetworkActivationTests(unittest.TestCase):
    def test_forward_pass_keeps_its_existing_return_contract(self):
        network = small_network("silu")
        activations = network.forward_pass([0.7, -0.2])
        self.assertEqual([a.shape for a in activations], [(2,), (3,), (2,)])

    def test_forward_pass_exposes_preactivations_for_derivatives(self):
        network = small_network("relu")
        activations, zs = network.forward_pass([0.7, -0.2], return_preactivations=True)
        self.assertEqual([z.shape for z in zs], [(3,), (2,)])
        np.testing.assert_allclose(zs[0], network.weights[0] @ activations[0] + network.biases[0])
        np.testing.assert_allclose(activations[1], np.maximum(0, zs[0]))
        self.assertEqual(network.activation_for_layer(0).func_name, "relu")
        self.assertEqual(network.activation_for_layer(1).func_name, "sigmoid")

    def test_backprop_gradients_match_independent_finite_differences(self):
        encoded = np.array([0.7, -0.2], dtype=np.float32)
        target = np.array([0., 1.])
        epsilon = 1e-6

        def loss(network):
            output = network.forward_pass(encoded)[-1]
            return 0.5 * np.sum((output - target) ** 2)

        for hidden in NAMES:
            for output in NAMES:
                with self.subTest(hidden=hidden, output=output):
                    network = small_network(hidden, output)
                    parameters = network.weights + network.biases
                    originals = [p.copy() for p in parameters]
                    gradients = []
                    for parameter in parameters:
                        gradient = np.empty_like(parameter)
                        for index in np.ndindex(parameter.shape):
                            original = parameter[index]
                            parameter[index] = original + epsilon
                            upper = loss(network)
                            parameter[index] = original - epsilon
                            lower = loss(network)
                            parameter[index] = original
                            gradient[index] = (upper - lower) / (2 * epsilon)
                        gradients.append(gradient)
                    initial_loss = loss(network)
                    returned_loss = _backprop_step(network, encoded, target)
                    self.assertAlmostEqual(returned_loss, initial_loss)
                    for before, after, expected in zip(originals, parameters, gradients):
                        actual = (before - after) / network.learning_rate
                        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
                    self.assertLess(loss(network), initial_loss)

    def test_save_load_preserves_activation_choices_and_predictions(self):
        with TemporaryDirectory(prefix="mytorch activations ") as directory:
            path = Path(directory) / "network.nn"
            for name in NAMES:
                with self.subTest(name=name):
                    network = small_network(name)
                    network.save(path)
                    restored = NeuralNetwork.load_from_file(path)
                    self.assertEqual(restored.activation_function.func_name, name)
                    self.assertEqual(restored.output_activation_function.func_name, "sigmoid")
                    np.testing.assert_array_equal(restored.forward_pass([0.7, -0.2])[-1],
                                                  network.forward_pass([0.7, -0.2])[-1])

    def test_legacy_models_default_to_sigmoid(self):
        with TemporaryDirectory(prefix="mytorch legacy activation ") as directory:
            path = Path(directory) / "network.nn"
            network = small_network()
            network.save(path)
            data = json.loads(path.read_text())
            del data["hidden_activation"], data["output_activation"]
            path.write_text(json.dumps(data))
            restored = NeuralNetwork.load_from_file(path)
            self.assertEqual(restored.activation_function.func_name, "sigmoid")
            self.assertEqual(restored.output_activation_function.func_name, "sigmoid")
            np.testing.assert_array_equal(restored.forward_pass([0.7, -0.2])[-1],
                                          network.forward_pass([0.7, -0.2])[-1])

    def test_unknown_saved_activation_is_rejected(self):
        with TemporaryDirectory(prefix="mytorch bad activation ") as directory:
            path = Path(directory) / "network.nn"
            small_network().save(path)
            data = json.loads(path.read_text())
            data["hidden_activation"] = "missing"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "Unknown activation"):
                NeuralNetwork.load_from_file(path)

    def test_generator_stores_configured_activations(self):
        config = {"layers": [2, 3, 2], "seed": 42, "init": 0.1,
                  "learning_rate": 0.01, "labels": ["first", "second"],
                  "hidden_activation": "silu", "output_activation": "sigmoid"}
        model = NeuralFile(SimpleNamespace(get_config=lambda: config)).get_data()
        self.assertEqual(model["hidden_activation"], "silu")
        self.assertEqual(model["output_activation"], "sigmoid")
        with TemporaryDirectory(prefix="mytorch generated activation ") as directory:
            path = Path(directory) / "network.nn"
            path.write_text(json.dumps(model))
            restored = NeuralNetwork.load_from_file(path)
            self.assertTrue(np.isfinite(restored.forward_pass([0.7, -0.2])[-1]).all())
