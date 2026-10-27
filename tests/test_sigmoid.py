"""Numerical regression tests for sigmoid and its prediction call site."""

import unittest

import numpy as np

from analyzer.neural_network import NeuralNetwork, sigmoid


class SigmoidTests(unittest.TestCase):
    def test_known_values(self):
        result = sigmoid([-np.log(3), 0, np.log(3)])
        np.testing.assert_allclose(result, [0.25, 0.5, 0.75], rtol=1e-14)

    def test_agrees_with_standard_formula_in_safe_range(self):
        values = np.linspace(-10, 10, 101)
        np.testing.assert_allclose(sigmoid(values), 1 / (1 + np.exp(-values)))

    def test_extreme_values_do_not_overflow(self):
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                maximum = np.finfo(dtype).max
                values = np.array([-maximum, -1000, 0, 1000, maximum], dtype=dtype)
                with np.errstate(all="raise"):
                    result = sigmoid(values)
                np.testing.assert_array_equal(result, [0, 0, 0.5, 1, 1])
                self.assertEqual(result.dtype, dtype)

    def test_infinite_limits(self):
        with np.errstate(all="raise"):
            result = sigmoid([-np.inf, np.inf])
        np.testing.assert_array_equal(result, [0, 1])

    def test_scalars_and_integer_arrays(self):
        self.assertEqual(float(sigmoid(0)), 0.5)
        values = np.array([np.iinfo(np.int64).min, 0, np.iinfo(np.int64).max])
        with np.errstate(all="raise"):
            result = sigmoid(values)
        np.testing.assert_array_equal(result, [0, 0.5, 1])

    def test_preserves_array_shape_and_float_dtype(self):
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                values = np.array([[-1, 0], [1, 2]], dtype=dtype)
                original = values.copy()
                result = sigmoid(values)
                self.assertEqual(result.shape, (2, 2))
                self.assertEqual(result.dtype, dtype)
                np.testing.assert_array_equal(values, original)
                self.assertTrue(np.isfinite(result).all())
                self.assertTrue(((result >= 0) & (result <= 1)).all())

    def test_is_monotonic_and_symmetric(self):
        values = np.linspace(-20, 20, 101)
        result = sigmoid(values)
        self.assertTrue((np.diff(result) >= 0).all())
        np.testing.assert_allclose(result + sigmoid(-values), 1)

    def test_prediction_handles_extreme_logits(self):
        network = NeuralNetwork("unused.nn")
        network.input_size = 2
        network.labels = ["Checkmate", "Check", "Nothing"]
        network.weights = [np.array([[1000, -1000], [0, 0], [-1000, 1000]],
                                    dtype=np.float32)]
        network.biases = [np.zeros(3, dtype=np.float32)]
        with np.errstate(all="raise"):
            label, score = network.predict([1, 0])
        self.assertEqual(label, "Checkmate")
        self.assertEqual(score, 1)
