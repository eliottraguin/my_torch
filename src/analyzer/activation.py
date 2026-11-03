"""NumPy activation functions and derivatives with respect to preactivations."""

import numpy as np


def _as_float_array(x):
    values = np.asarray(x)
    return values.astype(np.result_type(values.dtype, np.float32), copy=False)


def sigmoid(x):
    """Stable sigmoid, preserving float32/float64 precision and array shape."""
    values = _as_float_array(x)
    with np.errstate(under="ignore"):
        exp_value = np.exp(-np.abs(values))
    return np.where(values >= 0, 1 / (1 + exp_value), exp_value / (1 + exp_value))


def sigmoid_derivative(x):
    values = sigmoid(x)
    with np.errstate(under="ignore"):
        return values * (1 - values)


def relu(x):
    return np.maximum(0, _as_float_array(x))


def relu_derivative(x):
    """Use the conventional derivative zero at the origin."""
    values = _as_float_array(x)
    return (values > 0).astype(values.dtype)


def silu(x):
    values = _as_float_array(x)
    with np.errstate(under="ignore"):
        return values * sigmoid(values)


def silu_derivative(x):
    values = _as_float_array(x)
    s = sigmoid(values)
    with np.errstate(under="ignore"):
        return s + values * s * (1 - s)

def softmax(x):
    values = _as_float_array(x)
    max_value = np.max(values, axis=-1, keepdims=True)
    exp_values = np.exp(values - max_value)
    return exp_values / np.sum(exp_values, axis=-1, keepdims=True)

def softmax_derivative(z):
    p = softmax(z)
    return np.diag(p) - np.outer(p, p)

class ActivationFunction:
    """Select a callable and its derivative; both receive z = W @ a + b."""

    _FUNCTIONS = {
        "sigmoid": (sigmoid, sigmoid_derivative, False),
        "relu": (relu, relu_derivative, False),
        "silu": (silu, silu_derivative, False),
        "softmax": (softmax, softmax_derivative, True),
    }

    def __init__(self, func_name):
        if not isinstance(func_name, str) or func_name not in self._FUNCTIONS:
            names = ", ".join(self._FUNCTIONS)
            raise ValueError(f"Unknown activation {func_name!r}; expected one of: {names}")
        self.func_name = func_name
        self.func, self.derivative, self.gradient = self._FUNCTIONS[func_name]
