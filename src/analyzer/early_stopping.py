"""Monitor a loss and retain independent copies of the best parameters."""

import math


class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-6):
        if (not isinstance(patience, int) or isinstance(patience, bool)
                or patience < 1 or not math.isfinite(min_delta) or min_delta < 0):
            raise ValueError("patience must be positive and min_delta finite and nonnegative")
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = math.inf
        self.best_epoch = 0
        self._reference_loss = math.inf
        self._waiting = 0
        self._parameters = None

    def update(self, loss, network, epoch):
        """Return True after patience evaluations without meaningful improvement."""
        if not math.isfinite(loss):
            raise ValueError("Non-finite loss during training; previous checkpoint preserved")
        if loss < self.best_loss:
            self.best_loss = loss
            self.best_epoch = epoch
            self._parameters = (
                [w.copy() for w in network.weights],
                [b.copy() for b in network.biases],
            )
        if loss < self._reference_loss - self.min_delta:
            self._reference_loss = loss
            self._waiting = 0
        else:
            self._waiting += 1
        return self._waiting >= self.patience

    def restore(self, network):
        """Restore the lowest-loss checkpoint, including the initial model."""
        if self._parameters is None:
            raise ValueError("No checkpoint available")
        weights, biases = self._parameters
        network.weights = [w.copy() for w in weights]
        network.biases = [b.copy() for b in biases]
