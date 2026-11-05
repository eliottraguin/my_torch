"""Validate training controls shared by model generation, loading and training."""

import math

DEFAULT_MAX_EPOCHS = 200
DEFAULT_PATIENCE = 10
DEFAULT_MIN_DELTA = 1e-6


def training_settings(data):
    settings = {
        'max_epochs': data.get('max_epochs', DEFAULT_MAX_EPOCHS),
        'patience': data.get('patience', DEFAULT_PATIENCE),
        'min_delta': data.get('min_delta', DEFAULT_MIN_DELTA),
    }
    for name in ('max_epochs', 'patience'):
        value = settings[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    value = settings['min_delta']
    try:
        valid = (not isinstance(value, bool) and isinstance(value, (int, float))
                 and math.isfinite(value) and value >= 0)
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError('min_delta must be a finite nonnegative number')
    settings['min_delta'] = float(value)
    return settings
