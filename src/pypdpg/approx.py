"""Polynomial approximations of nonlinear functions, evaluable under CKKS."""

from __future__ import annotations

import numpy as np

from .core import CipherArray

# Standard CKKS logistic approximation (lowest-degree coefficient first):
#   sigmoid(x) ~= 0.5 + 0.197 x - 0.004 x^3, valid for x in roughly [-5, 5].
SIGMOID_COEFFS = [0.5, 0.197, 0.0, -0.004]


def sigmoid(x):
    """Degree-3 polynomial approximation of the logistic function.

    Works on CipherArrays (encrypted) and plain arrays alike, so the same
    model code runs on both. Accurate to ~1e-1 on [-5, 5]; diverges outside.
    """
    if isinstance(x, CipherArray):
        return x._polyval(SIGMOID_COEFFS)
    x = np.asarray(x, dtype=np.float64)
    return 0.5 + 0.197 * x - 0.004 * x**3
