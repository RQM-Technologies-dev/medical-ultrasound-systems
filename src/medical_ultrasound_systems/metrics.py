"""General benchmark metrics."""

from __future__ import annotations

import numpy as np


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Return mean squared error between arrays."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a and b must have matching shapes.")
    return float(np.mean((a - b) ** 2))


def normalized_error(a: np.ndarray, b: np.ndarray) -> float:
    """Return MSE normalized by reference signal energy."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("a and b must have matching shapes.")

    denom = float(np.mean(a**2))
    if denom == 0.0:
        return 0.0 if np.allclose(a, b) else float("inf")
    return mse(a, b) / denom
