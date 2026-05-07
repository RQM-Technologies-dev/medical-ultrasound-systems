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


def psnr(reference: np.ndarray, estimate: np.ndarray, data_range: float | None = None) -> float:
    """Return peak signal-to-noise ratio in dB."""
    reference = np.asarray(reference, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    if reference.shape != estimate.shape:
        raise ValueError("reference and estimate must have matching shapes.")

    if data_range is None:
        data_range = float(np.max(reference) - np.min(reference))
    else:
        data_range = float(data_range)
        if data_range <= 0.0:
            raise ValueError("data_range must be positive when provided.")

    err = mse(reference, estimate)
    if err == 0.0:
        return float("inf")

    eps = np.finfo(float).eps
    range_safe = max(data_range, eps)
    return float(20.0 * np.log10(range_safe) - 10.0 * np.log10(max(err, eps)))


def correlation_coefficient(a: np.ndarray, b: np.ndarray) -> float:
    """Return Pearson correlation coefficient for two arrays."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError("a and b must have matching shapes.")
    if a.size == 0:
        raise ValueError("a and b must be non-empty.")

    a_centered = a - np.mean(a)
    b_centered = b - np.mean(b)
    denom = np.sqrt(np.sum(a_centered**2) * np.sum(b_centered**2))
    if denom <= np.finfo(float).eps:
        return 0.0
    corr = np.sum(a_centered * b_centered) / denom
    return float(np.clip(corr, -1.0, 1.0))


def normalized_cross_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Return normalized cross-correlation between two arrays."""
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if a.shape != b.shape:
        raise ValueError("a and b must have matching shapes.")
    if a.size == 0:
        raise ValueError("a and b must be non-empty.")

    denom = np.sqrt(np.sum(a**2) * np.sum(b**2))
    if denom <= np.finfo(float).eps:
        return 0.0
    return float(np.sum(a * b) / denom)
