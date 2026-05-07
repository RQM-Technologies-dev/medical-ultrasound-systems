"""Coherence and alignment metrics for quaternion wavefields."""

from __future__ import annotations

import numpy as np

from .quaternion import quaternion_normalize


def coherence_score(reference: np.ndarray, observed: np.ndarray) -> float:
    """Compute a [0, 1] coherence score from normalized quaternion alignment."""
    reference = np.asarray(reference, dtype=float)
    observed = np.asarray(observed, dtype=float)

    if reference.shape != observed.shape:
        raise ValueError("reference and observed must have matching shapes.")
    if reference.size == 0:
        return 0.0
    if reference.shape[-1] != 4:
        raise ValueError("Quaternion arrays must have last dimension 4.")

    ref_n = quaternion_normalize(reference)
    obs_n = quaternion_normalize(observed)

    alignment = np.abs(np.sum(ref_n * obs_n, axis=-1))
    alignment = np.clip(alignment, 0.0, 1.0)
    return float(np.mean(alignment))


def channel_coherence_factor(channel_values: np.ndarray, axis: int = 0) -> np.ndarray:
    """Compute conventional coherence factor over channel data.

    CF = |sum(x)|^2 / (N * sum(|x|^2) + eps)
    """
    channel_values = np.asarray(channel_values)
    if channel_values.shape[axis] == 0:
        raise ValueError("channel_values must have at least one element on the coherence axis.")

    n = channel_values.shape[axis]
    numerator = np.abs(np.sum(channel_values, axis=axis)) ** 2
    denominator = (n * np.sum(np.abs(channel_values) ** 2, axis=axis)) + np.finfo(float).eps
    cf = numerator / denominator
    return np.clip(cf, 0.0, 1.0).astype(float, copy=False)


def quaternion_alignment_score(reference: np.ndarray, observed: np.ndarray) -> float:
    """Domain-readable wrapper for quaternion coherence alignment scoring."""
    return coherence_score(reference, observed)
