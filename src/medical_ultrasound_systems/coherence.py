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
