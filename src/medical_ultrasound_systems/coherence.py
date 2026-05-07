"""Coherence and alignment metrics for quaternion wavefields."""

from __future__ import annotations

import numpy as np

from .delay import pixel_travel_times_plane_wave, sample_rf_nearest
from .quaternion import quaternion_normalize
from .simulation import RFChannelData


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


def conventional_coherence_image(
    rf: RFChannelData,
    x_grid_m: np.ndarray,
    z_grid_m: np.ndarray,
    sound_speed_m_s: float | None = None,
) -> np.ndarray:
    """Compute a conventional channel-coherence image on a Cartesian grid."""
    x_grid_m = np.asarray(x_grid_m, dtype=float)
    z_grid_m = np.asarray(z_grid_m, dtype=float)
    if x_grid_m.ndim != 1 or z_grid_m.ndim != 1:
        raise ValueError("x_grid_m and z_grid_m must be 1D arrays.")

    c = rf.sound_speed_m_s if sound_speed_m_s is None else float(sound_speed_m_s)
    if c <= 0.0:
        raise ValueError("sound_speed_m_s must be positive.")

    image = np.zeros((z_grid_m.size, x_grid_m.size), dtype=float)
    for iz, z_m in enumerate(z_grid_m):
        for ix, x_m in enumerate(x_grid_m):
            travel_times_s = pixel_travel_times_plane_wave(
                rf.geometry, x_m=float(x_m), z_m=float(z_m), sound_speed_m_s=c
            )
            channel_values = sample_rf_nearest(rf, travel_times_s)
            image[iz, ix] = float(channel_coherence_factor(channel_values, axis=0))
    return image
