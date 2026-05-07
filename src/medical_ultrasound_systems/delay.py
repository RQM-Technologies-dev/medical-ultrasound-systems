"""Shared delay-model utilities for synthetic ultrasound research methods."""

from __future__ import annotations

import numpy as np

from .geometry import LinearArrayGeometry
from .simulation import RFChannelData


def pixel_travel_times_plane_wave(
    geometry: LinearArrayGeometry,
    x_m: float,
    z_m: float,
    sound_speed_m_s: float,
) -> np.ndarray:
    """Return per-channel total travel times for a plane-wave transmit model."""
    sound_speed_m_s = float(sound_speed_m_s)
    z_m = float(z_m)
    if sound_speed_m_s <= 0.0:
        raise ValueError("sound_speed_m_s must be positive.")
    if z_m < 0.0:
        raise ValueError("z_m must be non-negative.")

    element_positions = geometry.element_positions_m
    tx_time_s = z_m / sound_speed_m_s
    rx_time_s = np.sqrt((element_positions[:, 0] - float(x_m)) ** 2 + z_m**2) / sound_speed_m_s
    return tx_time_s + rx_time_s


def sample_rf_nearest(rf: RFChannelData, travel_times_s: np.ndarray) -> np.ndarray:
    """Sample one RF value per channel at nearest indices, zero-filling out-of-range."""
    travel_times_s = np.asarray(travel_times_s, dtype=float)
    if travel_times_s.ndim != 1:
        raise ValueError("travel_times_s must be a 1D array.")
    if travel_times_s.shape[0] != rf.n_channels:
        raise ValueError("travel_times_s length must match rf channel count.")

    sample_idx = np.rint(travel_times_s * rf.sample_rate_hz).astype(int)
    sampled = np.zeros(rf.n_channels, dtype=rf.samples.dtype)
    valid = (sample_idx >= 0) & (sample_idx < rf.n_samples)
    if np.any(valid):
        ch_idx = np.arange(rf.n_channels)[valid]
        sampled[valid] = rf.samples[ch_idx, sample_idx[valid]]
    return sampled
