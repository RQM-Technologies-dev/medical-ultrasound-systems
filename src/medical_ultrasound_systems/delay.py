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


def sample_array_linear_per_channel(
    samples: np.ndarray,
    sample_rate_hz: float,
    travel_times_s: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate one channel value per travel time.

    Parameters
    ----------
    samples:
        Array with shape (channels, samples). Supports real or complex values.
    sample_rate_hz:
        Sample rate used to map travel times onto sample indices.
    travel_times_s:
        Per-channel travel times in seconds with shape (channels,).
    """
    samples = np.asarray(samples)
    if samples.ndim != 2:
        raise ValueError("samples must be a 2D array with shape (channels, samples).")

    travel_times_s = np.asarray(travel_times_s, dtype=float)
    if travel_times_s.ndim != 1:
        raise ValueError("travel_times_s must be a 1D array.")
    if travel_times_s.shape[0] != samples.shape[0]:
        raise ValueError("travel_times_s length must match channel count.")

    sample_rate_hz = float(sample_rate_hz)
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive.")

    n_channels, n_samples = samples.shape
    sampled = np.zeros(n_channels, dtype=samples.dtype)
    if n_samples == 0 or n_channels == 0:
        return sampled

    sample_idx = travel_times_s * sample_rate_hz
    lower = np.floor(sample_idx).astype(int)
    upper = lower + 1
    frac = sample_idx - lower

    valid = (lower >= 0) & (lower < n_samples)
    if not np.any(valid):
        return sampled

    ch_idx = np.arange(n_channels)[valid]
    lower_valid = lower[valid]
    upper_valid = np.clip(upper[valid], 0, n_samples - 1)
    frac_valid = frac[valid]

    low = samples[ch_idx, lower_valid]
    high = samples[ch_idx, upper_valid]

    # At the last exact sample index, keep endpoint value instead of zero-filling.
    past_end = upper[valid] >= n_samples
    if np.any(past_end):
        high[past_end] = low[past_end]
        frac_valid[past_end] = 0.0

    sampled[valid] = (1.0 - frac_valid) * low + frac_valid * high
    return sampled


def sample_rf_linear(rf: RFChannelData, travel_times_s: np.ndarray) -> np.ndarray:
    """Sample one RF value per channel using linear interpolation."""
    travel_times_s = np.asarray(travel_times_s, dtype=float)
    if travel_times_s.ndim != 1:
        raise ValueError("travel_times_s must be a 1D array.")
    if travel_times_s.shape[0] != rf.n_channels:
        raise ValueError("travel_times_s length must match rf channel count.")
    return sample_array_linear_per_channel(
        samples=rf.samples,
        sample_rate_hz=rf.sample_rate_hz,
        travel_times_s=travel_times_s,
    )
