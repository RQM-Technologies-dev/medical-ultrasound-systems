"""Baseline beamforming and post-processing for research-only workflows."""

from __future__ import annotations

import numpy as np

from .simulation import RFChannelData


def delay_and_sum_plane_wave(
    rf: RFChannelData,
    x_grid_m: np.ndarray,
    z_grid_m: np.ndarray,
    sound_speed_m_s: float | None = None,
) -> np.ndarray:
    """Compute a simple plane-wave delay-and-sum image.

    This baseline is intended for synthetic benchmarking only and does not
    represent production imaging performance or clinical interpretation.
    """
    x_grid_m = np.asarray(x_grid_m, dtype=float)
    z_grid_m = np.asarray(z_grid_m, dtype=float)
    if x_grid_m.ndim != 1 or z_grid_m.ndim != 1:
        raise ValueError("x_grid_m and z_grid_m must be 1D arrays.")
    if np.any(z_grid_m < 0.0):
        raise ValueError("z_grid_m must be non-negative.")

    c = rf.sound_speed_m_s if sound_speed_m_s is None else float(sound_speed_m_s)
    if c <= 0.0:
        raise ValueError("sound_speed_m_s must be positive.")

    sample_rate_hz = rf.sample_rate_hz
    n_channels, n_samples = rf.samples.shape
    element_x = rf.geometry.element_positions_m[:, 0]
    image = np.zeros((z_grid_m.size, x_grid_m.size), dtype=float)

    for iz, z_m in enumerate(z_grid_m):
        tx_time_s = z_m / c
        for ix, x_m in enumerate(x_grid_m):
            rx_time_s = np.sqrt((element_x - x_m) ** 2 + z_m**2) / c
            total_time_s = tx_time_s + rx_time_s
            sample_idx = np.rint(total_time_s * sample_rate_hz).astype(int)
            valid = (sample_idx >= 0) & (sample_idx < n_samples)
            if np.any(valid):
                ch_idx = np.arange(n_channels)[valid]
                image[iz, ix] = float(np.sum(rf.samples[ch_idx, sample_idx[valid]]))
    return image


def envelope_detect_fft(signal: np.ndarray, axis: int = -1) -> np.ndarray:
    """Return envelope magnitude using an FFT analytic-signal approximation."""
    signal = np.asarray(signal)
    if np.iscomplexobj(signal):
        raise ValueError("envelope_detect_fft expects real-valued input.")
    signal = signal.astype(float, copy=False)
    n = signal.shape[axis]
    if n == 0:
        return np.zeros_like(signal, dtype=float)

    spectrum = np.fft.fft(signal, axis=axis)
    h = np.zeros(n, dtype=float)
    if n % 2 == 0:
        h[0] = 1.0
        h[n // 2] = 1.0
        h[1 : n // 2] = 2.0
    else:
        h[0] = 1.0
        h[1 : (n + 1) // 2] = 2.0

    shape = [1] * signal.ndim
    shape[axis] = n
    analytic = np.fft.ifft(spectrum * h.reshape(shape), axis=axis)
    return np.abs(analytic)


def log_compress(image: np.ndarray, dynamic_range_db: float = 60.0) -> np.ndarray:
    """Apply log compression and normalize output to [0, 1]."""
    dynamic_range_db = float(dynamic_range_db)
    if dynamic_range_db <= 0.0:
        raise ValueError("dynamic_range_db must be positive.")

    image = np.asarray(image, dtype=float)
    magnitude = np.abs(image)
    if magnitude.size == 0:
        return np.zeros_like(magnitude)

    peak = float(np.max(magnitude))
    if peak == 0.0:
        return np.zeros_like(magnitude)

    eps = np.finfo(float).eps
    db = 20.0 * np.log10(np.maximum(magnitude, peak * eps) / peak)
    normalized = (db + dynamic_range_db) / dynamic_range_db
    return np.clip(normalized, 0.0, 1.0)
