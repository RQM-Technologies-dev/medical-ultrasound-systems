"""Quaternionic delay-alignment baselines for synthetic ultrasound research."""

from __future__ import annotations

import numpy as np

from .analytic import analytic_signal_fft
from .delay import pixel_travel_times_plane_wave
from .qwavefield import make_pixel_orientation_axes
from .simulation import RFChannelData


def _sample_analytic_nearest(
    analytic_rf: np.ndarray,
    sample_rate_hz: float,
    travel_times_s: np.ndarray,
) -> np.ndarray:
    sample_idx = np.rint(travel_times_s * sample_rate_hz).astype(int)
    n_channels, n_samples = analytic_rf.shape
    sampled = np.zeros(n_channels, dtype=analytic_rf.dtype)
    valid = (sample_idx >= 0) & (sample_idx < n_samples)
    if np.any(valid):
        ch_idx = np.arange(n_channels)[valid]
        sampled[valid] = analytic_rf[ch_idx, sample_idx[valid]]
    return sampled


def _quaternionic_from_complex_channels(
    complex_channels: np.ndarray,
    axes: np.ndarray,
) -> np.ndarray:
    amplitude = np.abs(complex_channels)
    phase = np.angle(complex_channels)
    q0 = amplitude * np.cos(phase)
    q_vec = amplitude[:, np.newaxis] * np.sin(phase)[:, np.newaxis] * axes
    return np.concatenate((q0[:, np.newaxis], q_vec), axis=1)


def quaternionic_delay_align_pixel(
    rf: RFChannelData,
    x_m: float,
    z_m: float,
    sound_speed_m_s: float | None = None,
) -> np.ndarray:
    """Return delay-aligned quaternionic channel values for one pixel."""
    c = rf.sound_speed_m_s if sound_speed_m_s is None else float(sound_speed_m_s)
    travel_times_s = pixel_travel_times_plane_wave(rf.geometry, x_m=float(x_m), z_m=float(z_m), sound_speed_m_s=c)

    analytic_rf = analytic_signal_fft(rf.samples, axis=-1)
    delayed_channels = _sample_analytic_nearest(analytic_rf, rf.sample_rate_hz, travel_times_s)
    axes = make_pixel_orientation_axes(rf.geometry, x_m=float(x_m), z_m=float(z_m))
    return _quaternionic_from_complex_channels(delayed_channels, axes)


def quaternionic_alignment_factor(q_values: np.ndarray) -> float:
    """Return a [0, 1] quaternionic alignment score across channels."""
    q_values = np.asarray(q_values, dtype=float)
    if q_values.ndim != 2 or q_values.shape[1] != 4:
        raise ValueError("q_values must have shape (n_channels, 4).")
    if q_values.shape[0] == 0:
        return 0.0

    norms = np.linalg.norm(q_values, axis=1)
    valid = norms > np.finfo(float).eps
    if not np.any(valid):
        return 0.0

    q_norm = q_values[valid] / norms[valid, np.newaxis]
    q_mean = np.mean(q_norm, axis=0)
    mean_norm = np.linalg.norm(q_mean)
    if mean_norm <= np.finfo(float).eps:
        return 0.0

    q_ref = q_mean / mean_norm
    alignment = np.abs(np.sum(q_norm * q_ref[np.newaxis, :], axis=1))
    return float(np.clip(np.mean(alignment), 0.0, 1.0))


def quaternionic_alignment_image(
    rf: RFChannelData,
    x_grid_m: np.ndarray,
    z_grid_m: np.ndarray,
    sound_speed_m_s: float | None = None,
) -> np.ndarray:
    """Compute a quaternionic alignment map over a Cartesian pixel grid."""
    x_grid_m = np.asarray(x_grid_m, dtype=float)
    z_grid_m = np.asarray(z_grid_m, dtype=float)
    if x_grid_m.ndim != 1 or z_grid_m.ndim != 1:
        raise ValueError("x_grid_m and z_grid_m must be 1D arrays.")

    c = rf.sound_speed_m_s if sound_speed_m_s is None else float(sound_speed_m_s)
    image = np.zeros((z_grid_m.size, x_grid_m.size), dtype=float)
    analytic_rf = analytic_signal_fft(rf.samples, axis=-1)

    for iz, z_m in enumerate(z_grid_m):
        for ix, x_m in enumerate(x_grid_m):
            travel_times_s = pixel_travel_times_plane_wave(
                rf.geometry, x_m=float(x_m), z_m=float(z_m), sound_speed_m_s=c
            )
            delayed_channels = _sample_analytic_nearest(analytic_rf, rf.sample_rate_hz, travel_times_s)
            axes = make_pixel_orientation_axes(rf.geometry, x_m=float(x_m), z_m=float(z_m))
            q_values = _quaternionic_from_complex_channels(delayed_channels, axes)
            image[iz, ix] = quaternionic_alignment_factor(q_values)
    return image


def quaternionic_intensity_image(
    rf: RFChannelData,
    x_grid_m: np.ndarray,
    z_grid_m: np.ndarray,
    sound_speed_m_s: float | None = None,
) -> np.ndarray:
    """Compute norm of summed quaternionic channel values per pixel."""
    x_grid_m = np.asarray(x_grid_m, dtype=float)
    z_grid_m = np.asarray(z_grid_m, dtype=float)
    if x_grid_m.ndim != 1 or z_grid_m.ndim != 1:
        raise ValueError("x_grid_m and z_grid_m must be 1D arrays.")

    c = rf.sound_speed_m_s if sound_speed_m_s is None else float(sound_speed_m_s)
    image = np.zeros((z_grid_m.size, x_grid_m.size), dtype=float)
    analytic_rf = analytic_signal_fft(rf.samples, axis=-1)

    for iz, z_m in enumerate(z_grid_m):
        for ix, x_m in enumerate(x_grid_m):
            travel_times_s = pixel_travel_times_plane_wave(
                rf.geometry, x_m=float(x_m), z_m=float(z_m), sound_speed_m_s=c
            )
            delayed_channels = _sample_analytic_nearest(analytic_rf, rf.sample_rate_hz, travel_times_s)
            axes = make_pixel_orientation_axes(rf.geometry, x_m=float(x_m), z_m=float(z_m))
            q_values = _quaternionic_from_complex_channels(delayed_channels, axes)
            image[iz, ix] = float(np.linalg.norm(np.sum(q_values, axis=0)))
    return image
