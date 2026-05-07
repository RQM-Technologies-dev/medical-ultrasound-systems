"""Pulse-shape helpers for synthetic ultrasound channel-data experiments."""

from __future__ import annotations

import numpy as np


def gaussian_modulated_pulse(
    center_frequency_hz: float = 5e6,
    sample_rate_hz: float = 40e6,
    n_cycles: float = 2.5,
    fractional_bandwidth: float = 0.6,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a real Gaussian-envelope cosine pulse centered near t=0.

    The envelope standard deviation is approximated from center frequency and
    fractional bandwidth, yielding a practical synthetic pulse for baseline
    simulation studies (not a calibrated transducer model).
    """
    center_frequency_hz = float(center_frequency_hz)
    sample_rate_hz = float(sample_rate_hz)
    n_cycles = float(n_cycles)
    fractional_bandwidth = float(fractional_bandwidth)

    if center_frequency_hz <= 0.0:
        raise ValueError("center_frequency_hz must be positive.")
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive.")
    if n_cycles <= 0.0:
        raise ValueError("n_cycles must be positive.")
    if fractional_bandwidth <= 0.0:
        raise ValueError("fractional_bandwidth must be positive.")

    sigma_s = np.sqrt(2.0 * np.log(2.0)) / (np.pi * fractional_bandwidth * center_frequency_hz)
    half_duration_s = max(0.5 * n_cycles / center_frequency_hz, 3.0 * sigma_s)
    n_samples = int(np.ceil(2.0 * half_duration_s * sample_rate_hz)) + 1
    t = (np.arange(n_samples, dtype=float) - (n_samples // 2)) / sample_rate_hz

    envelope = np.exp(-0.5 * (t / sigma_s) ** 2)
    carrier = np.cos(2.0 * np.pi * center_frequency_hz * t)
    pulse = envelope * carrier
    return t, pulse


def normalize_pulse(pulse: np.ndarray) -> np.ndarray:
    """Normalize a pulse to max absolute value 1 while preserving sign."""
    pulse = np.asarray(pulse, dtype=float)
    if pulse.size == 0:
        return pulse.copy()
    peak = float(np.max(np.abs(pulse)))
    if peak == 0.0:
        return np.zeros_like(pulse)
    return pulse / peak
