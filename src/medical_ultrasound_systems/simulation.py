"""Synthetic wavefield generation helpers."""

from __future__ import annotations

import numpy as np

from .wavefield import Wavefield


def synthetic_plane_wave(
    n_samples: int = 1024,
    frequency_hz: float = 1e6,
    sample_rate_hz: float = 20e6,
    axis: tuple[float, float, float] = (1, 0, 0),
) -> Wavefield:
    """Generate a simple quaternion plane wave rotating about an imaginary axis."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    axis_v = np.asarray(axis, dtype=float)
    axis_norm = np.linalg.norm(axis_v)
    if axis_norm == 0.0:
        raise ValueError("axis must be non-zero.")
    axis_u = axis_v / axis_norm

    t = np.arange(n_samples, dtype=float) / float(sample_rate_hz)
    phase = 2.0 * np.pi * float(frequency_hz) * t

    w = np.cos(phase)
    imag = np.sin(phase)[:, np.newaxis] * axis_u[np.newaxis, :]
    samples = np.concatenate((w[:, np.newaxis], imag), axis=1)

    return Wavefield(
        samples=samples,
        sample_rate_hz=sample_rate_hz,
        metadata={"generator": "synthetic_plane_wave", "axis": tuple(map(float, axis_u))},
    )
