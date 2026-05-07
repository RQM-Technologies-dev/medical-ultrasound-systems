"""Quaternionic channel-wavefield representations for synthetic RF research."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .analytic import analytic_signal_fft
from .geometry import LinearArrayGeometry
from .simulation import RFChannelData


@dataclass
class QuaternionicChannelWavefield:
    """Container for quaternionic channel samples derived from RF data."""

    samples: np.ndarray
    sample_rate_hz: float
    metadata: dict | None = None

    def __post_init__(self) -> None:
        self.samples = np.asarray(self.samples, dtype=float)
        if self.samples.ndim != 3 or self.samples.shape[-1] != 4:
            raise ValueError("samples must have shape (n_channels, n_samples, 4).")
        self.sample_rate_hz = float(self.sample_rate_hz)
        if self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be positive.")

    @property
    def shape(self) -> tuple[int, ...]:
        """Return full sample-array shape."""
        return self.samples.shape

    @property
    def n_samples(self) -> int:
        """Return temporal sample count for channels x samples x 4 arrays."""
        return int(self.samples.shape[1])

    @property
    def n_channels(self) -> int:
        """Return channel count for channels x samples x 4 arrays."""
        return int(self.samples.shape[0])


def make_pixel_orientation_axes(
    geometry: LinearArrayGeometry,
    x_m: float,
    z_m: float,
) -> np.ndarray:
    """Return normalized element-to-pixel orientation vectors with shape `(N, 3)`."""
    z_m = float(z_m)
    if z_m <= 0.0:
        raise ValueError("z_m must be positive.")

    element_positions = geometry.element_positions_m
    vectors = np.column_stack(
        (
            float(x_m) - element_positions[:, 0],
            np.zeros(geometry.n_elements, dtype=float),
            np.full(geometry.n_elements, z_m, dtype=float) - element_positions[:, 1],
        )
    )
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, np.finfo(float).eps)


def rf_to_quaternionic_channels(
    rf: RFChannelData,
    orientation_axes: np.ndarray | None = None,
) -> QuaternionicChannelWavefield:
    """Lift RF channel data to quaternionic channel samples.

    The orientation model is a simplified geometry proxy for software research
    and should not be interpreted as a complete physical acoustics model.
    """
    analytic = analytic_signal_fft(rf.samples, axis=-1)
    amplitude = np.abs(analytic)
    phase = np.angle(analytic)

    if orientation_axes is None:
        element_x = rf.geometry.element_positions_m[:, 0]
        axes = np.column_stack(
            (
                element_x,
                np.zeros(rf.n_channels, dtype=float),
                np.ones(rf.n_channels, dtype=float),
            )
        )
    else:
        axes = np.asarray(orientation_axes, dtype=float)
        if axes.shape != (rf.n_channels, 3):
            raise ValueError("orientation_axes must have shape (n_channels, 3).")

    axis_norms = np.linalg.norm(axes, axis=1, keepdims=True)
    axes = axes / np.maximum(axis_norms, np.finfo(float).eps)

    cos_phase = np.cos(phase)
    sin_phase = np.sin(phase)
    q0 = amplitude * cos_phase
    q_vec = amplitude[..., np.newaxis] * sin_phase[..., np.newaxis] * axes[:, np.newaxis, :]
    q_samples = np.concatenate((q0[..., np.newaxis], q_vec), axis=-1)

    metadata = {
        "source": "RFChannelData",
        "analytic_method": "FFT Hilbert approximation",
        "orientation_model": "simplified array-geometry proxy",
    }
    return QuaternionicChannelWavefield(
        samples=q_samples,
        sample_rate_hz=rf.sample_rate_hz,
        metadata=metadata,
    )
