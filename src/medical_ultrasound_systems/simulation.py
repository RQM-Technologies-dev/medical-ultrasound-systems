"""Synthetic ultrasound data generators for research-only algorithm development."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import LinearArrayGeometry
from .phantom import PointScattererPhantom
from .pulse import gaussian_modulated_pulse, normalize_pulse
from .wavefield import Wavefield


@dataclass
class RFChannelData:
    """Container for synthetic RF channel samples from a linear array.

    This class stores baseline research simulation outputs. It does not represent
    calibrated scanner hardware or clinical acquisition behavior.
    """

    samples: np.ndarray
    sample_rate_hz: float
    geometry: LinearArrayGeometry
    sound_speed_m_s: float = 1540.0
    metadata: dict | None = None

    def __post_init__(self) -> None:
        self.samples = np.asarray(self.samples, dtype=float)
        if self.samples.ndim != 2:
            raise ValueError("samples must be 2D with shape (channels, samples).")

        self.sample_rate_hz = float(self.sample_rate_hz)
        self.sound_speed_m_s = float(self.sound_speed_m_s)
        if self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be positive.")
        if self.sound_speed_m_s <= 0.0:
            raise ValueError("sound_speed_m_s must be positive.")
        if self.samples.shape[0] != self.geometry.n_elements:
            raise ValueError("Number of channels must match geometry.n_elements.")

    @property
    def n_channels(self) -> int:
        """Return channel count."""
        return int(self.samples.shape[0])

    @property
    def n_samples(self) -> int:
        """Return number of temporal RF samples per channel."""
        return int(self.samples.shape[1])


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


def simulate_pulse_echo_rf(
    geometry: LinearArrayGeometry,
    phantom: PointScattererPhantom,
    sample_rate_hz: float = 40e6,
    center_frequency_hz: float = 5e6,
    duration_s: float = 80e-6,
    sound_speed_m_s: float = 1540.0,
) -> RFChannelData:
    """Simulate simplified pulse-echo RF data for point-scatterer phantoms.

    Model assumptions:
    - plane-wave transmit approximation
    - transmit path equals scatterer depth `z`
    - receive path is Euclidean distance to each element
    - additive pulse responses with lightweight geometric attenuation

    This function is a software-method placeholder for research benchmarking and
    should not be interpreted as a full acoustic propagation solver.
    """
    sample_rate_hz = float(sample_rate_hz)
    center_frequency_hz = float(center_frequency_hz)
    duration_s = float(duration_s)
    sound_speed_m_s = float(sound_speed_m_s)
    if sample_rate_hz <= 0.0:
        raise ValueError("sample_rate_hz must be positive.")
    if center_frequency_hz <= 0.0:
        raise ValueError("center_frequency_hz must be positive.")
    if duration_s <= 0.0:
        raise ValueError("duration_s must be positive.")
    if sound_speed_m_s <= 0.0:
        raise ValueError("sound_speed_m_s must be positive.")

    n_time_samples = int(np.ceil(duration_s * sample_rate_hz))
    if n_time_samples <= 0:
        raise ValueError("duration_s and sample_rate_hz must produce at least one sample.")
    t_samples = np.arange(n_time_samples, dtype=float) / sample_rate_hz
    channel_data = np.zeros((geometry.n_elements, n_time_samples), dtype=float)
    element_positions = geometry.element_positions_m

    pulse_t, pulse = gaussian_modulated_pulse(
        center_frequency_hz=center_frequency_hz,
        sample_rate_hz=sample_rate_hz,
    )
    pulse = normalize_pulse(pulse)

    scatterers = phantom.as_array()
    for x_m, z_m, amplitude in scatterers:
        tx_path_m = z_m
        rx_paths_m = np.sqrt((element_positions[:, 0] - x_m) ** 2 + z_m**2)
        travel_time_s = (tx_path_m + rx_paths_m) / sound_speed_m_s
        attenuation = 1.0 / np.maximum(tx_path_m + rx_paths_m, 1e-6)

        for ch_idx, delay_s in enumerate(travel_time_s):
            shifted = np.interp(t_samples - delay_s, pulse_t, pulse, left=0.0, right=0.0)
            channel_data[ch_idx, :] += float(amplitude) * attenuation[ch_idx] * shifted

    metadata = {
        "generator": "simulate_pulse_echo_rf",
        "assumptions": [
            "plane_wave_transmit",
            "point_scatterers",
            "linear_interpolated_delay",
            "geometric_1_over_distance_attenuation",
        ],
        "center_frequency_hz": center_frequency_hz,
        "duration_s": duration_s,
    }
    return RFChannelData(
        samples=channel_data,
        sample_rate_hz=sample_rate_hz,
        geometry=geometry,
        sound_speed_m_s=sound_speed_m_s,
        metadata=metadata,
    )
