"""Simplified ultrasound array geometry models for research software experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LinearArrayGeometry:
    """Linear 2D array layout used for synthetic ultrasound research pipelines.

    This class represents a simplified software geometry model for simulation and
    baseline algorithm benchmarking. It is not an acoustic hardware model and is
    not intended to represent a clinical ultrasound device design.
    """

    n_elements: int
    pitch_m: float
    center_frequency_hz: float | None = None
    metadata: dict | None = None

    def __post_init__(self) -> None:
        self.n_elements = int(self.n_elements)
        self.pitch_m = float(self.pitch_m)
        if self.n_elements <= 0:
            raise ValueError("n_elements must be positive.")
        if self.pitch_m <= 0.0:
            raise ValueError("pitch_m must be positive.")
        if self.center_frequency_hz is not None:
            self.center_frequency_hz = float(self.center_frequency_hz)
            if self.center_frequency_hz <= 0.0:
                raise ValueError("center_frequency_hz must be positive when provided.")

    @property
    def center_index(self) -> float:
        """Return fractional index of the aperture center."""
        return 0.5 * (self.n_elements - 1)

    @property
    def aperture_m(self) -> float:
        """Return element-to-element aperture width in meters."""
        return (self.n_elements - 1) * self.pitch_m

    @property
    def element_positions_m(self) -> np.ndarray:
        """Return `(x, z)` element positions in meters with aperture centered at x=0."""
        x_positions = (np.arange(self.n_elements, dtype=float) - self.center_index) * self.pitch_m
        z_positions = np.zeros(self.n_elements, dtype=float)
        return np.column_stack((x_positions, z_positions))
