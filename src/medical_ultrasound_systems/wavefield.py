"""Wavefield containers for quaternionic ultrasound samples."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Wavefield:
    """Container for quaternion wavefield samples with metadata."""

    samples: np.ndarray
    sample_rate_hz: float
    metadata: dict | None = None

    def __post_init__(self) -> None:
        self.samples = np.asarray(self.samples, dtype=float)
        if self.samples.ndim == 0 or self.samples.shape[-1] != 4:
            raise ValueError("Wavefield samples must have last dimension 4.")
        self.sample_rate_hz = float(self.sample_rate_hz)

    @property
    def n_samples(self) -> int:
        """Return number of time samples in the first dimension."""
        return int(self.samples.shape[0])
