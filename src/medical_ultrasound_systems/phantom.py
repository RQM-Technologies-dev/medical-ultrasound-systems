"""Point-scatterer phantom helpers for synthetic ultrasound research studies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PointScatterer:
    """Single point-scatterer definition for non-clinical synthetic experiments."""

    x_m: float
    z_m: float
    amplitude: float = 1.0

    def __post_init__(self) -> None:
        self.x_m = float(self.x_m)
        self.z_m = float(self.z_m)
        self.amplitude = float(self.amplitude)
        if self.z_m <= 0.0:
            raise ValueError("z_m must be positive for point scatterers.")


@dataclass
class PointScattererPhantom:
    """Collection of point scatterers used as a lightweight research phantom."""

    scatterers: list[PointScatterer]
    metadata: dict | None = None

    def __post_init__(self) -> None:
        self.scatterers = [s if isinstance(s, PointScatterer) else PointScatterer(*s) for s in self.scatterers]
        for scatterer in self.scatterers:
            if scatterer.z_m <= 0.0:
                raise ValueError("All scatterers must have positive z_m.")

    @property
    def n_scatterers(self) -> int:
        """Return the number of scatterers in the phantom."""
        return len(self.scatterers)

    def as_array(self) -> np.ndarray:
        """Return scatterers as `[[x_m, z_m, amplitude], ...]` array."""
        if not self.scatterers:
            return np.zeros((0, 3), dtype=float)
        return np.asarray(
            [[s.x_m, s.z_m, s.amplitude] for s in self.scatterers],
            dtype=float,
        )


def single_point_phantom(
    x_m: float = 0.0,
    z_m: float = 0.03,
    amplitude: float = 1.0,
) -> PointScattererPhantom:
    """Create a one-target phantom used for baseline localization checks."""
    return PointScattererPhantom(
        scatterers=[PointScatterer(x_m=x_m, z_m=z_m, amplitude=amplitude)],
        metadata={"generator": "single_point_phantom"},
    )


def random_point_phantom(
    n_scatterers: int,
    x_range_m: tuple[float, float] = (-0.01, 0.01),
    z_range_m: tuple[float, float] = (0.015, 0.06),
    seed: int | None = None,
) -> PointScattererPhantom:
    """Create a uniform random point-scatterer phantom for synthetic studies."""
    n_scatterers = int(n_scatterers)
    if n_scatterers <= 0:
        raise ValueError("n_scatterers must be positive.")

    x_min, x_max = map(float, x_range_m)
    z_min, z_max = map(float, z_range_m)
    if x_max <= x_min:
        raise ValueError("x_range_m must be an increasing range.")
    if z_max <= z_min:
        raise ValueError("z_range_m must be an increasing range.")
    if z_min <= 0.0:
        raise ValueError("z_range_m must be strictly positive.")

    rng = np.random.default_rng(seed)
    xs = rng.uniform(x_min, x_max, size=n_scatterers)
    zs = rng.uniform(z_min, z_max, size=n_scatterers)
    amplitudes = np.ones(n_scatterers, dtype=float)
    scatterers = [PointScatterer(x_m=x, z_m=z, amplitude=a) for x, z, a in zip(xs, zs, amplitudes)]
    return PointScattererPhantom(
        scatterers=scatterers,
        metadata={"generator": "random_point_phantom", "seed": seed},
    )
