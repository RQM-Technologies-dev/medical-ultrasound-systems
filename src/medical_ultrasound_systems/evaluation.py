"""Image-level evaluation helpers for synthetic benchmark comparisons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PeakResult:
    value: float
    x_m: float
    z_m: float
    index_z: int
    index_x: int


def find_peak(image: np.ndarray, x_grid_m: np.ndarray, z_grid_m: np.ndarray) -> PeakResult:
    """Find the maximum-magnitude image location and corresponding coordinates."""
    image = np.asarray(image)
    x_grid_m = np.asarray(x_grid_m, dtype=float)
    z_grid_m = np.asarray(z_grid_m, dtype=float)
    if image.ndim != 2:
        raise ValueError("image must be 2D.")
    if x_grid_m.ndim != 1 or z_grid_m.ndim != 1:
        raise ValueError("x_grid_m and z_grid_m must be 1D.")
    if image.shape != (z_grid_m.size, x_grid_m.size):
        raise ValueError("image shape must match (len(z_grid_m), len(x_grid_m)).")
    if image.size == 0:
        raise ValueError("image must be non-empty.")

    magnitude = np.abs(image)
    peak_idx = np.unravel_index(np.argmax(magnitude), magnitude.shape)
    index_z, index_x = int(peak_idx[0]), int(peak_idx[1])
    return PeakResult(
        value=float(magnitude[index_z, index_x]),
        x_m=float(x_grid_m[index_x]),
        z_m=float(z_grid_m[index_z]),
        index_z=index_z,
        index_x=index_x,
    )


def localization_error_m(peak: PeakResult, target_x_m: float, target_z_m: float) -> float:
    """Return Euclidean localization error in meters."""
    return float(np.hypot(peak.x_m - float(target_x_m), peak.z_m - float(target_z_m)))


def peak_to_sidelobe_ratio_db(
    image: np.ndarray,
    peak: PeakResult,
    exclusion_radius_px: int = 3,
) -> float:
    """Compute peak-to-sidelobe ratio (dB) after masking a local peak neighborhood."""
    image = np.asarray(image)
    if image.ndim != 2:
        raise ValueError("image must be 2D.")
    exclusion_radius_px = int(exclusion_radius_px)
    if exclusion_radius_px < 0:
        raise ValueError("exclusion_radius_px must be non-negative.")

    magnitude = np.abs(image).astype(float, copy=False)
    peak_mag = float(magnitude[peak.index_z, peak.index_x])

    zz, xx = np.indices(magnitude.shape)
    mask = (zz - peak.index_z) ** 2 + (xx - peak.index_x) ** 2 <= exclusion_radius_px**2
    sidelobes = magnitude[~mask]
    if sidelobes.size == 0:
        return float("inf")

    max_sidelobe = float(np.max(sidelobes))
    if max_sidelobe <= np.finfo(float).eps:
        return float("inf")

    ratio = max(peak_mag, np.finfo(float).eps) / max(max_sidelobe, np.finfo(float).eps)
    return float(20.0 * np.log10(ratio))


def summarize_image_result(
    name: str,
    image: np.ndarray,
    x_grid_m: np.ndarray,
    z_grid_m: np.ndarray,
    target_x_m: float,
    target_z_m: float,
) -> dict:
    """Summarize key localization and peak-quality metrics for one image."""
    peak = find_peak(image=image, x_grid_m=x_grid_m, z_grid_m=z_grid_m)
    return {
        "method": str(name),
        "peak_value": float(peak.value),
        "peak_x_m": float(peak.x_m),
        "peak_z_m": float(peak.z_m),
        "localization_error_m": localization_error_m(
            peak=peak,
            target_x_m=float(target_x_m),
            target_z_m=float(target_z_m),
        ),
        "peak_to_sidelobe_ratio_db": peak_to_sidelobe_ratio_db(image=image, peak=peak),
    }
