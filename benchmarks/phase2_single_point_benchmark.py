"""Phase 2 single-point synthetic benchmark for conventional vs quaternionic maps."""

from __future__ import annotations

import time

import numpy as np

from medical_ultrasound_systems.beamforming import delay_and_sum_plane_wave
from medical_ultrasound_systems.coherence import conventional_coherence_image
from medical_ultrasound_systems.geometry import LinearArrayGeometry
from medical_ultrasound_systems.phantom import single_point_phantom
from medical_ultrasound_systems.qbeamforming import (
    quaternionic_alignment_image,
    quaternionic_intensity_image,
)
from medical_ultrasound_systems.simulation import simulate_pulse_echo_rf


def _peak_position(image: np.ndarray, x_grid_m: np.ndarray, z_grid_m: np.ndarray) -> tuple[float, float]:
    idx = np.unravel_index(np.argmax(image), image.shape)
    return float(x_grid_m[idx[1]]), float(z_grid_m[idx[0]])


def _distance_m(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return float(np.hypot(p1[0] - p2[0], p1[1] - p2[1]))


def main() -> None:
    """Run a lightweight synthetic benchmark and print a text report."""
    phantom_x_m = 0.002
    phantom_z_m = 0.032

    t_start = time.perf_counter()
    geometry = LinearArrayGeometry(n_elements=32, pitch_m=0.0003, center_frequency_hz=5e6)
    phantom = single_point_phantom(x_m=phantom_x_m, z_m=phantom_z_m, amplitude=1.0)
    rf = simulate_pulse_echo_rf(geometry=geometry, phantom=phantom)
    t_setup = time.perf_counter()

    x_grid_m = np.linspace(-0.012, 0.012, 96)
    z_grid_m = np.linspace(0.01, 0.06, 128)
    target = (phantom_x_m, phantom_z_m)

    das_image = delay_and_sum_plane_wave(rf, x_grid_m=x_grid_m, z_grid_m=z_grid_m)
    t_das = time.perf_counter()
    coh_image = conventional_coherence_image(rf, x_grid_m=x_grid_m, z_grid_m=z_grid_m)
    t_coh = time.perf_counter()
    q_align = quaternionic_alignment_image(rf, x_grid_m=x_grid_m, z_grid_m=z_grid_m)
    t_q_align = time.perf_counter()
    q_int = quaternionic_intensity_image(rf, x_grid_m=x_grid_m, z_grid_m=z_grid_m)
    t_q_int = time.perf_counter()

    das_peak = _peak_position(das_image, x_grid_m, z_grid_m)
    coh_peak = _peak_position(coh_image, x_grid_m, z_grid_m)
    q_align_peak = _peak_position(q_align, x_grid_m, z_grid_m)
    q_int_peak = _peak_position(q_int, x_grid_m, z_grid_m)

    print("Phase 2 single-point synthetic benchmark (non-clinical)")
    print(f"Target position: x={target[0]:.4f} m, z={target[1]:.4f} m")
    print(f"DAS peak: x={das_peak[0]:.4f}, z={das_peak[1]:.4f}, error={_distance_m(das_peak, target):.6f} m")
    print(
        "Conventional coherence peak:"
        f" x={coh_peak[0]:.4f}, z={coh_peak[1]:.4f}, error={_distance_m(coh_peak, target):.6f} m"
    )
    print(
        "Quaternionic alignment peak:"
        f" x={q_align_peak[0]:.4f}, z={q_align_peak[1]:.4f}, error={_distance_m(q_align_peak, target):.6f} m"
    )
    print(
        "Quaternionic intensity peak:"
        f" x={q_int_peak[0]:.4f}, z={q_int_peak[1]:.4f}, error={_distance_m(q_int_peak, target):.6f} m"
    )
    print("Runtime summary (seconds):")
    print(f"  simulate+setup: {t_setup - t_start:.4f}")
    print(f"  delay-and-sum: {t_das - t_setup:.4f}")
    print(f"  coherence map: {t_coh - t_das:.4f}")
    print(f"  quaternionic alignment map: {t_q_align - t_coh:.4f}")
    print(f"  quaternionic intensity map: {t_q_int - t_q_align:.4f}")


if __name__ == "__main__":
    main()
