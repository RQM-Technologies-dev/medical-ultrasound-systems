"""Phase 2 synthetic quaternionic-alignment demo (research only)."""

from __future__ import annotations

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


def _peak_location(image: np.ndarray, x_grid_m: np.ndarray, z_grid_m: np.ndarray) -> tuple[float, float, float]:
    peak_idx = np.unravel_index(np.argmax(image), image.shape)
    return float(image[peak_idx]), float(x_grid_m[peak_idx[1]]), float(z_grid_m[peak_idx[0]])


def main() -> None:
    """Run synthetic Phase 2 comparisons without plotting."""
    geometry = LinearArrayGeometry(n_elements=32, pitch_m=0.0003, center_frequency_hz=5e6)
    phantom = single_point_phantom(x_m=0.0, z_m=0.03, amplitude=1.0)
    rf = simulate_pulse_echo_rf(geometry=geometry, phantom=phantom)

    x_grid_m = np.linspace(-0.012, 0.012, 96)
    z_grid_m = np.linspace(0.01, 0.06, 128)

    das_image = delay_and_sum_plane_wave(rf, x_grid_m=x_grid_m, z_grid_m=z_grid_m)
    coh_image = conventional_coherence_image(rf, x_grid_m=x_grid_m, z_grid_m=z_grid_m)
    q_align_image = quaternionic_alignment_image(rf, x_grid_m=x_grid_m, z_grid_m=z_grid_m)
    q_intensity_image = quaternionic_intensity_image(rf, x_grid_m=x_grid_m, z_grid_m=z_grid_m)

    coh_peak = _peak_location(coh_image, x_grid_m, z_grid_m)
    q_align_peak = _peak_location(q_align_image, x_grid_m, z_grid_m)
    q_int_peak = _peak_location(q_intensity_image, x_grid_m, z_grid_m)

    print("Synthetic research demo (non-clinical software)")
    print("RF shape:", rf.samples.shape)
    print("DAS image shape:", das_image.shape)
    print(
        f"Conventional coherence max={coh_peak[0]:.4f} at x={coh_peak[1]:.4f} m, z={coh_peak[2]:.4f} m"
    )
    print(
        f"Quaternionic alignment max={q_align_peak[0]:.4f} "
        f"at x={q_align_peak[1]:.4f} m, z={q_align_peak[2]:.4f} m"
    )
    print(
        f"Quaternionic intensity max={q_int_peak[0]:.4f} at x={q_int_peak[1]:.4f} m, z={q_int_peak[2]:.4f} m"
    )


if __name__ == "__main__":
    main()
