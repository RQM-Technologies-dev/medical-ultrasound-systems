"""Synthetic Phase 1 pulse-echo demo for research software evaluation."""

from __future__ import annotations

import numpy as np

from medical_ultrasound_systems.beamforming import delay_and_sum_plane_wave, log_compress
from medical_ultrasound_systems.geometry import LinearArrayGeometry
from medical_ultrasound_systems.phantom import single_point_phantom
from medical_ultrasound_systems.simulation import simulate_pulse_echo_rf


def main() -> None:
    """Run a lightweight synthetic pulse-echo pipeline demo."""
    geometry = LinearArrayGeometry(n_elements=32, pitch_m=0.0003, center_frequency_hz=5e6)
    phantom = single_point_phantom(x_m=0.0, z_m=0.03, amplitude=1.0)

    rf = simulate_pulse_echo_rf(
        geometry=geometry,
        phantom=phantom,
        sample_rate_hz=40e6,
        center_frequency_hz=5e6,
        duration_s=80e-6,
    )

    x_grid_m = np.linspace(-0.012, 0.012, 128)
    z_grid_m = np.linspace(0.01, 0.06, 192)
    image = delay_and_sum_plane_wave(rf=rf, x_grid_m=x_grid_m, z_grid_m=z_grid_m)
    image_log = log_compress(image)

    peak_idx = np.unravel_index(np.argmax(image_log), image_log.shape)
    peak_z_m = z_grid_m[peak_idx[0]]
    peak_x_m = x_grid_m[peak_idx[1]]

    print("Synthetic research demo (non-clinical)")
    print("RF shape:", rf.samples.shape)
    print("Image shape:", image_log.shape)
    print("Max image value:", float(np.max(image_log)))
    print(f"Approximate max response location: x={peak_x_m:.4f} m, z={peak_z_m:.4f} m")


if __name__ == "__main__":
    main()
