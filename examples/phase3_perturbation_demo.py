"""Phase 3 perturbation demo (synthetic, non-clinical, no plotting)."""

from __future__ import annotations

from medical_ultrasound_systems.experiments import single_point_comparison
from medical_ultrasound_systems.geometry import LinearArrayGeometry
from medical_ultrasound_systems.perturbations import perturb_rf_channel_data
from medical_ultrasound_systems.phantom import single_point_phantom
from medical_ultrasound_systems.simulation import simulate_pulse_echo_rf


def main() -> None:
    """Demonstrate one reproducible perturbation profile on synthetic RF data."""
    geometry = LinearArrayGeometry(n_elements=32, pitch_m=0.0003, center_frequency_hz=5e6)
    phantom = single_point_phantom(x_m=0.002, z_m=0.032, amplitude=1.0)
    rf = simulate_pulse_echo_rf(geometry=geometry, phantom=phantom)

    perturbation = {
        "snr_db": 20.0,
        "dropout_fraction": 0.25,
        "gain_std": 0.1,
        "max_jitter_samples": 1,
    }
    rf_perturbed = perturb_rf_channel_data(rf=rf, seed=7, **perturbation)
    result = single_point_comparison(perturbation=perturbation, seed=7)

    print("Phase 3 perturbation demo (synthetic benchmark candidate)")
    print("Research metrics only; these outputs require validation and are non-clinical.")
    print(f"RF shape before perturbation: {rf.samples.shape}")
    print(f"RF shape after perturbation:  {rf_perturbed.samples.shape}")
    print("Peak localization by method:")
    for peak in result.peaks:
        print(
            f"  - {peak.method}: "
            f"peak=({peak.peak_x_m:.6f}, {peak.peak_z_m:.6f}) m, "
            f"error={peak.localization_error_m:.6e} m, "
            f"value={peak.peak_value:.6e}"
        )


if __name__ == "__main__":
    main()
