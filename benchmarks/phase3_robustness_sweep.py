"""Phase 3 robustness sweep for synthetic conventional and quaternionic methods."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from medical_ultrasound_systems.beamforming import delay_and_sum_plane_wave
from medical_ultrasound_systems.coherence import conventional_coherence_image
from medical_ultrasound_systems.evaluation import summarize_image_result
from medical_ultrasound_systems.geometry import LinearArrayGeometry
from medical_ultrasound_systems.perturbation import add_awgn, apply_gain_jitter, drop_channels
from medical_ultrasound_systems.phantom import single_point_phantom
from medical_ultrasound_systems.qbeamforming import quaternionic_alignment_image, quaternionic_intensity_image
from medical_ultrasound_systems.reporting import write_csv_report, write_json_report
from medical_ultrasound_systems.simulation import RFChannelData, simulate_pulse_echo_rf


def _format_value(value: float) -> str:
    if np.isinf(value):
        return "inf"
    return f"{value:.6f}"


def _print_table(records: list[dict]) -> None:
    headers = [
        "condition",
        "method",
        "peak_x_m",
        "peak_z_m",
        "localization_error_m",
        "peak_to_sidelobe_ratio_db",
        "runtime_s",
    ]
    print("Phase 3 robustness sweep (synthetic, non-clinical research)")
    print(" | ".join(f"{header:>24s}" for header in headers))
    print("-" * (len(headers) * 27))
    for record in records:
        print(
            " | ".join(
                [
                    f"{str(record['condition']):>24s}",
                    f"{str(record['method']):>24s}",
                    f"{_format_value(float(record['peak_x_m'])):>24s}",
                    f"{_format_value(float(record['peak_z_m'])):>24s}",
                    f"{_format_value(float(record['localization_error_m'])):>24s}",
                    f"{_format_value(float(record['peak_to_sidelobe_ratio_db'])):>24s}",
                    f"{_format_value(float(record['runtime_s'])):>24s}",
                ]
            )
        )


def _make_conditions(clean_rf: RFChannelData) -> list[tuple[str, RFChannelData]]:
    return [
        ("clean", clean_rf),
        ("noise_snr30_db", add_awgn(clean_rf, snr_db=30.0, seed=100)),
        ("noise_snr20_db", add_awgn(clean_rf, snr_db=20.0, seed=101)),
        ("dropout_10pct", drop_channels(clean_rf, drop_fraction=0.10, seed=200)),
        ("dropout_25pct", drop_channels(clean_rf, drop_fraction=0.25, seed=201)),
        ("gain_jitter_005", apply_gain_jitter(clean_rf, gain_std=0.05, seed=300)),
        ("gain_jitter_015", apply_gain_jitter(clean_rf, gain_std=0.15, seed=301)),
    ]


def main() -> None:
    target_x_m = 0.002
    target_z_m = 0.032
    geometry = LinearArrayGeometry(n_elements=32, pitch_m=0.0003, center_frequency_hz=5e6)
    phantom = single_point_phantom(x_m=target_x_m, z_m=target_z_m, amplitude=1.0)
    clean_rf = simulate_pulse_echo_rf(geometry=geometry, phantom=phantom)

    x_grid_m = np.linspace(-0.012, 0.012, 96)
    z_grid_m = np.linspace(0.01, 0.06, 128)
    conditions = _make_conditions(clean_rf)
    method_funcs = [
        ("das", delay_and_sum_plane_wave),
        ("conventional_coherence", conventional_coherence_image),
        ("quaternionic_alignment_analytic", lambda rf, x_grid_m, z_grid_m: quaternionic_alignment_image(rf, x_grid_m, z_grid_m, method="analytic")),
        ("quaternionic_intensity_analytic", lambda rf, x_grid_m, z_grid_m: quaternionic_intensity_image(rf, x_grid_m, z_grid_m, method="analytic")),
    ]

    records: list[dict] = []
    for condition_name, rf in conditions:
        for method_name, method_fn in method_funcs:
            t0 = time.perf_counter()
            image = method_fn(rf, x_grid_m, z_grid_m)
            runtime_s = time.perf_counter() - t0
            summary = summarize_image_result(
                name=method_name,
                image=image,
                x_grid_m=x_grid_m,
                z_grid_m=z_grid_m,
                target_x_m=target_x_m,
                target_z_m=target_z_m,
            )
            summary["condition"] = condition_name
            summary["runtime_s"] = float(runtime_s)
            records.append(summary)

    _print_table(records)

    reports_dir = Path("reports")
    write_json_report(str(reports_dir / "phase3_robustness_sweep.json"), records)
    write_csv_report(str(reports_dir / "phase3_robustness_sweep.csv"), records)
    print("Reports written to reports/phase3_robustness_sweep.json and .csv")


if __name__ == "__main__":
    main()
