import numpy as np

from medical_ultrasound_systems.experiments import (
    ExperimentResult,
    PeakResult,
    experiment_results_to_rows,
    find_peak,
    localization_error_m,
    single_point_comparison,
)


def test_find_peak_returns_correct_coordinates():
    x_grid_m = np.array([-0.001, 0.0, 0.001], dtype=float)
    z_grid_m = np.array([0.02, 0.03], dtype=float)
    image = np.array([[0.0, 0.1, 0.2], [0.5, 2.0, 0.3]], dtype=float)
    peak_x_m, peak_z_m, peak_value = find_peak(image, x_grid_m, z_grid_m)
    assert np.isclose(peak_x_m, 0.0)
    assert np.isclose(peak_z_m, 0.03)
    assert np.isclose(peak_value, 2.0)


def test_localization_error_m_computes_distance():
    error_m = localization_error_m(peak_x_m=0.001, peak_z_m=0.032, target_x_m=0.0, target_z_m=0.03)
    assert np.isclose(error_m, np.hypot(0.001, 0.002))


def test_single_point_comparison_includes_expected_method_names():
    result = single_point_comparison(
        x_grid_m=np.linspace(-0.005, 0.005, 16),
        z_grid_m=np.linspace(0.02, 0.04, 20),
        seed=21,
    )
    assert isinstance(result, ExperimentResult)
    methods = {peak.method for peak in result.peaks}
    assert methods == {
        "delay_and_sum_plane_wave",
        "conventional_coherence_image",
        "quaternionic_alignment_image",
        "quaternionic_intensity_image",
    }


def test_experiment_results_to_rows_returns_flat_rows():
    result = ExperimentResult(
        name="unit_test",
        parameters={"snr_db": 20.0, "trial_index": 0},
        peaks=[
            PeakResult(
                method="delay_and_sum_plane_wave",
                peak_x_m=0.0,
                peak_z_m=0.03,
                target_x_m=0.0,
                target_z_m=0.03,
                localization_error_m=0.0,
                peak_value=1.0,
            )
        ],
        runtime_s={"delay_and_sum_plane_wave": 0.01},
    )
    rows = experiment_results_to_rows([result])
    assert len(rows) == 1
    row = rows[0]
    assert row["experiment_name"] == "unit_test"
    assert row["method"] == "delay_and_sum_plane_wave"
    assert row["param_snr_db"] == 20.0
