import numpy as np

from medical_ultrasound_systems.evaluation import (
    PeakResult,
    find_peak,
    localization_error_m,
    summarize_image_result,
)


def test_find_peak_returns_expected_coordinate():
    x_grid_m = np.array([-0.001, 0.0, 0.001], dtype=float)
    z_grid_m = np.array([0.02, 0.03], dtype=float)
    image = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 1.5, 0.7],
        ]
    )
    peak = find_peak(image, x_grid_m=x_grid_m, z_grid_m=z_grid_m)
    assert np.isclose(peak.x_m, 0.0)
    assert np.isclose(peak.z_m, 0.03)
    assert np.isclose(peak.value, 1.5)


def test_localization_error_m_computes_distance():
    peak = PeakResult(value=1.0, x_m=0.001, z_m=0.032, index_z=0, index_x=0)
    err = localization_error_m(peak, target_x_m=0.0, target_z_m=0.03)
    assert np.isclose(err, np.hypot(0.001, 0.002))


def test_summarize_image_result_expected_keys():
    x_grid_m = np.array([-0.001, 0.0, 0.001], dtype=float)
    z_grid_m = np.array([0.02, 0.03], dtype=float)
    image = np.array([[0.1, 0.2, 0.3], [0.4, 1.0, 0.7]], dtype=float)
    summary = summarize_image_result(
        name="test_method",
        image=image,
        x_grid_m=x_grid_m,
        z_grid_m=z_grid_m,
        target_x_m=0.0,
        target_z_m=0.03,
    )
    expected_keys = {
        "method",
        "peak_value",
        "peak_x_m",
        "peak_z_m",
        "localization_error_m",
        "peak_to_sidelobe_ratio_db",
    }
    assert expected_keys.issubset(summary.keys())
