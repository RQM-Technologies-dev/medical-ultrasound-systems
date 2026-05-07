import numpy as np

from medical_ultrasound_systems.metrics import (
    correlation_coefficient,
    mse,
    normalized_cross_correlation,
    normalized_error,
    psnr,
)


def test_mse_and_normalized_error_basic_behavior():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 4.0])

    assert np.isclose(mse(a, b), 1.0 / 3.0)
    assert normalized_error(a, b) > 0.0
    assert np.isclose(normalized_error(a, a), 0.0)


def test_psnr_and_correlation_metrics_sanity():
    reference = np.array([0.0, 1.0, 2.0, 3.0], dtype=float)
    estimate = np.array([0.0, 1.0, 1.8, 2.9], dtype=float)

    assert np.isinf(psnr(reference, reference))
    assert psnr(reference, estimate) > 0.0
    assert np.isclose(correlation_coefficient(reference, reference), 1.0)
    assert correlation_coefficient(reference, -reference) < 0.0
    assert np.isclose(normalized_cross_correlation(reference, reference), 1.0)
