import numpy as np

from medical_ultrasound_systems.metrics import mse, normalized_error


def test_mse_and_normalized_error_basic_behavior():
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 4.0])

    assert np.isclose(mse(a, b), 1.0 / 3.0)
    assert normalized_error(a, b) > 0.0
    assert np.isclose(normalized_error(a, a), 0.0)
