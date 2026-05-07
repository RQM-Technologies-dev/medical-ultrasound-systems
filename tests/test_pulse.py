import numpy as np

from medical_ultrasound_systems.pulse import gaussian_modulated_pulse, normalize_pulse


def test_gaussian_modulated_pulse_shape_and_finite_values():
    t, pulse = gaussian_modulated_pulse()
    assert t.ndim == 1
    assert pulse.ndim == 1
    assert t.shape == pulse.shape
    assert t.size > 0
    assert np.isfinite(t).all()
    assert np.isfinite(pulse).all()

    pulse_n = normalize_pulse(pulse)
    assert np.isclose(np.max(np.abs(pulse_n)), 1.0)
