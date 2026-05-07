import numpy as np

from medical_ultrasound_systems.analytic import (
    analytic_signal_fft,
    instantaneous_amplitude,
    instantaneous_phase,
)


def test_analytic_signal_fft_returns_complex_same_shape():
    signal = np.random.default_rng(0).standard_normal((4, 32))
    analytic = analytic_signal_fft(signal, axis=-1)
    assert analytic.shape == signal.shape
    assert np.iscomplexobj(analytic)


def test_instantaneous_phase_and_amplitude_are_finite():
    signal = np.random.default_rng(1).standard_normal((3, 64))
    phase = instantaneous_phase(signal, axis=-1)
    amplitude = instantaneous_amplitude(signal, axis=-1)

    assert phase.shape == signal.shape
    assert amplitude.shape == signal.shape
    assert np.isfinite(phase).all()
    assert np.isfinite(amplitude).all()
