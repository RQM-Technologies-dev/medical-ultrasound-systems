import numpy as np

from medical_ultrasound_systems.beamforming import (
    delay_and_sum_plane_wave,
    envelope_detect_fft,
    log_compress,
)
from medical_ultrasound_systems.geometry import LinearArrayGeometry
from medical_ultrasound_systems.phantom import single_point_phantom
from medical_ultrasound_systems.simulation import simulate_pulse_echo_rf


def test_delay_and_sum_plane_wave_returns_image_shape():
    geometry = LinearArrayGeometry(n_elements=8, pitch_m=0.0003)
    phantom = single_point_phantom(z_m=0.03)
    rf = simulate_pulse_echo_rf(geometry=geometry, phantom=phantom, duration_s=40e-6)

    x_grid_m = np.linspace(-0.004, 0.004, 21)
    z_grid_m = np.linspace(0.01, 0.05, 31)
    image = delay_and_sum_plane_wave(rf=rf, x_grid_m=x_grid_m, z_grid_m=z_grid_m)

    assert image.shape == (z_grid_m.size, x_grid_m.size)


def test_envelope_detect_fft_nonnegative_and_log_compress_unit_range():
    signal = np.array([[0.0, 1.0, 0.0, -1.0], [1.0, 0.5, -0.5, -1.0]])
    envelope = envelope_detect_fft(signal, axis=-1)
    assert np.all(envelope >= 0.0)

    compressed = log_compress(signal)
    assert np.all(compressed >= 0.0)
    assert np.all(compressed <= 1.0)
