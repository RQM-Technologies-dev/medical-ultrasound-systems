import numpy as np

from medical_ultrasound_systems.geometry import LinearArrayGeometry
from medical_ultrasound_systems.phantom import single_point_phantom
from medical_ultrasound_systems.qwavefield import (
    make_pixel_orientation_axes,
    rf_to_quaternionic_channels,
)
from medical_ultrasound_systems.simulation import simulate_pulse_echo_rf


def test_rf_to_quaternionic_channels_shape():
    geometry = LinearArrayGeometry(n_elements=8, pitch_m=0.0003)
    phantom = single_point_phantom(x_m=0.0, z_m=0.03)
    rf = simulate_pulse_echo_rf(geometry=geometry, phantom=phantom, duration_s=40e-6)

    qwf = rf_to_quaternionic_channels(rf)
    assert qwf.samples.shape == (rf.n_channels, rf.n_samples, 4)


def test_make_pixel_orientation_axes_returns_unit_vectors():
    geometry = LinearArrayGeometry(n_elements=8, pitch_m=0.0003)
    axes = make_pixel_orientation_axes(geometry, x_m=0.001, z_m=0.03)
    norms = np.linalg.norm(axes, axis=1)
    assert axes.shape == (geometry.n_elements, 3)
    assert np.allclose(norms, 1.0)
