import numpy as np

from medical_ultrasound_systems.delay import pixel_travel_times_plane_wave, sample_rf_nearest
from medical_ultrasound_systems.geometry import LinearArrayGeometry
from medical_ultrasound_systems.phantom import single_point_phantom
from medical_ultrasound_systems.simulation import simulate_pulse_echo_rf


def test_pixel_travel_times_are_positive():
    geometry = LinearArrayGeometry(n_elements=8, pitch_m=0.0003)
    travel_times_s = pixel_travel_times_plane_wave(geometry, x_m=0.0, z_m=0.03, sound_speed_m_s=1540.0)
    assert travel_times_s.shape == (geometry.n_elements,)
    assert np.all(travel_times_s > 0.0)


def test_sample_rf_nearest_returns_one_value_per_channel():
    geometry = LinearArrayGeometry(n_elements=8, pitch_m=0.0003)
    phantom = single_point_phantom(x_m=0.0, z_m=0.03)
    rf = simulate_pulse_echo_rf(geometry=geometry, phantom=phantom, duration_s=40e-6)
    travel_times_s = pixel_travel_times_plane_wave(geometry, x_m=0.0, z_m=0.03, sound_speed_m_s=1540.0)
    sampled = sample_rf_nearest(rf, travel_times_s)
    assert sampled.shape == (rf.n_channels,)


def test_sample_rf_nearest_out_of_range_returns_zeros():
    geometry = LinearArrayGeometry(n_elements=4, pitch_m=0.0003)
    phantom = single_point_phantom(x_m=0.0, z_m=0.03)
    rf = simulate_pulse_echo_rf(geometry=geometry, phantom=phantom, duration_s=20e-6)

    # Extremely large travel times force all channels out of the RF range.
    sampled = sample_rf_nearest(rf, np.full(rf.n_channels, 1.0))
    assert sampled.shape == (rf.n_channels,)
    assert np.allclose(sampled, 0.0)
