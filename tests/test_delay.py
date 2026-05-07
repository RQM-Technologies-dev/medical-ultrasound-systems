import numpy as np

from medical_ultrasound_systems.delay import (
    pixel_travel_times_plane_wave,
    sample_array_linear_per_channel,
    sample_rf_linear,
    sample_rf_nearest,
)
from medical_ultrasound_systems.geometry import LinearArrayGeometry
from medical_ultrasound_systems.phantom import single_point_phantom
from medical_ultrasound_systems.simulation import RFChannelData, simulate_pulse_echo_rf


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


def test_sample_rf_linear_matches_nearest_at_integer_indices():
    geometry = LinearArrayGeometry(n_elements=3, pitch_m=0.0003)
    samples = np.array(
        [
            [0.0, 1.0, 2.0, 3.0],
            [5.0, 6.0, 7.0, 8.0],
            [10.0, 11.0, 12.0, 13.0],
        ]
    )
    rf = RFChannelData(samples=samples, sample_rate_hz=1.0, geometry=geometry)
    travel_times = np.array([1.0, 2.0, 3.0])
    nearest = sample_rf_nearest(rf, travel_times)
    linear = sample_rf_linear(rf, travel_times)
    assert np.allclose(linear, nearest)


def test_sample_array_linear_per_channel_halfway_interpolates():
    samples = np.array(
        [
            [0.0, 2.0, 4.0],
            [10.0, 20.0, 30.0],
        ],
        dtype=float,
    )
    travel_times = np.array([0.5, 1.5], dtype=float)
    sampled = sample_array_linear_per_channel(
        samples=samples,
        sample_rate_hz=1.0,
        travel_times_s=travel_times,
    )
    assert np.allclose(sampled, np.array([1.0, 25.0]))


def test_sample_array_linear_per_channel_out_of_range_returns_zeros():
    samples = np.array(
        [
            [1.0 + 1.0j, 2.0 + 1.0j, 3.0 + 1.0j],
            [4.0 + 2.0j, 5.0 + 2.0j, 6.0 + 2.0j],
        ],
        dtype=np.complex128,
    )
    sampled = sample_array_linear_per_channel(
        samples=samples,
        sample_rate_hz=1.0,
        travel_times_s=np.array([-1.0, 10.0]),
    )
    assert sampled.shape == (2,)
    assert np.allclose(sampled, 0.0)
