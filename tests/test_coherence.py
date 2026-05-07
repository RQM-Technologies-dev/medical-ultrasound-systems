import numpy as np

from medical_ultrasound_systems.coherence import (
    channel_coherence_factor,
    coherence_score,
    conventional_coherence_image,
    quaternion_alignment_score,
)
from medical_ultrasound_systems.geometry import LinearArrayGeometry
from medical_ultrasound_systems.phantom import single_point_phantom
from medical_ultrasound_systems.simulation import simulate_pulse_echo_rf
from medical_ultrasound_systems.simulation import synthetic_plane_wave


def test_coherence_identical_is_one():
    wf = synthetic_plane_wave(n_samples=128)
    score = coherence_score(wf.samples, wf.samples.copy())
    assert np.isclose(score, 1.0)


def test_channel_coherence_factor_constant_channels_is_one():
    channels = np.ones((8, 32), dtype=float)
    cf = channel_coherence_factor(channels, axis=0)
    assert np.allclose(cf, 1.0)


def test_quaternion_alignment_score_wraps_coherence_score():
    wf = synthetic_plane_wave(n_samples=32)
    assert np.isclose(
        quaternion_alignment_score(wf.samples, wf.samples),
        coherence_score(wf.samples, wf.samples),
    )


def test_conventional_coherence_image_shape():
    geometry = LinearArrayGeometry(n_elements=8, pitch_m=0.0003)
    phantom = single_point_phantom(z_m=0.03)
    rf = simulate_pulse_echo_rf(geometry=geometry, phantom=phantom, duration_s=40e-6)
    x_grid_m = np.linspace(-0.004, 0.004, 9)
    z_grid_m = np.linspace(0.01, 0.05, 11)
    image = conventional_coherence_image(rf, x_grid_m=x_grid_m, z_grid_m=z_grid_m)
    assert image.shape == (z_grid_m.size, x_grid_m.size)
