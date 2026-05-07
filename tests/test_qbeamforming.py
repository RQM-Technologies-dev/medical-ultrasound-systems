import pytest
import numpy as np

from medical_ultrasound_systems.geometry import LinearArrayGeometry
from medical_ultrasound_systems.phantom import single_point_phantom
from medical_ultrasound_systems.qbeamforming import (
    quaternionic_alignment_factor,
    quaternionic_alignment_image,
    quaternionic_delay_align_pixel_analytic,
    quaternionic_intensity_image,
)
from medical_ultrasound_systems.simulation import simulate_pulse_echo_rf


def test_quaternionic_alignment_factor_identical_values_near_one():
    q_values = np.tile(np.array([[1.0, 0.0, 0.0, 0.0]]), (8, 1))
    assert np.isclose(quaternionic_alignment_factor(q_values), 1.0)


def test_quaternionic_alignment_and_intensity_image_shapes_and_finite():
    geometry = LinearArrayGeometry(n_elements=8, pitch_m=0.0003)
    phantom = single_point_phantom(x_m=0.0, z_m=0.03)
    rf = simulate_pulse_echo_rf(geometry=geometry, phantom=phantom, duration_s=40e-6)
    x_grid_m = np.linspace(-0.004, 0.004, 9)
    z_grid_m = np.linspace(0.01, 0.05, 11)

    q_align = quaternionic_alignment_image(
        rf, x_grid_m=x_grid_m, z_grid_m=z_grid_m, method="analytic"
    )
    q_intensity = quaternionic_intensity_image(
        rf, x_grid_m=x_grid_m, z_grid_m=z_grid_m, method="analytic"
    )

    assert q_align.shape == (z_grid_m.size, x_grid_m.size)
    assert q_intensity.shape == (z_grid_m.size, x_grid_m.size)
    assert np.isfinite(q_intensity).all()


def test_quaternionic_delay_align_pixel_analytic_shape():
    geometry = LinearArrayGeometry(n_elements=8, pitch_m=0.0003)
    phantom = single_point_phantom(x_m=0.0, z_m=0.03)
    rf = simulate_pulse_echo_rf(geometry=geometry, phantom=phantom, duration_s=40e-6)
    q_values = quaternionic_delay_align_pixel_analytic(rf=rf, x_m=0.0, z_m=0.03)
    assert q_values.shape == (rf.n_channels, 4)


def test_quaternionic_alignment_image_signed_method_still_works():
    geometry = LinearArrayGeometry(n_elements=8, pitch_m=0.0003)
    phantom = single_point_phantom(x_m=0.0, z_m=0.03)
    rf = simulate_pulse_echo_rf(geometry=geometry, phantom=phantom, duration_s=40e-6)
    x_grid_m = np.linspace(-0.004, 0.004, 9)
    z_grid_m = np.linspace(0.01, 0.05, 11)

    analytic = quaternionic_alignment_image(rf, x_grid_m=x_grid_m, z_grid_m=z_grid_m, method="analytic")
    signed = quaternionic_alignment_image(rf, x_grid_m=x_grid_m, z_grid_m=z_grid_m, method="signed")
    signed_intensity = quaternionic_intensity_image(
        rf, x_grid_m=x_grid_m, z_grid_m=z_grid_m, method="signed"
    )

    assert analytic.shape == signed.shape == (z_grid_m.size, x_grid_m.size)
    assert signed_intensity.shape == (z_grid_m.size, x_grid_m.size)
    assert np.isfinite(signed).all()
    assert np.isfinite(signed_intensity).all()


def test_quaternionic_alignment_unknown_method_raises():
    geometry = LinearArrayGeometry(n_elements=8, pitch_m=0.0003)
    phantom = single_point_phantom(x_m=0.0, z_m=0.03)
    rf = simulate_pulse_echo_rf(geometry=geometry, phantom=phantom, duration_s=40e-6)
    x_grid_m = np.linspace(-0.004, 0.004, 9)
    z_grid_m = np.linspace(0.01, 0.05, 11)

    with pytest.raises(ValueError):
        quaternionic_alignment_image(rf, x_grid_m=x_grid_m, z_grid_m=z_grid_m, method="unknown")
    with pytest.raises(ValueError):
        quaternionic_intensity_image(rf, x_grid_m=x_grid_m, z_grid_m=z_grid_m, method="unknown")
