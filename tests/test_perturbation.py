import numpy as np
import pytest

from medical_ultrasound_systems.geometry import LinearArrayGeometry
from medical_ultrasound_systems.perturbation import (
    add_awgn,
    apply_gain_jitter,
    copy_rf_with_samples,
    drop_channels,
)
from medical_ultrasound_systems.phantom import single_point_phantom
from medical_ultrasound_systems.simulation import simulate_pulse_echo_rf


def _make_rf():
    geometry = LinearArrayGeometry(n_elements=8, pitch_m=0.0003)
    phantom = single_point_phantom(x_m=0.0, z_m=0.03)
    return simulate_pulse_echo_rf(geometry=geometry, phantom=phantom, duration_s=40e-6)


def test_add_awgn_preserves_shape_and_finite():
    rf = _make_rf()
    noisy = add_awgn(rf, snr_db=20.0, seed=7)
    assert noisy.samples.shape == rf.samples.shape
    assert np.isfinite(noisy.samples).all()


def test_drop_channels_zeros_at_least_one_channel():
    rf = _make_rf()
    dropped = drop_channels(rf, drop_fraction=0.2, seed=9)
    zero_channel_mask = np.all(np.isclose(dropped.samples, 0.0), axis=1)
    assert dropped.samples.shape == rf.samples.shape
    assert np.any(zero_channel_mask)


def test_apply_gain_jitter_preserves_shape():
    rf = _make_rf()
    jittered = apply_gain_jitter(rf, gain_std=0.05, seed=11)
    assert jittered.samples.shape == rf.samples.shape


def test_invalid_perturbation_parameters_raise():
    rf = _make_rf()
    with pytest.raises(ValueError):
        drop_channels(rf, drop_fraction=-0.1)
    with pytest.raises(ValueError):
        drop_channels(rf, drop_fraction=1.0)
    with pytest.raises(ValueError):
        apply_gain_jitter(rf, gain_std=-0.01)


def test_copy_rf_with_samples_applies_metadata_update():
    rf = _make_rf()
    copied = copy_rf_with_samples(
        rf=rf,
        samples=rf.samples.copy(),
        metadata_update={"test_tag": "phase3"},
    )
    assert copied.samples.shape == rf.samples.shape
    assert copied.metadata is not None
    assert copied.metadata["test_tag"] == "phase3"
