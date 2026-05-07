import numpy as np

from medical_ultrasound_systems.geometry import LinearArrayGeometry
from medical_ultrasound_systems.perturbations import (
    add_awgn,
    apply_channel_gain_variation,
    apply_timing_jitter_nearest,
    dropout_channels,
    perturb_rf_channel_data,
)
from medical_ultrasound_systems.phantom import single_point_phantom
from medical_ultrasound_systems.simulation import RFChannelData, simulate_pulse_echo_rf


def _make_rf() -> RFChannelData:
    geometry = LinearArrayGeometry(n_elements=8, pitch_m=0.0003, center_frequency_hz=5e6)
    phantom = single_point_phantom(x_m=0.001, z_m=0.03, amplitude=1.0)
    return simulate_pulse_echo_rf(geometry=geometry, phantom=phantom, duration_s=40e-6)


def test_add_awgn_preserves_shape_and_changes_nonzero_signal():
    rf = _make_rf()
    perturbed = add_awgn(rf.samples, snr_db=20.0, seed=5)
    assert perturbed.shape == rf.samples.shape
    assert not np.allclose(perturbed, rf.samples)


def test_dropout_channels_returns_mask_and_preserves_shape():
    rf = _make_rf()
    perturbed, mask = dropout_channels(rf.samples, dropout_fraction=0.25, seed=7)
    assert perturbed.shape == rf.samples.shape
    assert mask.shape == (rf.n_channels,)
    assert mask.dtype == bool


def test_apply_channel_gain_variation_returns_gains():
    rf = _make_rf()
    perturbed, gains = apply_channel_gain_variation(rf.samples, gain_std=0.1, seed=9)
    assert perturbed.shape == rf.samples.shape
    assert gains.shape == (rf.n_channels,)
    assert np.all(gains >= 0.0)


def test_apply_timing_jitter_nearest_preserves_shape():
    rf = _make_rf()
    perturbed, shifts = apply_timing_jitter_nearest(rf.samples, max_jitter_samples=2, seed=11)
    assert perturbed.shape == rf.samples.shape
    assert shifts.shape == (rf.n_channels,)


def test_perturb_rf_channel_data_preserves_rf_shape():
    rf = _make_rf()
    perturbed = perturb_rf_channel_data(
        rf,
        snr_db=20.0,
        dropout_fraction=0.25,
        gain_std=0.1,
        max_jitter_samples=1,
        seed=13,
    )
    assert isinstance(perturbed, RFChannelData)
    assert perturbed.samples.shape == rf.samples.shape
