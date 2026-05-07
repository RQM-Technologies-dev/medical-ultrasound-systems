import numpy as np

from medical_ultrasound_systems.geometry import LinearArrayGeometry
from medical_ultrasound_systems.phantom import single_point_phantom
from medical_ultrasound_systems.simulation import RFChannelData, simulate_pulse_echo_rf


def test_simulate_pulse_echo_rf_returns_rf_channel_data_shape():
    geometry = LinearArrayGeometry(n_elements=8, pitch_m=0.0003)
    phantom = single_point_phantom(z_m=0.03)

    rf = simulate_pulse_echo_rf(
        geometry=geometry,
        phantom=phantom,
        sample_rate_hz=20e6,
        center_frequency_hz=4e6,
        duration_s=40e-6,
    )

    assert isinstance(rf, RFChannelData)
    assert rf.samples.ndim == 2
    assert rf.samples.shape[0] == geometry.n_elements
    assert rf.n_channels == geometry.n_elements
    assert rf.n_samples == int(np.ceil(40e-6 * 20e6))
