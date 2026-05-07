from medical_ultrasound_systems.simulation import synthetic_plane_wave
from medical_ultrasound_systems.wavefield import Wavefield


def test_synthetic_plane_wave_shape_and_container():
    wf = synthetic_plane_wave(n_samples=64)
    assert isinstance(wf, Wavefield)
    assert wf.samples.shape == (64, 4)
    assert wf.n_samples == 64
