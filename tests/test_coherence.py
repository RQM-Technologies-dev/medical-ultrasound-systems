import numpy as np

from medical_ultrasound_systems.coherence import (
    channel_coherence_factor,
    coherence_score,
    quaternion_alignment_score,
)
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
