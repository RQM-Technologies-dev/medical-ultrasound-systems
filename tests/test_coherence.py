import numpy as np

from medical_ultrasound_systems.coherence import coherence_score
from medical_ultrasound_systems.simulation import synthetic_plane_wave


def test_coherence_identical_is_one():
    wf = synthetic_plane_wave(n_samples=128)
    score = coherence_score(wf.samples, wf.samples.copy())
    assert np.isclose(score, 1.0)
