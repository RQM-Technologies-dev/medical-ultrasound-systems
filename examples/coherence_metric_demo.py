"""Basic coherence metric demo."""

from medical_ultrasound_systems.coherence import coherence_score
from medical_ultrasound_systems.simulation import synthetic_plane_wave

reference = synthetic_plane_wave(n_samples=128)
observed = synthetic_plane_wave(n_samples=128)

print("Coherence:", coherence_score(reference.samples, observed.samples))
