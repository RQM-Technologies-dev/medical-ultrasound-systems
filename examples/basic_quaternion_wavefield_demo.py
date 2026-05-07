"""Basic quaternion wavefield generation demo."""

from medical_ultrasound_systems.simulation import synthetic_plane_wave

wave = synthetic_plane_wave(n_samples=16)
print("Samples shape:", wave.samples.shape)
print("First sample:", wave.samples[0])
