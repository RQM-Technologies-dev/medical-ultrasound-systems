"""medical_ultrasound_systems package."""

from .coherence import channel_coherence_factor, coherence_score, quaternion_alignment_score
from .geometry import LinearArrayGeometry
from .phantom import PointScatterer, PointScattererPhantom
from .simulation import RFChannelData, simulate_pulse_echo_rf, synthetic_plane_wave
from .wavefield import Wavefield

__version__ = "0.1.0"

__all__ = [
    "Wavefield",
    "coherence_score",
    "channel_coherence_factor",
    "quaternion_alignment_score",
    "synthetic_plane_wave",
    "RFChannelData",
    "simulate_pulse_echo_rf",
    "LinearArrayGeometry",
    "PointScatterer",
    "PointScattererPhantom",
    "__version__",
]
