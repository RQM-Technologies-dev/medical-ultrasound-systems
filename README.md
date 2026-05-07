# medical-ultrasound-systems

Research software for exploring quaternionic and orientation-aware signal-processing methods for medical ultrasound systems.

## What this repo is

This repository gathers theory notes, software prototypes, simulation utilities, benchmarking infrastructure, and validation materials for studying whether quaternionic representations may improve ultrasound wave analysis and reconstruction workflows.

## What this repo is not

- Not a medical device.
- Not clinical software.
- Not FDA-cleared.
- Not for diagnosis, treatment, or patient-care decisions.
- Not a replacement for existing ultrasound hardware or certified reconstruction pipelines.

## Motivation

Conventional ultrasound pipelines often separate amplitude, phase, channel geometry, beamforming, and reconstruction into isolated steps. This project explores whether quaternionic representations can encode richer orientation, phase, and wavefield structure in a unified software object, enabling better research diagnostics, coherence scoring, artifact detection, and reconstruction experiments.

## Core research hypothesis

Quaternionic and QSG-inspired software methods may provide a useful layer for representing ultrasound wavefields with phase, orientation, polarization-like directional structure, channel geometry, and coherence in a single mathematical object.

## Initial technical modules

- `quaternion.py`: lightweight quaternion math utilities
- `wavefield.py`: synthetic ultrasound wavefield containers
- `coherence.py`: coherence and alignment metrics
- `beamforming.py`: baseline delay-and-sum beamforming and post-processing
- `reconstruction.py`: reconstruction experiment interfaces
- `metrics.py`: benchmark metrics
- `simulation.py`: synthetic phantom and channel simulation helpers

## Current capability

### Phase 1

- Synthetic pulse-echo RF simulation
- Linear-array geometry
- Point-scatterer phantom generation
- Baseline delay-and-sum beamforming
- Conventional coherence metrics

### Phase 2

- FFT analytic-signal extraction
- Quaternionic RF channel lift
- Pixel-specific orientation-axis encoding
- Quaternionic alignment image
- Conventional-vs-quaternionic synthetic benchmark scaffolding

## Near-term roadmap

- Synthetic wavefield simulation
- Quaternionic channel representation
- Baseline delay-and-sum comparison hooks
- Coherence metric development
- Artifact and multipath diagnostics
- Benchmark notebook examples
- Partner/OEM validation pathway

## Installation

```bash
pip install -e ".[dev]"
```

## Quickstart

```python
import numpy as np

from medical_ultrasound_systems.coherence import coherence_score
from medical_ultrasound_systems.simulation import synthetic_plane_wave

reference = synthetic_plane_wave(n_samples=128)
observed = synthetic_plane_wave(n_samples=128)

score = coherence_score(reference.samples, observed.samples)
print(f"Placeholder coherence score: {score:.3f}")
```

## Medical and regulatory disclaimer

This repository is for research and engineering exploration only. It is a benchmark candidate and software-method prototype set that requires further validation. It is not for clinical use.

## RQM Technologies positioning

RQM Technologies develops geometry-native software methods for wave analysis, quantum systems, sensing, imaging, and signal-processing workflows.
