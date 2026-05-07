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

## Medical and regulatory disclaimer

This repository is for research and engineering exploration only. It is software-method and benchmark scaffolding for synthetic RF simulation, baseline beamforming, quaternionic wavefield analysis, and reproducibility studies. It is not for clinical use.

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

### Phase 1: Synthetic pulse-echo baseline

- synthetic pulse-echo RF simulation
- linear array geometry
- point-scatterer phantom
- baseline delay-and-sum beamforming
- conventional coherence metrics

### Phase 2: Quaternionic comparison layer (research prototype)

- FFT analytic-signal extraction
- quaternionic RF channel lift
- pixel-specific orientation axes
- quaternionic alignment image
- quaternionic intensity image

### Phase 3: Robustness and reproducible reports

- additive noise experiments
- channel dropout experiments
- gain/timing perturbation experiments
- localization-error reporting
- JSON/CSV/Markdown benchmark outputs

## Run Phase 3 benchmark

```bash
python benchmarks/phase3_robustness_sweep.py
```

Outputs are written to:

- `benchmarks/output/phase3_results.json`
- `benchmarks/output/phase3_results.csv`
- `benchmarks/output/phase3_summary.md`

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

from medical_ultrasound_systems.beamforming import delay_and_sum_plane_wave, log_compress
from medical_ultrasound_systems.geometry import LinearArrayGeometry
from medical_ultrasound_systems.phantom import single_point_phantom
from medical_ultrasound_systems.simulation import simulate_pulse_echo_rf

geometry = LinearArrayGeometry(n_elements=32, pitch_m=0.0003, center_frequency_hz=5e6)
phantom = single_point_phantom(x_m=0.0, z_m=0.03, amplitude=1.0)

rf = simulate_pulse_echo_rf(
    geometry=geometry,
    phantom=phantom,
    sample_rate_hz=40e6,
    center_frequency_hz=5e6,
    duration_s=80e-6,
)

x_grid_m = np.linspace(-0.012, 0.012, 96)
z_grid_m = np.linspace(0.01, 0.06, 128)

das = delay_and_sum_plane_wave(rf=rf, x_grid_m=x_grid_m, z_grid_m=z_grid_m)
image = log_compress(das)

print("RF shape:", rf.samples.shape)
print("Image shape:", image.shape)
```

## RQM Technologies positioning

RQM Technologies develops geometry-native software methods for wave analysis, quantum systems, sensing, imaging, and signal-processing workflows. RQM Technologies is not an ultrasound hardware manufacturer.
