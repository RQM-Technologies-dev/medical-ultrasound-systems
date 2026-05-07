# medical-ultrasound-systems

Software research layer for quaternionic ultrasound wavefield analysis, coherence diagnostics, and synthetic benchmark evaluation.

## Executive summary

Ultrasound systems already produce rich RF channel data before final image formation. That RF stage contains amplitude, phase, channel-by-channel timing behavior, and array-geometry context that can be valuable for engineering analysis and algorithm development.

Conventional pipelines typically process amplitude, phase, channel delay, geometry, beamforming, and coherence through separate steps. This is effective for production imaging workflows, but it can make it harder to study some channel-level interactions in one unified software representation during research.

This repository explores whether a quaternionic representation can combine amplitude, phase, and receive geometry into a single software object for research diagnostics and benchmark comparisons. The intent is to create a candidate software layer that may support deeper technical evaluation of coherence and orientation structure in synthetic studies.

The proposed upgrade is software-only and sits after RF/channel acquisition, before or alongside beamforming/reconstruction evaluation. It does not replace probes, scanners, transducers, or certified clinical pipelines. This repository is not clinical software; it is a synthetic research and evidence-building repository.

## What this repo is

This repository gathers theory notes, software prototypes, synthetic simulation utilities, benchmark infrastructure, and validation guidance for evaluating quaternionic wavefield-analysis methods alongside conventional ultrasound processing baselines.

## What this repo is not

- Not a medical device.
- Not clinical software.
- Not FDA-cleared.
- Not for diagnosis, treatment, or patient-care decisions.
- Not a replacement for existing ultrasound hardware or certified reconstruction pipelines.

## Medical and regulatory disclaimer

This repository is for research and engineering exploration only. It is software-method and benchmark scaffolding for synthetic RF simulation, baseline beamforming, quaternionic wavefield analysis, and reproducibility studies. It is not for clinical use.

## What problem are we addressing?

Ultrasound image quality and robustness are influenced by phase alignment, channel coherence, multipath, reverberation, sidelobes, gain imbalance, channel dropout, and noise. These effects are deeply geometric because they depend on array layout, propagation paths, and channel timing relationships.

Conventional scalar or complex-only views can collapse some geometric structure early in the processing chain. That may reduce visibility into certain orientation-aware or channel-interaction effects during algorithm research.

The research question here is whether quaternionic wavefield representations can expose more useful coherence and orientation structure for software analysis, synthetic benchmarking, and candidate-method comparison.

## What is the proposed software upgrade?

Existing ultrasound system:

Probe / transducer array  
→ RF channel data  
→ beamforming  
→ image reconstruction  
→ display / analysis

RQM research layer:

RF channel data  
→ analytic signal  
→ quaternionic wavefield representation  
→ quaternionic alignment/coherence metrics  
→ comparison against conventional beamforming/coherence outputs

This is a software-only evaluation layer for research and engineering. It does not replace hardware, it is not used for clinical decision-making, and it is not an FDA-cleared workflow.

```text
Conventional ultrasound stack

Transducer array
   ↓
RF channel data
   ↓
Delay / phase processing
   ↓
Beamforming
   ↓
Image reconstruction
   ↓
Display / downstream analysis


RQM research upgrade layer

RF channel data
   ↓
Analytic amplitude + phase extraction
   ↓
Quaternionic wavefield lift
   ↓
Orientation-aware channel alignment
   ↓
Quaternionic coherence / intensity maps
   ↓
Benchmark comparison against conventional outputs
```

## What parts of the ultrasound stack does this touch?

- RF channel data handling (software-side representation only)
- analytic signal extraction and phase-aware transforms
- channel alignment and coherence diagnostics
- synthetic benchmark comparison against conventional beamforming/coherence outputs
- reporting of localization and runtime research metrics

It does not modify probe hardware, transducer physics, scanner firmware, or certified clinical workflow components.

## What evidence currently exists?

Current evidence in this repository is synthetic and engineering-focused:

- reproducible pulse-echo RF simulation and linear-array geometry modeling
- baseline delay-and-sum and conventional coherence reference methods
- quaternionic channel lift, alignment, and intensity map prototypes
- Phase 3 synthetic robustness experiments (noise, dropout, gain variation, timing jitter)
- benchmark outputs in JSON/CSV/Markdown for method comparison and auditability

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

## What validation would be needed before any real-world deployment?

Synthetic robustness tests are not clinical validation. These tests establish engineering behavior in controlled simulations only.

Any future real-world deployment pathway would require, at minimum:

- independent non-synthetic datasets with documented quality controls
- scanner and site variability studies across hardware/software configurations
- protocol-defined performance and failure-mode characterization
- regulatory strategy and compliance planning
- domain-expert review with clinical governance and safety oversight

This repository does not make efficacy or regulatory claims. It is a research prototype and synthetic benchmark environment that requires further validation.

## Run Phase 3 benchmark

```bash
python benchmarks/phase3_robustness_sweep.py
```

Outputs are written to:

- `benchmarks/output/phase3_results.json`
- `benchmarks/output/phase3_results.csv`
- `benchmarks/output/phase3_summary.md`

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
