# Project overview

`medical-ultrasound-systems` is a research software repository for quaternionic and orientation-aware ultrasound signal-processing prototypes.

The repository focuses on simulation tooling, wavefield representation experiments, coherence metrics, and benchmark infrastructure for software-method research and reproducibility studies.

This repository is not for clinical use and requires validation before any downstream integration discussions.

## Phase 1 capability

Phase 1 introduces a lightweight synthetic pulse-echo baseline stack for research software evaluation:

- Linear array geometry definitions for centered 2D `(x, z)` element coordinates
- Point-scatterer phantom generators for controlled synthetic targets
- Gaussian-modulated pulse generation
- RF channel-data simulation using a simplified plane-wave pulse-echo approximation
- Baseline delay-and-sum beamforming for 2D reconstruction grids
- Envelope/log-compression utilities for image-domain baseline outputs
- Coherence and image-comparison metrics, including conventional channel coherence factor

## Phase 2 capability (prototype layer)

Phase 2 adds a quaternionic comparison layer:

- FFT analytic-signal lift for RF channels
- Quaternionic channel-wavefield encoding with orientation-axis proxies
- Quaternionic alignment and quaternionic intensity image baselines
- Direct comparison hooks against conventional coherence maps

These capabilities are for software-method research, benchmarking, and reproducibility experiments only.
