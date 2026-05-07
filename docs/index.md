# Project overview

`medical-ultrasound-systems` is a research software repository for quaternionic and orientation-aware ultrasound signal-processing prototypes.

The repository focuses on simulation tooling, wavefield representation experiments, coherence metrics, and benchmark infrastructure that may support technical evaluation by ultrasound OEMs and research collaborators.

This repository is not for clinical use and requires validation before any downstream integration discussions.

## Phase 1 capability

Phase 1 introduces a lightweight synthetic pulse-echo baseline stack for research software evaluation:

- Linear array geometry definitions for centered 2D `(x, z)` element coordinates
- Point-scatterer phantom generators for controlled synthetic targets
- RF channel-data simulation using a simplified plane-wave pulse-echo approximation
- Baseline delay-and-sum beamforming for 2D reconstruction grids
- Envelope/log-compression utilities for image-domain baseline outputs
- Coherence and error metrics, including conventional channel coherence factor and quaternion alignment score

These capabilities are for software-method research, benchmarking, and reproducibility experiments only.
