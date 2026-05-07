# Planned benchmarks

Initial benchmark tracks include:

- Delay-and-sum baseline comparisons
- Coherence score stability under perturbations
- Artifact sensitivity and multipath diagnostics
- Reconstruction quality metrics for synthetic scenarios
- Compute cost and scaling behavior

## Phase 1 initial benchmark plan

- Single point-target localization against known phantom coordinates
- Multiple-scatterer reconstruction consistency on synthetic phantoms
- Noise sensitivity under additive RF perturbations
- Channel-dropout robustness and graceful baseline degradation
- Conventional coherence factor versus quaternionic alignment metric comparison
- Runtime scaling versus `n_elements`, RF duration, and image-grid size

## Phase 2 benchmark plan

- Single-point localization with conventional and quaternionic map comparisons
- Quaternionic alignment vs conventional coherence behavior
- Robustness to channel dropout in synthetic RF channels
- Robustness to additive RF noise perturbations
- Multiple-scatterer ambiguity analysis on controlled phantoms
- Runtime scaling with `n_elements` and image-grid size

## Phase 3 robustness sweep

Phase 3 adds a no-plot synthetic robustness sweep with reproducible reports:

- clean baseline
- additive-noise SNR perturbations
- channel-dropout perturbations
- channel gain-jitter perturbations
- localization error against known target coordinates
- peak-to-sidelobe ratio for peak quality
- runtime tracking per reconstruction method

Outputs are written as JSON/CSV records for reproducibility and downstream analysis.

These benchmarks are intended to produce technical evidence for research and partner evaluation, not clinical claims.
