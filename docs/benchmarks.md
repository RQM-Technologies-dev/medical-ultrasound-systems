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

## Phase 3: Robustness Benchmark Layer

Phase 3 introduces a lightweight synthetic robustness benchmark layer for
comparing conventional and quaternionic research metrics under controlled
perturbations.

Synthetic perturbations are designed to stress the processing pipeline.
The goal is not to prove clinical superiority. The goal is to compare
conventional and quaternionic benchmark candidates under reproducible synthetic
conditions.

### Metrics

- peak localization error
- peak value
- runtime
- method-by-method summary statistics

### Perturbations

- additive white Gaussian noise (AWGN)
- channel dropout
- channel gain variation
- timing jitter

Phase 3 outputs JSON/CSV/Markdown artifacts for reproducible synthetic
evaluation and requires further validation before any external deployment claims.
