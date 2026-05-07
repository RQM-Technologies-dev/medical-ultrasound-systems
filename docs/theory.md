# Theory notes (research framing)

Quaternionic and QSG-inspired signal models offer a way to represent scalar and directional signal components in a unified algebraic form.

In this repository, the theory is treated as a **research hypothesis** for software methods that may improve wavefield interpretability and coherence diagnostics in ultrasound pipelines. Any observed benefit is a benchmark candidate and requires careful validation.

No clinical efficacy claims are made here. The work is limited to simulation, reconstruction diagnostics, and engineering-first software experiments.

## Quaternionic Ultrasound Wavefield Representation

Ultrasound in this repository is treated as an acoustic/mechanical wave-analysis
problem, not an electromagnetic propagation model. The quaternionic layer is a
software representation of received RF channel data for benchmarking and method
comparison.

In the Phase 2 prototype, quaternionic channel samples combine:

- amplitude from the analytic RF signal
- phase from the analytic RF signal
- an orientation axis derived from array/pixel geometry proxies

This representation is used to study coherence, alignment, and artifact behavior
in synthetic research settings. It is not a clinical interpretation model and is
not intended for diagnosis or patient-care decisions.
