"""Phase 3 synthetic robustness benchmark (research-only, non-clinical).

This script compares benchmark-candidate signal-processing methods under
controlled synthetic perturbations. It is not a clinical validation workflow.
"""

from __future__ import annotations

from pathlib import Path

from medical_ultrasound_systems.experiments import dropout_sweep, gain_jitter_sweep, noise_sweep
from medical_ultrasound_systems.reporting import (
    summarize_results_by_method,
    write_results_csv,
    write_results_json,
    write_summary_markdown,
)


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_trials = 2
    seed = 2026

    noise_results = noise_sweep([40, 30, 20, 10], n_trials=n_trials, seed=seed + 1)
    dropout_results = dropout_sweep([0.0, 0.1, 0.25, 0.5], n_trials=n_trials, seed=seed + 2)
    gain_jitter_results = gain_jitter_sweep(
        gain_std_values=[0.0, 0.1, 0.25],
        max_jitter_samples_values=[0, 1, 2],
        n_trials=n_trials,
        seed=seed + 3,
    )

    all_results = noise_results + dropout_results + gain_jitter_results
    json_path = output_dir / "phase3_results.json"
    csv_path = output_dir / "phase3_results.csv"
    summary_path = output_dir / "phase3_summary.md"

    write_results_json(all_results, json_path)
    write_results_csv(all_results, csv_path)
    write_summary_markdown(
        all_results,
        summary_path,
        title="Phase 3 Synthetic Robustness Benchmark Summary",
    )

    summary = summarize_results_by_method(all_results)
    print("Phase 3 robustness benchmark completed (synthetic research only).")
    print("These benchmark-candidate metrics are non-clinical and require validation.")
    print(f"Total experiments: {len(all_results)}")
    print(f"Methods summarized: {len(summary)}")
    for method, stats in sorted(summary.items()):
        print(
            f"  - {method}: n={stats['count']}, "
            f"mean_error_m={stats['mean_error_m']:.6e}, "
            f"max_error_m={stats['max_error_m']:.6e}"
        )
    print(f"JSON: {json_path}")
    print(f"CSV: {csv_path}")
    print(f"Markdown summary: {summary_path}")


if __name__ == "__main__":
    main()
