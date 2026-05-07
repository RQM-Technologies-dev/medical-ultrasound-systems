import csv
import json

from medical_ultrasound_systems.experiments import ExperimentResult, PeakResult
from medical_ultrasound_systems.reporting import (
    format_summary_markdown,
    summarize_results_by_method,
    write_results_csv,
    write_results_json,
    write_summary_markdown,
)


def _make_results() -> list[ExperimentResult]:
    return [
        ExperimentResult(
            name="noise_sweep",
            parameters={"snr_db": 20.0, "trial_index": 0},
            peaks=[
                PeakResult(
                    method="delay_and_sum_plane_wave",
                    peak_x_m=0.0,
                    peak_z_m=0.03,
                    target_x_m=0.0,
                    target_z_m=0.03,
                    localization_error_m=0.001,
                    peak_value=1.0,
                ),
                PeakResult(
                    method="quaternionic_alignment_image",
                    peak_x_m=0.0,
                    peak_z_m=0.03,
                    target_x_m=0.0,
                    target_z_m=0.03,
                    localization_error_m=0.0005,
                    peak_value=0.8,
                ),
            ],
            runtime_s={"delay_and_sum_plane_wave": 0.01, "quaternionic_alignment_image": 0.02},
        )
    ]


def test_summarize_results_by_method_small_case():
    summary = summarize_results_by_method(_make_results())
    assert summary["delay_and_sum_plane_wave"]["count"] == 1
    assert summary["quaternionic_alignment_image"]["mean_error_m"] == 0.0005


def test_format_summary_markdown_returns_markdown_string():
    summary = summarize_results_by_method(_make_results())
    markdown = format_summary_markdown(summary, title="Phase 3 Summary")
    assert markdown.startswith("# Phase 3 Summary")
    assert "| method | count |" in markdown


def test_write_results_json_csv_and_markdown(tmp_path):
    results = _make_results()
    json_path = tmp_path / "results.json"
    csv_path = tmp_path / "results.csv"
    markdown_path = tmp_path / "summary.md"

    write_results_json(results, json_path)
    write_results_csv(results, csv_path)
    write_summary_markdown(results, markdown_path)

    assert json_path.exists()
    assert csv_path.exists()
    assert markdown_path.exists()

    parsed_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert isinstance(parsed_json, list)
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert "Synthetic Ultrasound Benchmark Summary" in markdown_path.read_text(encoding="utf-8")
