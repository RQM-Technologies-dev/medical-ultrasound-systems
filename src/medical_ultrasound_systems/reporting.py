"""Serialization and summary helpers for synthetic benchmark results."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .experiments import ExperimentResult, experiment_result_to_dict, experiment_results_to_rows


def _write_json_records(path: str | Path, records: list[dict]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, sort_keys=True)


def _write_csv_records(path: str | Path, records: list[dict]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(records) == 0:
        out_path.write_text("", encoding="utf-8")
        return

    fieldnames = sorted({key for record in records for key in record.keys()})
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def write_results_json(results: list[ExperimentResult], path: str | Path) -> None:
    """Write experiment results to JSON."""
    records = [experiment_result_to_dict(result) for result in results]
    _write_json_records(path=path, records=records)


def write_results_csv(results: list[ExperimentResult], path: str | Path) -> None:
    """Write flattened experiment rows to CSV."""
    rows = experiment_results_to_rows(results)
    _write_csv_records(path=path, records=rows)


def summarize_results_by_method(results: list[ExperimentResult]) -> dict:
    """Summarize localization error statistics by reconstruction method."""
    method_errors: dict[str, list[float]] = {}
    for result in results:
        for peak in result.peaks:
            method_errors.setdefault(peak.method, []).append(float(peak.localization_error_m))

    summary: dict[str, dict] = {}
    for method, errors in method_errors.items():
        sorted_errors = sorted(errors)
        count = len(sorted_errors)
        if count == 0:
            continue

        midpoint = count // 2
        if count % 2 == 0:
            median_error = 0.5 * (sorted_errors[midpoint - 1] + sorted_errors[midpoint])
        else:
            median_error = sorted_errors[midpoint]

        summary[method] = {
            "count": count,
            "mean_error_m": float(sum(sorted_errors) / count),
            "median_error_m": float(median_error),
            "max_error_m": float(sorted_errors[-1]),
            "min_error_m": float(sorted_errors[0]),
        }
    return summary


def format_summary_markdown(
    summary: dict,
    title: str = "Synthetic Ultrasound Benchmark Summary",
) -> str:
    """Format method-level summary statistics as markdown."""
    lines = [
        f"# {title}",
        "",
        "Synthetic benchmark-candidate summary for research metrics only. "
        "These outputs require validation and are not clinical evidence.",
        "",
    ]
    if not summary:
        lines.append("No results available.")
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "| method | count | mean_error_m | median_error_m | min_error_m | max_error_m |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for method in sorted(summary):
        stats = summary[method]
        lines.append(
            "| "
            f"{method} | {int(stats['count'])} | {float(stats['mean_error_m']):.6e} | "
            f"{float(stats['median_error_m']):.6e} | {float(stats['min_error_m']):.6e} | "
            f"{float(stats['max_error_m']):.6e} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_summary_markdown(
    results: list[ExperimentResult],
    path: str | Path,
    title: str = "Synthetic Ultrasound Benchmark Summary",
) -> None:
    """Write a markdown localization summary grouped by method."""
    summary = summarize_results_by_method(results)
    text = format_summary_markdown(summary=summary, title=title)
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")


def write_json_report(path: str, records: list[dict]) -> None:
    """Backward-compatible JSON writer for generic record lists."""
    _write_json_records(path=path, records=records)


def write_csv_report(path: str, records: list[dict]) -> None:
    """Backward-compatible CSV writer for generic record lists."""
    _write_csv_records(path=path, records=records)
