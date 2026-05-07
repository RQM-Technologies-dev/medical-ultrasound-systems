"""Report serialization helpers for benchmark records."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def write_json_report(path: str, records: list[dict]) -> None:
    """Write records to a JSON file, creating parent directories as needed."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, sort_keys=True)


def write_csv_report(path: str, records: list[dict]) -> None:
    """Write records to a CSV file with stable field ordering."""
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
