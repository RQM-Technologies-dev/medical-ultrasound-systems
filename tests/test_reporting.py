import csv
import json

from medical_ultrasound_systems.reporting import write_csv_report, write_json_report


def test_write_json_report_creates_file(tmp_path):
    out_path = tmp_path / "report.json"
    records = [{"method": "das", "score": 1.0}]
    write_json_report(str(out_path), records)
    assert out_path.exists()
    parsed = json.loads(out_path.read_text(encoding="utf-8"))
    assert parsed == records


def test_write_csv_report_creates_file(tmp_path):
    out_path = tmp_path / "report.csv"
    records = [{"method": "das", "score": 1.0}, {"method": "q", "score": 2.0}]
    write_csv_report(str(out_path), records)
    assert out_path.exists()
    with out_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["method"] == "das"
