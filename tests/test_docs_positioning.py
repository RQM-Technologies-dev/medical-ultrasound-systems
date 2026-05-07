from pathlib import Path


def test_readme_and_docs_preserve_non_clinical_positioning():
    root = Path(__file__).resolve().parents[1]
    readme = (root / "README.md").read_text(encoding="utf-8").lower()
    docs_text = "\n".join(
        path.read_text(encoding="utf-8").lower()
        for path in sorted((root / "docs").glob("*.md"))
    )
    corpus = f"{readme}\n{docs_text}"

    assert "not a medical device" in readme
    assert "not clinical software" in readme
    assert "not fda-cleared" in readme
    assert "not for diagnosis" in readme
    assert "not for clinical use" in corpus

    prohibited_claims = [
        "diagnostic improvement",
        "clinical benefit",
        "fda clearance",
        "patient outcome improvement",
        "device readiness",
        "improves diagnosis",
        "improves treatment",
    ]
    for phrase in prohibited_claims:
        assert phrase not in corpus
