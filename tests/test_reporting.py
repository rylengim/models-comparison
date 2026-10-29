import pytest

from freshqa.experiment import dataset_hash, configurations, write_json
from freshqa.reporting import freshness, render_report


def test_freshness_requires_every_successful_sonnet_call():
    rows = [dict(correct=False, id="one", status="ok")]
    assert freshness(rows, 1, "anthropic/claude-sonnet-4.6")["passed"] is True
    assert freshness(rows, 2, "anthropic/claude-sonnet-4.6")["passed"] is None
    assert freshness(rows, 1, "some-other-model")["passed"] is None
    rows[0]["status"] = "error"
    assert freshness(rows, 1, "anthropic/claude-sonnet-4.6")["passed"] is None


def test_report_refuses_synthetic_trace_even_with_matching_manifest(tmp_path):
    rows = [{"id": "one", "answer": "Alice"}]
    root = tmp_path / "traces"
    write_json(root / "manifest.json", {"signature": {"dataset_sha256": dataset_hash(rows), "configs": configurations()}})
    write_json(root / "cheap_no_tools" / "one.json", {"id": "one", "measured": False})
    with pytest.raises(ValueError, match="тестовый"):
        render_report(rows, root, tmp_path / "reports")
