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


@pytest.mark.parametrize("status,finish_reason", [
    ("output_truncated", "length"),
    ("output_truncated", "stop"),
    ("ok", "length"),
])
def test_report_does_not_credit_truncated_matching_final(tmp_path, monkeypatch, status, finish_reason):
    rows = [{"id": "one", "answer": "Alice"}]
    root = tmp_path / "traces"
    write_json(root / "manifest.json", {"signature": {
        "dataset_sha256": dataset_hash(rows), "configs": configurations()}})
    path = root / "cheap_no_tools" / "one.json"
    write_json(path, dict(id="one", measured=True, config="cheap_no_tools", final="Alice",
                         expected="Alice", correct=True, status=status, cost_usd=.01,
                         steps=1, elapsed_s=.1, events=[
                             {"type": "llm", "finish_reason": finish_reason,
                              "message": {"role": "assistant", "content": "FINAL: Alice"}}]))
    monkeypatch.setattr("freshqa.reporting._chart", lambda *args: None)
    original = path.read_bytes()
    summary, _ = render_report(rows, root, tmp_path / "reports")
    assert summary["cheap_no_tools"]["correct"] == 0
    assert summary["cheap_no_tools"]["complete"]
    assert summary["cheap_no_tools"]["cost_total"] == pytest.approx(.01)
    assert path.read_bytes() == original


@pytest.mark.parametrize("historical_correct,final,status,current_correct", [
    (True, "Alice", "output_truncated", False),
    (False, "Alice", "max_steps", True),
    (False, "unknown", "ok", False),
])
def test_regrading_preserves_original_score_and_is_idempotent(
        tmp_path, monkeypatch, historical_correct, final, status, current_correct):
    import json
    from freshqa.dataset import METRIC_VERSION
    from freshqa.evaluation import GRADING_POLICY_VERSION, HISTORICAL_POLICY

    rows = [{"id": "one", "answer": "Alice"}]
    root, out = tmp_path / "traces", tmp_path / "current"
    write_json(root / "manifest.json", {"signature": {
        "dataset_sha256": dataset_hash(rows), "configs": configurations()}})
    path = root / "cheap_no_tools" / "one.json"
    write_json(path, dict(id="one", measured=True, config="cheap_no_tools", final=final,
                         expected="Alice", correct=historical_correct, status=status,
                         cost_usd=.01, steps=1, elapsed_s=.1, events=[]))
    monkeypatch.setattr("freshqa.reporting._chart", lambda *args: None)
    original = {p.relative_to(root): p.read_bytes() for p in root.rglob("*.json")}
    first = render_report(rows, root, out)
    artifacts = {p.name: p.read_bytes() for p in out.iterdir()}
    assert render_report(rows, root, out) == first
    assert {p.name: p.read_bytes() for p in out.iterdir()} == artifacts
    assert {p.relative_to(root): p.read_bytes() for p in root.rglob("*.json")} == original
    assert first[0]["cheap_no_tools"]["correct"] == int(current_correct)
    sidecar = json.loads((out / "grading.json").read_text())
    assert sidecar["selected_policy"] == sidecar["grading_policy_version"] == GRADING_POLICY_VERSION
    assert sidecar["metric_version"] == METRIC_VERSION
    grade, = sidecar["grades"]
    assert grade["historical_correct"] is historical_correct
    assert grade["current_correct"] is current_correct
    assert grade["historical_grading_policy_version"] is None
    historical, _ = render_report(rows, root, tmp_path / "historical", grading_policy=HISTORICAL_POLICY)
    assert historical["cheap_no_tools"]["correct"] == int(historical_correct)
    explicit = json.loads((tmp_path / "historical/grading.json").read_text())
    assert explicit["selected_policy"] == HISTORICAL_POLICY
    assert explicit["grades"] == sidecar["grades"]
    assert {p.relative_to(root): p.read_bytes() for p in root.rglob("*.json")} == original


@pytest.mark.parametrize("policy", ["freshqa-final-eligibility-v1", "historical-recorded"])
@pytest.mark.parametrize("relative", ["reports", "reports/revisions/2026-11-15",
                                      "reports/revisions/2026-11-15/hw1",
                                      "reports/revisions/2026-11-15-sources"])
def test_regrading_refuses_canonical_reports_before_any_write(monkeypatch, policy, relative):
    from pathlib import Path
    from freshqa import reporting

    canonical = Path(reporting.__file__).resolve().parents[1] / relative

    def forbidden(*args, **kwargs):
        raise AssertionError("Archive protection must run before any write")

    monkeypatch.setattr(Path, "mkdir", forbidden)
    with pytest.raises(ValueError, match="отдельный --out"):
        render_report([], out_dir=canonical, grading_policy=policy)


def test_report_rejects_unknown_grading_policy_before_writes(tmp_path):
    with pytest.raises(ValueError, match="политика"):
        render_report([], out_dir=tmp_path / "not-created", grading_policy="typo")
    assert not (tmp_path / "not-created").exists()
