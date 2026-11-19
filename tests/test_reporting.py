import os
from pathlib import Path

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


@pytest.mark.parametrize("policy", ["freshqa-final-eligibility-v1", "historical-recorded"])
def test_report_symlink_does_not_overwrite_historical_sentinel(tmp_path, monkeypatch, policy):
    sentinel = tmp_path / "historical-report.md"
    original = b"accepted historical report\n"
    sentinel.write_bytes(original)
    out = tmp_path / "current"
    out.mkdir()
    (out / "report.md").symlink_to(sentinel)
    monkeypatch.setattr("freshqa.reporting._chart", lambda *args: None)

    try:
        with pytest.raises(ValueError, match="символическ"):
            render_report([], tmp_path / "no-traces", out, grading_policy=policy)
    finally:
        assert sentinel.read_bytes() == original
    assert {p.name for p in out.iterdir()} == {"report.md"}


@pytest.mark.parametrize("policy", ["freshqa-final-eligibility-v1", "historical-recorded"])
@pytest.mark.parametrize("relative", [
    "report.md", "report.csv", "money_chart.png", "failure_candidates.md",
    "grading.json", "report.json", "freshness.json", "grading.tmp", "report.tmp",
    "freshness.tmp", "nested/unrelated.tmp", "nested/broken.tmp",
])
def test_report_rejects_child_symlinks_before_mkdir(tmp_path, monkeypatch, policy, relative):
    out = tmp_path / "current"
    link = out / relative
    link.parent.mkdir(parents=True)
    target = tmp_path / "historical-artifact"
    original = b"accepted artifact\x00\n"
    if relative != "nested/broken.tmp":
        target.write_bytes(original)
    link.symlink_to(target)

    def forbidden(*args, **kwargs):
        raise AssertionError("Symlink validation must precede mkdir and all writes")

    monkeypatch.setattr(Path, "mkdir", forbidden)
    with pytest.raises(ValueError, match="символическ"):
        render_report([], tmp_path / "no-traces", out, grading_policy=policy)
    if target.exists():
        assert target.read_bytes() == original
    else:
        assert relative == "nested/broken.tmp"
    assert link.is_symlink()


@pytest.mark.parametrize("policy", ["freshqa-final-eligibility-v1", "historical-recorded"])
def test_report_rejects_directory_symlink_without_scanning_target(tmp_path, monkeypatch, policy):
    out = tmp_path / "current"
    (out / "nested").mkdir(parents=True)
    target = tmp_path / "historical-directory"
    target.mkdir()
    sentinel = target / "report.md"
    sentinel.write_bytes(b"accepted report\n")
    link = out / "nested" / "directory-link"
    link.symlink_to(target, target_is_directory=True)
    original_scandir = os.scandir

    def guarded_scandir(path):
        if Path(path) == link or Path(path).is_relative_to(target):
            raise AssertionError("Validation must not traverse directory symlinks")
        return original_scandir(path)

    monkeypatch.setattr(os, "scandir", guarded_scandir)

    def forbidden(*args, **kwargs):
        raise AssertionError("Symlink validation must precede mkdir and all writes")

    monkeypatch.setattr(Path, "mkdir", forbidden)
    with pytest.raises(ValueError, match="символическ"):
        render_report([], tmp_path / "no-traces", out, grading_policy=policy)
    assert sentinel.read_bytes() == b"accepted report\n"


@pytest.mark.parametrize("policy", ["freshqa-final-eligibility-v1", "historical-recorded"])
@pytest.mark.parametrize("location", [
    "output", "ancestor", "ancestor-before-dotdot", "broken-output", "broken-ancestor",
])
def test_report_rejects_symlink_in_output_path_before_resolve(
        tmp_path, monkeypatch, policy, location):
    target = tmp_path / "target"
    if not location.startswith("broken"):
        target.mkdir()
    link = tmp_path / "alias"
    link.symlink_to(target, target_is_directory=True)
    if location.endswith("output"):
        out = link
    elif location == "ancestor-before-dotdot":
        out = link / ".." / "new-output"
    else:
        out = link / "not-created" / "output"

    def forbidden(*args, **kwargs):
        raise AssertionError("Output links must be rejected before resolve, mkdir or writes")

    monkeypatch.setattr(Path, "resolve", forbidden)
    monkeypatch.setattr(Path, "mkdir", forbidden)
    with pytest.raises(ValueError, match="символическ"):
        render_report([], tmp_path / "no-traces", out, grading_policy=policy)
    assert not (target / "not-created").exists()
    assert not (tmp_path / "new-output").exists()


@pytest.mark.parametrize("policy", ["freshqa-final-eligibility-v1", "historical-recorded"])
def test_report_allows_clean_nested_output_and_preserves_unrelated_files(tmp_path, monkeypatch, policy):
    out = tmp_path / "current"
    nested = out / "nested"
    nested.mkdir(parents=True)
    sentinel = nested / "notes.txt"
    sentinel.write_bytes(b"local notes\n")
    monkeypatch.setattr("freshqa.reporting._chart", lambda *args: None)
    first = render_report([], tmp_path / "no-traces", out, grading_policy=policy)
    artifacts = {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()}
    assert render_report([], tmp_path / "no-traces", out, grading_policy=policy) == first
    assert {p.relative_to(out): p.read_bytes() for p in out.rglob("*") if p.is_file()} == artifacts
    assert sentinel.read_bytes() == b"local notes\n"
