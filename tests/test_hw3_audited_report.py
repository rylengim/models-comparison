import copy
import json

import pytest

from hw3.audited_report import apply_overrides, build_report
from hw3.report import render_report


def gold():
    return [{"id": "one", "question": "Who won?", "answer": "Alice",
             "answer_aliases": ["Alicia"], "evidence": "Alice won.",
             "url": "https://example.com/old", "metadata": {"tags": ["frozen"]}}]


def patches():
    return [{"id": "one", "old_answer": "Alice", "answer": "Bob",
             "answer_aliases": ["Bobby"], "evidence": "Bob won.",
             "source_url": "https://example.com/official", "question": "Do not replace this"}]


def test_override_copies_gold_and_replaces_only_audited_fields():
    rows, overrides = gold(), patches()
    before_rows, before_patches = copy.deepcopy(rows), copy.deepcopy(overrides)
    corrected = apply_overrides(rows, overrides)
    assert corrected[0]["answer"] == "Bob"
    assert corrected[0]["answer_aliases"] == ["Bobby"]
    assert corrected[0]["evidence"] == "Bob won."
    assert corrected[0]["url"] == "https://example.com/official"
    assert corrected[0]["question"] == rows[0]["question"]
    assert corrected[0]["metadata"] == rows[0]["metadata"]
    corrected[0]["metadata"]["tags"].append("derived")
    corrected[0]["answer_aliases"].append("Robert")
    assert rows == before_rows
    assert overrides == before_patches


@pytest.mark.parametrize("change,match", [
    ({"id": "missing"}, "Unknown"),
    ({"old_answer": "Wrong original"}, "old_answer"),
    ({"answer_aliases": "Bobby"}, "aliases"),
])
def test_override_rejects_wrong_identity_gold_or_alias_type(change, match):
    override = patches()[0]
    override.update(change)
    rows = gold()
    before = copy.deepcopy(rows)
    with pytest.raises(ValueError, match=match):
        apply_overrides(rows, [override])
    assert rows == before


def test_override_rejects_duplicate_patches_and_duplicate_gold_ids():
    with pytest.raises(ValueError, match="Duplicate"):
        apply_overrides(gold(), patches() * 2)
    with pytest.raises(ValueError, match="unique"):
        apply_overrides(gold() * 2, patches())


def write_trace(root, config, final, initial=None):
    record = {"id": "one", "question": "Who won?", "config": config, "final": final,
              "status": "ok", "measured": True, "steps": 1, "prompt_tokens": 10,
              "completion_tokens": 2, "cost_usd": .01,
              "call_records": [{"question_id": "one", "config": config, "cost_usd": .01,
                                "prompt_tokens": 10, "completion_tokens": 2,
                                "request_id": config + "-one", "status_code": 200}]}
    if initial is not None:
        record["initial_answer"] = initial
    path = root / config / "one.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    return path


def test_offline_report_regrades_both_critic_answers_and_preserves_raw_files(tmp_path):
    rows = gold()
    critic_path = write_trace(tmp_path, "critic", "Bob", initial="Alice")
    bare_path = write_trace(tmp_path, "bare", "Alice")
    render_report(tmp_path, rows, tmp_path)
    (tmp_path / "grading_overrides.json").write_text(json.dumps(patches()))
    (tmp_path / "manual_audit.md").write_text("Full review: [trace](bare/one.json).")
    (tmp_path / "manual_reviews.json").write_text(json.dumps([
        {"config": "bare", "id": "one", "judgement": "incorrect", "note": "Uses the old gold."}
    ]))
    preserved = {p: p.read_bytes() for p in [critic_path, bare_path, tmp_path / "report.md",
                 tmp_path / "summary.json", tmp_path / "manual_audit.md",
                 tmp_path / "grading_overrides.json"]}
    before = copy.deepcopy(rows)
    summary = build_report(tmp_path, rows)
    critic = next(r for r in summary if r["config"] == "critic")
    original = json.loads(preserved[tmp_path / "summary.json"])
    raw_critic = next(r for r in original if r["config"] == "critic")
    assert (raw_critic["correct"], raw_critic["fixed"], raw_critic["broken"]) == (0, 0, 1)
    assert (critic["correct"], critic["fixed"], critic["broken"]) == (1, 1, 0)
    assert critic["cost_usd"] == raw_critic["cost_usd"] == .01
    audited = tmp_path / "audited"
    candidates = json.loads((audited / "audit_candidates.json").read_text())
    bare = next(r for r in candidates if r["config"] == "bare")
    assert bare["expected"] == "Bob"
    assert bare["audit_status"] == "incorrect"
    assert bare["audit_note"] == "Uses the old gold."
    assert "../manual_audit.md" in (audited / "manual_audit.md").read_text()
    report = (audited / "report.md").read_text()
    assert "https://example.com/official" in report
    assert "../report.md" in report
    assert "без новых" in report
    for path, content in preserved.items():
        assert path.read_bytes() == content
    assert rows == before


def test_failed_override_leaves_no_derived_report(tmp_path):
    override = patches()[0]
    override["old_answer"] = "A different snapshot"
    (tmp_path / "grading_overrides.json").write_text(json.dumps([override]))
    with pytest.raises(ValueError, match="old_answer"):
        build_report(tmp_path, gold())
    assert not (tmp_path / "audited").exists()


def test_comparison_recomputes_original_totals_instead_of_trusting_stale_summary(tmp_path):
    write_trace(tmp_path, "critic", "Bob", initial="Alice")
    (tmp_path / "grading_overrides.json").write_text(json.dumps(patches()))
    raw_summary = tmp_path / "summary.json"
    raw_summary.write_text('[{"config":"critic","correct":999}]')
    before = raw_summary.read_bytes()
    build_report(tmp_path, gold())
    comparison = json.loads((tmp_path / "audited/grading_summary.json").read_text())
    original = next(r for r in comparison["original"] if r["config"] == "critic")
    assert (original["correct"], original["fixed"], original["broken"]) == (0, 0, 1)
    assert raw_summary.read_bytes() == before
