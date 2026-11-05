import json
from pathlib import Path

import pytest

from freshqa.dataset import load_rows
from hw2.prior_runs import ROOT, load_prior_runs


def test_real_hw1_subset_reuses_only_matching_questions_and_completed_inference():
    row = load_rows(ROOT / "hw2" / "data" / "questions.jsonl")[0]
    records = load_prior_runs([row])
    assert [r["config"] for r in records] == ["strong_no_tools", "cheap_live_search"]
    for record in records:
        assert record["question"] == row["question"] and record["origin"] == "hw1"
        original = json.loads((ROOT / record["trace_path"]).read_text(encoding="utf-8"))
        calls = [e for e in original["events"] if e["type"] == "llm"]
        assert record["cost_usd"] == pytest.approx(sum(e["cost_usd"] for e in calls))
        assert record["prompt_tokens"] == sum(e["prompt_tokens"] for e in calls)
        assert record["final_message"] == calls[-1]["message"]
    for change in ({"id": "a-new-id"}, {"question": row["question"] + "?"}, {"answerable": False}):
        assert not load_prior_runs([{**row, **change}])


def test_inconsistent_cost_is_not_imported(tmp_path):
    trace_dir = ROOT / "traces"
    manifest = Path("manifest.json")
    (tmp_path / manifest).write_bytes((trace_dir / manifest).read_bytes())
    row = load_rows(ROOT / "hw2" / "data" / "questions.jsonl")[0]
    trace = Path("strong_no_tools") / (row["id"] + ".json")
    altered = json.loads((trace_dir / trace).read_text(encoding="utf-8"))
    altered["cost_usd"] = 0
    (tmp_path / trace).parent.mkdir()
    (tmp_path / trace).write_text(json.dumps(altered), encoding="utf-8")
    with pytest.raises(ValueError, match="расходов"):
        load_prior_runs([row], trace_dir=tmp_path)
