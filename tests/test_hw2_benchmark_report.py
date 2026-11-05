"""Benchmark scoring must distinguish explicit refusals, fallback text and unknown charges."""
import json

import pytest

from hw2.benchmark_report import LABELS, classify, failure_examples, render_report, summarize


YES = {"id": "yes", "question": "Where?", "answer": "United Kingdom", "answer_aliases": ["UK"],
       "url": "wiki/uk", "evidence": "The UK is the source."}
NO = {"id": "no", "question": "Which unknown detail?", "answerable": False, "answer": "unknown"}


def record(row, final, *, status="ok", config="strong_no_tools", **fields):
    return {"id": row["id"], "question": row["question"], "config": config, "model": "m", "final": final,
            "origin": "hw2", "measured": True, "cost_usd": 0.02, "cost_basis": "provider_reported",
            "status": status, "final_message": {"role": "assistant", "content": "FINAL: " + final},
            "finish_reason": "stop", **fields}


def test_partial_measured_rows_aliases_denominators_and_dollars(tmp_path):
    records = [record(YES, "UK", origin="hw1"), record(NO, "unknown\n\nDetail missing.", status="max_steps")]
    report = render_report([YES, NO], records, tmp_path)
    values = report["configs"]["strong_no_tools"]
    assert values["complete"] and values["accuracy"] == 1
    assert values["cost_total"] == .04 and values["cost_per_correct"] == .02
    assert values["correct_answer"] == values["correct_refusal"] == 1
    assert values["known_cost_by_origin"] == {"hw1": .02, "hw2": .02}
    assert report["configs"]["cheap_rag"]["accuracy"] is None
    assert report["failures"]["best_config"] is None
    assert report == json.loads((tmp_path / "report.json").read_text())
    assert (tmp_path / "money_chart.png").read_bytes().startswith(b"\x89PNG")

    partial = summarize([YES, NO], records[:1])["configs"]["strong_no_tools"]
    assert partial["n"] == 1 and partial["target"] == 2 and not partial["complete"]


def test_invalid_output_no_hits_and_unknown_charges():
    invalid = record(NO, "unknown", status="invalid_output")
    missing = record(NO, "unknown", status="no_hits", cost_usd=0)
    assert classify(NO, invalid) == "invalid_output"
    assert classify(NO, missing) == "no_hits"
    assert classify(NO, record(NO, "unknown", status="max_steps", final_message={})) == "invalid_output"
    assert classify(NO, record(NO, "unknown", status="output_truncated")) == "invalid_output"
    assert classify(YES, record(YES, "unknown\n\nI lack this information.")) == "false_refusal"
    v = summarize([YES, NO], [invalid])["configs"]["strong_no_tools"]
    assert v["correct"] == v["correct_refusal"] == 0 and v["cost_per_correct"] is None
    v = summarize([YES, NO], [missing])["configs"]["strong_no_tools"]
    assert v["correct"] == v["no_hits_negative"] == 1 and v["correct_refusal"] == 0
    failed = record(YES, "unknown", status="cost_unknown", cost_usd=None, known_cost_usd=.01)
    v = summarize([YES, NO], [failed, invalid])["configs"]["strong_no_tools"]
    assert v["execution_error"] == v["unknown_charges"] == 1 and not v["complete"]
    assert v["cost_total"] is None and v["known_cost_by_origin"]["hw2"] == .03
    assert v["cost_per_question"] is None and v["cost_per_correct"] is None


def test_reject_duplicate_and_synthetic_measurements():
    row = record(YES, "UK")
    with pytest.raises(ValueError):
        summarize([YES], [row, row])
    with pytest.raises(ValueError):
        summarize([YES], [{**row, "measured": False}])
    v = summarize([YES], [{**row, "cost_basis": "estimated"}])["configs"]["strong_no_tools"]
    assert v["cost_total"] is None


def test_failures_use_actual_returned_chunks_on_complete_comparison():
    missing_quote = {**YES, "id": "b", "question": "Separate question?"}
    successful = {**YES, "id": "c", "question": "Other question?"}
    rows = [YES, missing_quote, successful]
    traces = []
    for name in LABELS:
        for row in rows:
            answer = "UK" if name == "cheap_rag" and row is successful else "wrong"
            trace = record(row, answer, config=name, status="answered" if name == "cheap_rag" else "ok")
            trace["retrieved_ids"] = ["gold"] if row is YES else []
            traces.append(trace)
    report = summarize(rows, traces)
    failures = failure_examples(report, rows, traces, [{"id": "gold", "text": YES["evidence"], "url": YES["url"]}])
    assert failures["best_config"] == "cheap_rag"
    assert failures["counts"] == {"answer_failed": 1, "evidence_missing": 1}
    assert failures["examples"]["answer_failed"]["context"]["id"] == "gold"
