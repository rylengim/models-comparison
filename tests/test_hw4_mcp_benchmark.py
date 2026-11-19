import json
import subprocess
import sys
from pathlib import Path

import pytest

from hw4.mcp_benchmark_report import evaluate_case, render_report, summarize


SOURCE = "https://example.invalid/publication"
OTHER = "https://example.invalid/other"


def question(identifier, query):
    return {"id": identifier, "source": "fresh", "question": query, "url": SOURCE,
            "answer": "Alpha", "evidence": "The comet is named Alpha."}


def trace(row, *, source=SOURCE, evidence=None, status="ok", tokens=100, cached=50):
    passage = evidence if evidence is not None else row["evidence"]
    return {
        "model": "stub", "provider": "stub", "cost_basis": "stub",
        "status": status, "final": row["evidence"], "source_url": source,
        "model_calls": 0, "cost_usd": 0, "simulated_cost_usd": 0.003,
        "steps": 3, "elapsed_s": 0.02,
        "events": [
            {"type": "tool", "name": "web_search", "arguments": {"query": row["question"]},
             "output": json.dumps({"results": [{"url": source}]})},
            {"type": "tool", "name": "page_find", "arguments": {"url": source},
             "output": json.dumps({"found": True, "url": source, "excerpts": [passage]})},
        ],
        "prompt_cache": {"mode": "simulated", "input_tokens": tokens, "cached_tokens": cached},
    }


@pytest.mark.parametrize(("source", "evidence", "source_matches", "evidence_found"), [
    (SOURCE, None, True, True),
    (OTHER, None, False, False),
    (SOURCE, "A different excerpt.", True, False),
])
def test_benchmark_checks_answer_source_and_read_separately(source, evidence, source_matches, evidence_found):
    row = question("fresh-test", "What is the name of the new comet?")
    case = evaluate_case(row, trace(row, source=source, evidence=evidence), returncode=0, duration_s=0.3)

    assert case["answer_correct"] is True
    assert case["source_matches"] is source_matches
    assert case["evidence_found"] is evidence_found
    assert case["verified_correct"] is (source_matches and evidence_found)


def test_benchmark_counts_failed_questions_and_partial_usage():
    first = question("fresh-first", "What is the name of the new comet?")
    second = question("fresh-second", "What comet name does the report give?")
    third = question("fresh-third", "Which name appears in the report?")
    cases = [
        evaluate_case(first, trace(first), returncode=0, duration_s=0.3),
        evaluate_case(second, trace(second, status="budget_exceeded", tokens=300, cached=0),
                      returncode=2, duration_s=0.4),
        evaluate_case(third, None, returncode=2, duration_s=0.5, diagnostic="startup failed"),
    ]
    summary = summarize(cases)
    totals = summary["totals"]
    assert (totals["questions"], totals["completed"], totals["errors"]) == (3, 1, 2)
    assert totals["answer_correct"] == totals["verified_correct"] == 1
    assert totals["model_calls"] == totals["cost_usd"] == 0
    assert totals["simulated_cost_usd"] == pytest.approx(0.006)
    assert totals["input_tokens_estimate"] == 400
    assert totals["cached_tokens_estimate"] == 50
    assert totals["cached_input_fraction"] == pytest.approx(50 / 400)
    assert "fresh-third" in render_report(summary)


def test_benchmark_rejects_trace_claiming_real_model_calls():
    row = question("fresh-live", "What is the name of the new comet?")
    case = evaluate_case(row, {**trace(row), "model_calls": 1}, returncode=0, duration_s=0.3)

    assert case["status"] != "ok"
    assert case["verified_correct"] is False


def test_runner_uses_ask_with_isolated_traces_and_continues_after_error(monkeypatch, tmp_path):
    from hw4.run_mcp_benchmark import run_benchmark

    rows = [question("fresh-first", "What is the name of the new comet?"),
            question("fresh-second", "What comet name does the report give?")]
    dataset = tmp_path / "questions.jsonl"
    dataset.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    output = tmp_path / "benchmark"
    trace_paths, ledgers = [], []

    def run(argv, **kwargs):
        row = rows[len(trace_paths)]
        assert argv[:5] == [sys.executable, "-m", "freshqa", "ask", "--mcp"]
        assert argv[-2:] == ["--", row["question"]]
        assert row["answer"] not in argv and row["evidence"] not in argv and row["url"] not in argv
        assert argv[argv.index("--budget-usd") + 1] == "0.003"
        assert "OPENROUTER_API_KEY" not in kwargs["env"]
        assert kwargs["env"]["PYTHON_DOTENV_DISABLED"] == "1"
        trace_path = Path(argv[argv.index("--output") + 1])
        ledger_path = Path(argv[argv.index("--ledger") + 1])
        assert trace_path.is_relative_to(output) and ledger_path.is_relative_to(output)
        trace_paths.append(trace_path)
        ledgers.append(ledger_path)
        if len(trace_paths) == 1:
            return subprocess.CompletedProcess(argv, 2, stdout="", stderr="MCP startup failed")
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps(trace(row)), encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", run)
    summary = run_benchmark(dataset, output)

    assert len(set(trace_paths)) == 2 and len(set(ledgers)) == 1
    assert summary["totals"]["completed"] == summary["totals"]["errors"] == 1
    assert json.loads((output / "summary.json").read_text(encoding="utf-8")) == summary
    assert "fresh-first" in (output / "report.md").read_text(encoding="utf-8")
    with pytest.raises(FileExistsError):
        run_benchmark(dataset, output)
    assert len(trace_paths) == 2
