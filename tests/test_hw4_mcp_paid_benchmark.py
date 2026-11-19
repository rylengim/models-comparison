import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest

from freshqa.dataset import load_rows
from hw4 import run_mcp_benchmark as benchmark
from hw4.mcp_benchmark_report import evaluate_case


MODEL = "anthropic/claude-sonnet-4.6"
MOCK_KEY = "synthetic-benchmark-key"


def child_arg(argv, name):
    return argv[argv.index(name) + 1]


def check_child(argv, env, row, output):
    assert argv[:5] == [sys.executable, "-m", "freshqa", "ask", "--mcp"]
    assert "--openrouter" in argv
    assert child_arg(argv, "--model") == MODEL
    assert argv[-2:] == ["--", row["question"]]
    assert all(value not in argv for value in (row["answer"], row["url"], row["evidence"], MOCK_KEY))
    assert env["OPENROUTER_API_KEY"] == MOCK_KEY
    assert env["PYTHON_DOTENV_DISABLED"] == "1"
    assert "UNRELATED_API_KEY" not in env
    ledger = Path(child_arg(argv, "--ledger"))
    assert ledger.parent == output and ledger.suffix == ".jsonl"
    trace = Path(child_arg(argv, "--output"))
    assert trace == output / "traces" / f"{row['id']}.json"
    return ledger, trace


def write_mock_case(argv, row, ledger, trace, costs, *, status="ok"):
    steps = []
    for number, cost in enumerate(costs, 1):
        usage = {
            "step": number, "status_code": 200, "request_id": f"mock-{row['id']}-{number}",
            "cost_usd": float(cost) if cost is not None else None,
            "prompt_tokens": 1200, "completion_tokens": 40,
            "cached_tokens": 100 if number > 1 else 0, "cache_write_tokens": 0,
        }
        with ledger.open("a", encoding="utf-8") as stream:
            receipt = {**usage, "config": "hw4_mcp_openrouter", "question_id": row["id"],
                       "model": MODEL, "timestamp": "2026-11-19T09:00:00+00:00"}
            stream.write(json.dumps(receipt) + "\n")
        steps.append(usage)

    known_cost = float(sum(cost for cost in costs if cost is not None))
    events = [
        {"type": "tool", "name": "web_search", "arguments": {"query": row["question"]},
         "output": json.dumps({"results": [{"url": row["url"]}]})},
        {"type": "tool", "name": "page_find", "arguments": {"url": row["url"]},
         "output": json.dumps({"found": True, "url": row["url"], "excerpts": [row["evidence"]]})},
    ] if status == "ok" else []
    result = {
        "model": MODEL, "provider": "openrouter", "cost_basis": "provider_reported",
        "status": status, "final": f'{row["answer"]} — "{row["evidence"]}"' if status == "ok" else "unknown",
        "source_url": row["url"] if status == "ok" else None,
        "model_calls": len(steps), "steps": len(steps), "question_budget_usd": float(child_arg(argv, "--budget-usd")),
        "cost_usd": None if None in costs else known_cost, "known_cost_usd": known_cost, "events": events,
        "prompt_cache": {"mode": "provider_reported", "steps": steps, **{
            key: sum(step[key] for step in steps)
            for key in ("prompt_tokens", "completion_tokens", "cached_tokens", "cache_write_tokens")
        }},
    }
    trace.write_text(json.dumps(result), encoding="utf-8")
    return subprocess.CompletedProcess(argv, 0 if status == "ok" else 2, stdout="", stderr="")


@pytest.fixture
def mock_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", MOCK_KEY)
    monkeypatch.setenv("UNRELATED_API_KEY", "another-synthetic-key")


def test_paid_cli_runs_all_30_questions_and_writes_report(mock_key, monkeypatch, tmp_path):
    output = tmp_path / "benchmark"
    rows = load_rows(benchmark.QUESTIONS)
    calls = []
    costs = [Decimal("0.004413"), Decimal("0.006114"), Decimal("0.004710")]

    def run(argv, **kwargs):
        row = rows[len(calls)]
        ledger, trace = check_child(argv, kwargs["env"], row, output)
        assert Decimal(child_arg(argv, "--budget-usd")) == Decimal("0.06")
        calls.append(ledger)
        return write_mock_case(argv, row, ledger, trace, costs)

    monkeypatch.setattr(subprocess, "run", run)
    assert benchmark.main(["--openrouter", "--budget-usd", "1", "--out", str(output)]) == 0

    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    totals = summary["totals"]
    assert summary["mode"] == "openrouter" and summary["model"] == MODEL
    assert summary["question_budget_usd"] == 0.06 and summary["run_budget_usd"] == 1
    assert summary["stop_reason"] is None
    assert len(calls) == totals["questions"] == totals["completed"] == totals["verified_correct"] == 30
    assert len(set(calls)) == 1 and totals["model_calls"] == 90
    assert totals["prompt_tokens"] == 108000 and totals["completion_tokens"] == 3600
    assert totals["cached_tokens"] == 6000 and totals["cached_input_fraction"] == pytest.approx(6000 / 108000)
    assert summary["run_cost_usd"] == summary["run_known_cost_usd"] == pytest.approx(0.45711)
    assert summary["run_remaining_usd"] == pytest.approx(0.54289)
    report = (output / "report.md").read_text(encoding="utf-8")
    assert MODEL in report and "fresh-152" in report and "30/30" in report
    assert MOCK_KEY not in report and MOCK_KEY not in json.dumps(summary)


def test_paid_runner_floors_remaining_budget_and_stops_on_run_limit(mock_key, monkeypatch, tmp_path):
    output = tmp_path / "budgeted"
    rows = load_rows(benchmark.QUESTIONS)
    budgets = []

    def run(argv, **kwargs):
        row = rows[len(budgets)]
        ledger, trace = check_child(argv, kwargs["env"], row, output)
        budgets.append(Decimal(child_arg(argv, "--budget-usd")))
        if len(budgets) == 1:
            costs = [Decimal("0.004413"), Decimal("0.006114"), Decimal("0.00471025")]
            return write_mock_case(argv, row, ledger, trace, costs)
        assert len(budgets) == 2, "Runner dispatched another question after its run budget stop"
        return write_mock_case(argv, row, ledger, trace, [Decimal("0.004413")], status="budget_exceeded")

    monkeypatch.setattr(subprocess, "run", run)
    summary = benchmark.run_benchmark(benchmark.QUESTIONS, output, openrouter=True, budget_usd="0.061")

    assert budgets == [Decimal("0.06"), Decimal("0.045762")]
    assert summary["stop_reason"] == "run_budget_exceeded"
    assert summary["run_cost_usd"] == summary["run_known_cost_usd"] == pytest.approx(0.01965025)
    assert summary["run_remaining_usd"] == pytest.approx(0.04134975)
    cases = summary["cases"]
    assert [case["id"] for case in cases] == [row["id"] for row in rows]
    assert [case["status"] for case in cases] == ["ok", "budget_exceeded"] + ["run_budget_exceeded"] * 28
    assert summary["totals"]["questions"] == 30 and summary["totals"]["model_calls"] == 4
    assert len(list((output / "traces").glob("*.json"))) == 2


def test_question_budget_stop_leaves_remaining_questions_eligible(mock_key, monkeypatch, tmp_path):
    output = tmp_path / "individual-stop"
    rows = load_rows(benchmark.QUESTIONS)
    budgets = []

    def run(argv, **kwargs):
        row = rows[len(budgets)]
        ledger, trace = check_child(argv, kwargs["env"], row, output)
        budgets.append(Decimal(child_arg(argv, "--budget-usd")))
        status = "budget_exceeded" if len(budgets) == 1 else "ok"
        return write_mock_case(argv, row, ledger, trace, [Decimal("0.004413")], status=status)

    monkeypatch.setattr(subprocess, "run", run)
    summary = benchmark.run_benchmark(benchmark.QUESTIONS, output, openrouter=True, budget_usd="1")

    assert budgets == [Decimal("0.06")] * 30
    assert summary["stop_reason"] is None
    assert summary["cases"][0]["status"] == "budget_exceeded"
    assert summary["totals"]["completed"] == 29
    assert summary["run_cost_usd"] == pytest.approx(0.13239)


@pytest.mark.parametrize("uncertainty", ("inflight", "unknown_charge"))
def test_paid_runner_stops_on_unresolved_charge(mock_key, monkeypatch, tmp_path, uncertainty):
    rows = load_rows(benchmark.QUESTIONS)
    output = tmp_path / "uncertain"
    calls = []

    def run(argv, **kwargs):
        assert not calls, "Runner dispatched another question while billing is uncertain"
        row = rows[0]
        ledger, trace = check_child(argv, kwargs["env"], row, output)
        calls.append(ledger)
        if uncertainty == "inflight":
            write_mock_case(argv, row, ledger, trace, [Decimal("0.004413")], status="cost_unknown")
            trace.unlink()  # The child stopped before it could publish a final trace.
            ledger.with_suffix(".inflight.json").write_text("{}", encoding="utf-8")
            raise subprocess.TimeoutExpired(argv, 90)
        return write_mock_case(argv, row, ledger, trace, [Decimal("0.004413"), None], status="cost_unknown")

    monkeypatch.setattr(subprocess, "run", run)
    summary = benchmark.run_benchmark(benchmark.QUESTIONS, output, openrouter=True, budget_usd="1")

    assert len(calls) == 1 and summary["stop_reason"] == "accounting_uncertain"
    assert summary["run_cost_usd"] is None and summary["run_remaining_usd"] is None
    assert summary["run_known_cost_usd"] == pytest.approx(0.004413)
    assert len(summary["cases"]) == 30
    assert all(case["status"] == "accounting_uncertain" for case in summary["cases"][1:])
    assert json.loads((output / "summary.json").read_text(encoding="utf-8")) == summary


@pytest.mark.parametrize("ledger_cost", (None, 0.001))
def test_paid_runner_stops_when_ledger_misses_a_trace_charge(mock_key, monkeypatch, tmp_path, ledger_cost):
    first = load_rows(benchmark.QUESTIONS)[0]
    output = tmp_path / "mismatch"
    calls = []

    def run(argv, **kwargs):
        assert not calls, "Runner dispatched another question despite a missing receipt amount"
        ledger, trace = check_child(argv, kwargs["env"], first, output)
        calls.append(ledger)
        completed = write_mock_case(argv, first, ledger, trace, [Decimal("0.004413")])
        if ledger_cost is None:
            ledger.unlink()
        else:
            receipt = json.loads(ledger.read_text(encoding="utf-8"))
            receipt["cost_usd"] = ledger_cost
            ledger.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
        return completed

    monkeypatch.setattr(subprocess, "run", run)
    summary = benchmark.run_benchmark(benchmark.QUESTIONS, output, openrouter=True, budget_usd="1")

    assert len(calls) == 1 and summary["stop_reason"] == "accounting_uncertain"
    assert summary["run_cost_usd"] is None and summary["run_remaining_usd"] is None
    assert all(case["status"] == "accounting_uncertain" for case in summary["cases"][1:])


@pytest.mark.parametrize(("key", "budget"), ((None, "1"), (MOCK_KEY, None), (MOCK_KEY, "-1"), (MOCK_KEY, "NaN")))
def test_paid_runner_rejects_missing_key_or_bad_budget_before_output(monkeypatch, tmp_path, key, budget):
    if key is None:
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENROUTER_API_KEY", key)

    def run(*args, **kwargs):
        raise AssertionError("Invalid configuration must not start ask")

    monkeypatch.setattr(subprocess, "run", run)
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        benchmark.run_benchmark(benchmark.QUESTIONS, output, openrouter=True, budget_usd=budget)
    assert not output.exists()


def test_zero_run_budget_produces_report_without_key_or_attempts(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def run(*args, **kwargs):
        raise AssertionError("Zero run budget must not start ask")

    monkeypatch.setattr(subprocess, "run", run)
    output = tmp_path / "zero"
    summary = benchmark.run_benchmark(benchmark.QUESTIONS, output, openrouter=True, budget_usd="0")

    assert summary["stop_reason"] == "run_budget_exceeded"
    assert summary["run_cost_usd"] == summary["run_remaining_usd"] == 0
    assert [case["status"] for case in summary["cases"]] == ["run_budget_exceeded"] * 30
    assert (output / "report.md").is_file()


def test_paid_evaluator_does_not_credit_wrong_answer_with_correct_quote(tmp_path):
    row = next(row for row in load_rows(benchmark.QUESTIONS) if row["id"] == "fresh-152")
    trace_path = tmp_path / "trace.json"
    write_mock_case(["--budget-usd", "0.06"], row, tmp_path / "spend.jsonl", trace_path, [Decimal("0.004413")])
    result = json.loads(trace_path.read_text(encoding="utf-8"))
    result["final"] = f'The designation is 1P/Halley — "{row["evidence"]}"'

    case = evaluate_case(row, result, model=MODEL, returncode=0, duration_s=0.1)

    assert case["status"] == "ok" and case["source_matches"] and case["evidence_found"]
    assert not case["answer_correct"] and not case["verified_correct"]


@pytest.mark.parametrize(("question_id", "final", "correct"), [
    pytest.param(
        "fresh-126",
        'The acronym is **RELHIC**, standing for **"Reionization-Limited H I Cloud"** '
        '— the announcement states: *"this is the first confirmed detection… '
        'Reionization-Limited H I Cloud, or \'RELHIC.\'"*',
        True, id="fresh-126-or-in-appended-quote",
    ),
    pytest.param(
        "fresh-154",
        "The hidden compact companion (a neutron star or stellar-mass black hole) would pull material "
        'from AT 2016blu **every 113 days** — *"is pulling material from AT 2016blu every 113 days '
        'as they orbit one another."*',
        True, id="fresh-154-or-in-parenthetical-explanation",
    ),
    pytest.param("fresh-126", "The acronym is **RELHIC** or AGB.", False, id="unresolved-answer"),
    pytest.param("fresh-126", "The acronym is RELHIC (or AGB).", False, id="parenthesized-answer-alternative"),
    pytest.param("fresh-126", 'The acronym is **"RELHIC"**.', True, id="quoted-answer"),
    pytest.param("fresh-126", 'The acronym — **"RELHIC"**.', True, id="quoted-answer-after-label"),
    pytest.param("fresh-126", 'The acronym is "RELHIC" or "AGB".', False, id="quoted-alternatives"),
    pytest.param(
        "fresh-126", 'The acronym is **AGB** — the announcement states: *"Reionization-Limited H I Cloud, '
        'or \'RELHIC.\'"*', False, id="wrong-answer-with-formatted-quote",
    ),
])
def test_paid_evaluator_distinguishes_alternatives_from_explanations(tmp_path, question_id, final, correct):
    row = next(row for row in load_rows(benchmark.QUESTIONS) if row["id"] == question_id)
    trace_path = tmp_path / "trace.json"
    write_mock_case(["--budget-usd", "0.06"], row, tmp_path / "spend.jsonl", trace_path, [Decimal("0.004413")])
    result = json.loads(trace_path.read_text(encoding="utf-8"))
    result["final"] = final

    case = evaluate_case(row, result, model=MODEL, returncode=0, duration_s=0.1)

    assert case["status"] == "ok" and case["source_matches"] and case["evidence_found"]
    assert case["answer_correct"] is correct
    assert case["verified_correct"] is correct
