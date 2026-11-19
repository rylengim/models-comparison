import json
import subprocess
import sys
from decimal import Decimal
from itertools import count
from pathlib import Path

import httpx
import pytest

from freshqa.dataset import load_rows
from hw4 import run_mcp_benchmark as benchmark


CLAUDE = "anthropic/claude-sonnet-4.6"
MINI = "openai/gpt-4o-mini"
MOCK_KEY = "synthetic-hw4-comparison-key"
PROFILES = {
    "claude_no_tools": (CLAUDE, False, 1, 0.002),
    "claude_mcp": (CLAUDE, True, 3, 0.002),
    "gpt4o_mini_mcp": (MINI, True, 3, 0.0003),
}


def _arg(argv, name):
    return argv[argv.index(name) + 1]


def _mock_child(argv, row, model, use_mcp, calls, price, *, unknown_cost=False, correct=True):
    cache = {"mode": "provider_reported", "prompt_tokens": 1000 * calls,
             "completion_tokens": 40 * calls, "cached_tokens": 100 * (calls - 1),
             "cache_write_tokens": 0}
    with Path(_arg(argv, "--ledger")).open("a", encoding="utf-8") as ledger:
        for index in range(calls):
            receipt = {"cost_usd": None if unknown_cost and index == calls - 1 else price,
                       "model": model, "prompt_tokens": 1000, "completion_tokens": 40,
                       "cached_tokens": 100 if index else 0, "cache_write_tokens": 0}
            ledger.write(json.dumps(receipt) + "\n")
    events = []
    if use_mcp and not unknown_cost:
        events = [{"type": "tool", "name": "page_find", "arguments": {"url": row["url"]},
                   "output": json.dumps({"found": True, "url": row["url"], "excerpts": [row["evidence"]]})}]
    trace = {
        "model": model, "provider": "openrouter", "cost_basis": "provider_reported",
        "status": "cost_unknown" if unknown_cost else "ok",
        "final": "unknown" if unknown_cost or not correct else row["answer"],
        "source_url": row["url"] if events else None,
        "model_calls": calls, "events": events, "prompt_cache": cache,
        "cost_usd": None if unknown_cost else price * calls,
        "known_cost_usd": price * (calls - int(unknown_cost)),
    }
    Path(_arg(argv, "--output")).write_text(json.dumps(trace), encoding="utf-8")
    return subprocess.CompletedProcess(argv, 2 if unknown_cost else 0, stdout="", stderr="")


def test_comparison_runs_same_30_questions_in_all_profiles(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", MOCK_KEY)
    monkeypatch.setenv("UNRELATED_API_KEY", "must-not-reach-child")
    rows = load_rows(benchmark.QUESTIONS)
    out = tmp_path / "comparison"
    dispatched = {profile: [] for profile in PROFILES}
    monkeypatch.setattr(benchmark, "perf_counter", count(step=0.25).__next__)

    def run(argv, **kwargs):
        ledger = Path(_arg(argv, "--ledger"))
        profile = ledger.parent.name
        model, use_mcp, calls, price = PROFILES[profile]
        row = rows[len(dispatched[profile])]
        assert argv[:5] == [sys.executable, "-m", "freshqa", "ask", "--mcp" if use_mcp else "--no-tools"]
        assert "--openrouter" in argv and _arg(argv, "--model") == model
        assert float(_arg(argv, "--budget-usd")) == 0.06
        assert argv[-2:] == ["--", row["question"]]
        assert all(value not in argv for value in (row["answer"], row["evidence"], row["url"], MOCK_KEY))
        assert ledger.parent == out / profile
        assert Path(_arg(argv, "--output")) == out / profile / "traces" / f"{row['id']}.json"
        assert kwargs["env"]["OPENROUTER_API_KEY"] == MOCK_KEY
        assert kwargs["env"]["PYTHON_DOTENV_DISABLED"] == "1"
        assert "UNRELATED_API_KEY" not in kwargs["env"]
        dispatched[profile].append(row["id"])
        return _mock_child(argv, row, model, use_mcp, calls, price,
                           correct=use_mcp or len(dispatched[profile]) <= 15)

    monkeypatch.setattr(subprocess, "run", run)
    assert benchmark.main(["--compare", "--budget-usd", "0.50", "--out", str(out)]) == 0

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "comparison" and summary["run_cost_usd"] == pytest.approx(0.267)
    assert summary["run_remaining_usd"] == pytest.approx(0.233) and summary["stop_reason"] is None
    assert len(rows) == 30 and all(ids == [row["id"] for row in rows] for ids in dispatched.values())
    configs = summary["configurations"]
    assert [config["id"] for config in configs] == list(PROFILES)
    for config in configs:
        model, use_mcp, calls, price = PROFILES[config["id"]]
        assert config["model"] == model and config["use_mcp"] is use_mcp
        totals = config["totals"]
        assert totals["questions"] == totals["completed"] == 30
        assert totals["answer_correct"] == (30 if use_mcp else 15)
        assert totals["verified_correct"] == (30 if use_mcp else 0)
        assert totals["model_calls"] == 30 * calls
        assert totals["prompt_tokens"] == 30_000 * calls and totals["completion_tokens"] == 1200 * calls
        assert totals["cached_tokens"] == 3000 * (calls - 1)
        assert config["run_cost_usd"] == pytest.approx(30 * calls * price)

    report = (out / "report.md").read_text(encoding="utf-8")
    cells = {
        config["id"]: [cell.strip() for cell in line.strip("|").split("|")][1:]
        for config in configs for line in report.splitlines() if line.startswith(f"| {config['label']} |")
    }
    assert cells == {
        "claude_no_tools": ["15/30 (50.0%)", "—", "30/30", "0", "1.00", "1040.0",
                            "0.20000", "0.40000", "0.250", "0 (0.0%)"],
        "claude_mcp": ["30/30 (100.0%)", "30/30", "30/30", "0", "3.00", "3120.0",
                       "0.60000", "0.60000", "0.250", "6000 (6.7%)"],
        "gpt4o_mini_mcp": ["30/30 (100.0%)", "30/30", "30/30", "0", "3.00", "3120.0",
                           "0.09000", "0.09000", "0.250", "6000 (6.7%)"],
    }
    assert MOCK_KEY not in report and MOCK_KEY not in json.dumps(summary)


def test_comparison_keeps_global_budget_and_stops_all_profiles_on_unknown_charge(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", MOCK_KEY)
    out = tmp_path / "uncertain-comparison"
    rows = load_rows(benchmark.QUESTIONS)
    dispatched = []

    def run(argv, **_kwargs):
        profile = Path(_arg(argv, "--ledger")).parent.name
        requested_budget = Decimal(_arg(argv, "--budget-usd"))
        dispatched.append(profile)
        if profile == "claude_no_tools":
            return _mock_child(argv, rows[len(dispatched) - 1], CLAUDE, False, 1, 0.002)
        assert len(dispatched) == 31 and profile == "claude_mcp"
        assert requested_budget == Decimal("0.040001")
        return _mock_child(argv, rows[0], CLAUDE, True, 3, 0.002, unknown_cost=True)

    monkeypatch.setattr(subprocess, "run", run)
    assert benchmark.main(["--compare", "--budget-usd", "0.100001", "--out", str(out)]) == 2

    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert dispatched == ["claude_no_tools"] * 30 + ["claude_mcp"]
    assert summary["stop_reason"] == "accounting_uncertain"
    assert summary["run_cost_usd"] is None and summary["run_remaining_usd"] is None
    assert summary["run_known_cost_usd"] == pytest.approx(0.064)
    for config, expected_statuses in zip(summary["configurations"], [
        ["ok"] * 30, ["cost_unknown"] + ["accounting_uncertain"] * 29, ["accounting_uncertain"] * 30,
    ], strict=True):
        assert [case["status"] for case in config["cases"]] == expected_statuses


def test_openrouter_no_tools_baseline_uses_one_metered_request(monkeypatch, tmp_path):
    from freshqa import __main__ as cli
    from hw4 import mcp_openrouter

    def refuse_external_setup(*_args, **_kwargs):
        pytest.fail("The no-tools baseline must not initialize MCP or load dotenv")

    monkeypatch.setattr(mcp_openrouter, "with_publication_tools", refuse_external_setup)
    monkeypatch.setattr(cli, "load_dotenv", refuse_external_setup)
    monkeypatch.setenv("OPENROUTER_API_KEY", MOCK_KEY)
    row = load_rows(benchmark.QUESTIONS)[0]
    bodies = []

    def handle(request):
        assert request.method == "POST" and request.headers["Authorization"] == f"Bearer {MOCK_KEY}"
        body = json.loads(request.content)
        bodies.append(body)
        assert body["model"] == CLAUDE
        assert "tools" not in body and "tool_choice" not in body
        assert [message["role"] for message in body["messages"]] == ["system", "user"]
        assert body["messages"][-1]["content"] == row["question"]
        assert all(value.encode() not in request.content for value in (row["answer"], row["url"], row["evidence"]))
        return httpx.Response(200, json={
            "id": "test-receipt",
            "choices": [{"message": {"role": "assistant", "content": f'FINAL: {row["answer"]}'}}],
            "usage": {"cost": 0.003, "prompt_tokens": 325, "completion_tokens": 20,
                      "prompt_tokens_details": {"cached_tokens": 5, "cache_write_tokens": 10}},
        })

    ask = mcp_openrouter.ask_no_tools

    async def mock_ask(question, **kwargs):
        return await ask(question, transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(mcp_openrouter, "ask_no_tools", mock_ask)
    ledger, out = tmp_path / "spend.jsonl", tmp_path / "trace.json"
    assert cli.main(["ask", "--no-tools", "--openrouter", "--model", CLAUDE,
                     "--budget-usd", "0.06", "--ledger", str(ledger), "--output", str(out),
                     "--", row["question"]]) == 0
    trace = json.loads(out.read_text(encoding="utf-8"))
    assert len(bodies) == trace["model_calls"] == 1
    assert trace["status"] == "ok" and trace["final"] == row["answer"] and trace["source_url"] is None
    assert not trace["events"][0]["tools_enabled"]
    assert trace["prompt_cache"]["prompt_tokens"] == 325
    assert trace["prompt_cache"]["cached_tokens"] == 5
    receipt = json.loads(ledger.read_text(encoding="utf-8"))
    assert receipt["cost_usd"] == trace["cost_usd"] == pytest.approx(0.003)
