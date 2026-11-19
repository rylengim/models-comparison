"""Run the HW4 question set through the FreshQA CLI."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from math import isclose
from pathlib import Path
from time import perf_counter
from zoneinfo import ZoneInfo

from freshqa.dataset import load_rows, validate_rows
from freshqa.experiment import dataset_hash, write_json

from .mcp_benchmark_report import evaluate_case, render_comparison, render_report, summarize
from .mcp_openrouter import DEFAULT_MODEL
from .spending import MOSCOW, limit_micros


REPO = Path(__file__).resolve().parents[1]
QUESTIONS = REPO / "hw4/data/questions.jsonl"
QUESTION_BUDGET = "0.003"
CLI_TIMEOUT = 90
OPENROUTER_QUESTION_BUDGET = Decimal("0.06")
OPENROUTER_CLI_TIMEOUT = 900
OPENROUTER_LEDGER_NAME = "spend.jsonl"
MICRO_DOLLAR = Decimal("0.000001")
COMPARISON_CONFIGS = (
    ("claude_no_tools", "Claude Sonnet 4.6 без инструментов", DEFAULT_MODEL, False),
    ("claude_mcp", "Claude Sonnet 4.6 с MCP", DEFAULT_MODEL, True),
    ("gpt4o_mini_mcp", "GPT-4o mini с MCP", "openai/gpt-4o-mini", True),
)


def run_case(row, output_dir, *, model="stub", budget_usd=QUESTION_BUDGET, api_key=None, use_mcp=True):
    trace = Path("traces") / f"{row['id']}.json"
    trace_file = output_dir / trace
    command = [sys.executable, "-m", "freshqa", "ask", "--mcp" if use_mcp else "--no-tools"]
    env = {name: os.environ[name] for name in ("PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR") if name in os.environ}
    env.update(PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1", PYTHON_DOTENV_DISABLED="1")
    if model == "stub":
        ledger = output_dir / "spend.sqlite3"
        timeout = CLI_TIMEOUT
    else:
        ledger = output_dir / OPENROUTER_LEDGER_NAME
        timeout = OPENROUTER_CLI_TIMEOUT
        command.extend(["--openrouter", "--model", model])
        env["OPENROUTER_API_KEY"] = api_key
    command.extend(["--budget-usd", str(budget_usd), "--output", str(trace_file),
                    "--ledger", str(ledger), "--", row["question"]])
    started = perf_counter()
    try:
        completed = subprocess.run(command, cwd=REPO, env=env, capture_output=True,
                                   encoding="utf-8", errors="replace", timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return evaluate_case(row, None, model=model, use_mcp=use_mcp, returncode=None, duration_s=perf_counter() - started,
                             diagnostic=f"CLI не завершился ({type(error).__name__}).")

    duration = perf_counter() - started
    diagnostic = None
    result = None
    if trace_file.exists():
        try:
            result = json.loads(trace_file.read_text(encoding="utf-8"))
            if not isinstance(result, dict):
                raise ValueError("Expected a trace object")
        except (ValueError, OSError, RecursionError):
            diagnostic = "Повреждённый JSON-трейс."
            result = None
    else:
        messages = (completed.stderr or "").strip().splitlines()
        diagnostic = messages[-1] if messages else "CLI не записал трейс."
        if api_key:
            diagnostic = diagnostic.replace(api_key, "[ключ скрыт]")
        diagnostic = diagnostic[:400]
    return evaluate_case(row, result, model=model, use_mcp=use_mcp, returncode=completed.returncode, duration_s=duration,
                         trace_path=trace.as_posix() if trace_file.exists() else None, diagnostic=diagnostic)


def _ledger_spend(ledger):
    """Read the confirmed portion even if a receipt is missing or damaged."""
    known = Decimal(0)
    certain = not ledger.with_suffix(".inflight.json").exists()
    try:
        with ledger.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    cost = json.loads(line, parse_float=Decimal)["cost_usd"]
                except (ValueError, KeyError, TypeError, RecursionError):
                    certain = False
                    continue
                if type(cost) not in (int, Decimal) or not Decimal(cost).is_finite() or cost < 0:
                    certain = False
                    continue
                known += cost
    except FileNotFoundError:
        pass
    except (OSError, UnicodeError):
        certain = False
    return known, certain


def _budget_summary(budget, known, stop_reason):
    confirmed = stop_reason != "accounting_uncertain"
    return {
        "question_budget_usd": float(OPENROUTER_QUESTION_BUDGET),
        "run_budget_usd": float(budget),
        "run_cost_usd": float(known) if confirmed else None,
        "run_known_cost_usd": float(known),
        "run_remaining_usd": float(max(Decimal(0), budget - known)) if confirmed else None,
        "stop_reason": stop_reason,
    }


def _run_openrouter(rows, output_dir, budget, api_key, *, model=DEFAULT_MODEL, use_mcp=True, stop_reason=None):
    known = Decimal(0)
    cases = []
    ledger = output_dir / OPENROUTER_LEDGER_NAME
    for row in rows:
        if stop_reason is None:
            remaining = (budget - known).quantize(MICRO_DOLLAR, rounding=ROUND_DOWN)
            if remaining <= 0:
                stop_reason = "run_budget_exceeded"
        if stop_reason is not None:
            case = evaluate_case(row, None, model=model, use_mcp=use_mcp, returncode=None, duration_s=0.0)
            case["status"] = stop_reason
            cases.append(case)
            continue

        effective = min(OPENROUTER_QUESTION_BUDGET, remaining)
        case = run_case(row, output_dir, model=model, budget_usd=effective, api_key=api_key, use_mcp=use_mcp)
        cases.append(case)
        recorded, certain = _ledger_spend(ledger)
        trace_cost = case["known_cost_usd"]
        agrees = trace_cost is None or isclose(float(recorded - known), trace_cost, rel_tol=0, abs_tol=1e-12)
        if not certain or recorded < known or not agrees or case["status"] == "cost_unknown":
            stop_reason = "accounting_uncertain"
        elif recorded > budget or (effective < OPENROUTER_QUESTION_BUDGET and case["status"] == "budget_exceeded"):
            stop_reason = "run_budget_exceeded"
        known = max(known, recorded)

    return cases, {"model": model, **_budget_summary(budget, known, stop_reason)}, known


def _compare(rows, output_dir, budget, api_key):
    known = Decimal(0)
    stop_reason = None
    configs = []
    for identifier, label, model, use_mcp in COMPARISON_CONFIGS:
        directory = output_dir / identifier
        (directory / "traces").mkdir(parents=True)
        cases, billing, charge = _run_openrouter(
            rows, directory, max(Decimal(0), budget - known), api_key,
            model=model, use_mcp=use_mcp, stop_reason=stop_reason,
        )
        known += charge
        stop_reason = billing["stop_reason"]
        for case in cases:
            if case["trace_path"] is not None:
                case["trace_path"] = f"{identifier}/{case['trace_path']}"
        configs.append({
            "id": identifier, "label": label, "use_mcp": use_mcp, **billing, **summarize(cases, mode="openrouter"),
        })
    return {"mode": "comparison", "configurations": configs, **_budget_summary(budget, known, stop_reason)}


def run_benchmark(questions_path, output_dir, *, openrouter=False, compare=False, budget_usd=None):
    if openrouter and compare:
        raise ValueError("Выберите один режим запуска из --openrouter и --compare.")
    if openrouter or compare:
        flag = "--compare" if compare else "--openrouter"
        if budget_usd is None:
            raise ValueError(f"Для {flag} укажите общий бюджет прогона через --budget-usd.")
        budget = limit_micros(budget_usd) * MICRO_DOLLAR
        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if budget and (not api_key or any(char.isspace() for char in api_key)):
            raise ValueError(f"Для {flag} задайте OPENROUTER_API_KEY в окружении.")
    elif budget_usd is not None:
        raise ValueError("Общий --budget-usd доступен с --openrouter или --compare.")
    rows = load_rows(questions_path)
    today = datetime.now(ZoneInfo(MOSCOW)).date().isoformat()
    if problems := validate_rows(rows, today=today):
        raise ValueError("\n".join(problems))
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    if not compare:
        (output_dir / "traces").mkdir()
    render = render_report
    if compare:
        summary = _compare(rows, output_dir, budget, api_key)
        render = render_comparison
    elif openrouter:
        cases, budget_summary, _ = _run_openrouter(rows, output_dir, budget, api_key)
        summary = summarize(cases, mode="openrouter")
        summary.update(budget_summary)
    else:
        summary = summarize([run_case(row, output_dir) for row in rows])
        summary["question_budget_usd"] = float(QUESTION_BUDGET)
    summary["dataset_hash"] = dataset_hash(rows)
    write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(render(summary), encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Прогнать 30 вопросов ДЗ 4 через FreshQA")
    parser.add_argument("--out", type=Path, help="Новый каталог для трейсов и отчёта")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--openrouter", action="store_true", help="Платный прогон на Claude Sonnet 4.6 с MCP")
    modes.add_argument("--compare", action="store_true", help="Платное сравнение трёх конфигураций")
    parser.add_argument("--budget-usd", metavar="USD", help="Общий бюджет платного прогона или сравнения")
    args = parser.parse_args(argv)
    stamp = datetime.now(ZoneInfo(MOSCOW)).strftime("%Y%m%d-%H%M%S-%f")
    prefix = "hw4-mcp-benchmark-openrouter" if args.openrouter else "hw4-mcp-benchmark"
    if args.compare:
        prefix = "hw4-mcp-benchmark-comparison"
    output = args.out if args.out is not None else REPO / "tmp" / f"{prefix}-{stamp}"
    try:
        summary = run_benchmark(QUESTIONS, output, openrouter=args.openrouter, compare=args.compare, budget_usd=args.budget_usd)
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    if args.compare:
        for config in summary["configurations"]:
            counts = config["totals"]
            print(f"{config['label']} — {counts['answer_correct']}/{counts['questions']} правильных ответов.")
        print(f"Отчёт {output.resolve() / 'report.md'}")
        return 2 if summary["stop_reason"] or any(config["totals"]["errors"] for config in summary["configurations"]) else 0
    totals = summary["totals"]
    print(f"Вопросов обработано {totals['completed']}/{totals['questions']}. "
          f"Ответов с подтверждённым источником {totals['verified_correct']}. "
          f"Отчёт {output.resolve() / 'report.md'}")
    return 2 if totals["errors"] or summary.get("stop_reason") else 0


if __name__ == "__main__":
    raise SystemExit(main())
