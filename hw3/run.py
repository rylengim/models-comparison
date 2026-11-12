"""Run a resumable measured experiment; use --report for offline reproduction."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from dotenv import dotenv_values
from freshqa.agent import run_agent
from freshqa.dataset import load_rows
from .agents import run_bare, run_graph, run_critic, run_planned
from .client import MeteredClient
from .evaluation import score
from .report import render_report
from .snapshot_tools import SnapshotRegistry

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ["bare", "langgraph", "strong_direct", "critic", "plan"]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def choose_runner(root, rows):
    candidates = []
    for name in ("bare", "langgraph"):
        paths = [root / name / (r["id"] + ".json") for r in rows]
        if not all(p.exists() for p in paths):
            raise ValueError("Complete bare and langgraph before selecting a pattern executor")
        records = [json.loads(p.read_text()) for p in paths]
        if any(r["cost_usd"] is None or r["status"] in {"error", "cost_unknown", "budget_exceeded", "output_truncated", "pending"} for r in records):
            raise ValueError("Cannot select an executor from an interrupted configuration")
        candidates.append({"config": name, "correct": sum(score(t, r["final"]) for t, r in zip(rows, records)),
                           "cost_usd": sum(r["cost_usd"] for r in records)})
    best = min(candidates, key=lambda r: (-r["correct"], r["cost_usd"]))["config"]
    save(root / "executor_selection.json", {"rule": "highest accuracy, then lowest measured cost", "candidates": candidates, "selected": best})
    return (run_bare if best == "bare" else run_graph), best


def probe(client, registry, root, cheap):
    if (root / "framework_probe.json").exists():
        return
    question = "What is 2 + 2? Reply with the number only."
    first = {}
    for config, runner in [("probe_bare", run_bare), ("probe_langgraph", run_graph)]:
        client.set_scope(config, "empty-query", max_calls=6)
        before = len(client.records)
        result = runner(question, client=client, registry=registry, model=cheap)
        records = client.records[before:]
        if not records or result["status"] in {"error", "cost_unknown", "budget_exceeded"}:
            raise RuntimeError("Framework probe failed")
        first[config] = records[0]
        save(root / (config + "_request.json"), records[0]["request"])
    a, b = first.values()
    save(root / "framework_probe.json", {"question": question,
         "byte_equal": a["request_bytes"] == b["request_bytes"],
         "bare_prompt_tokens": a["prompt_tokens"], "langgraph_prompt_tokens": b["prompt_tokens"],
         "token_tax": b["prompt_tokens"] - a["prompt_tokens"],
         "bare_sha256": hashlib.sha256(a["request_bytes"].encode()).hexdigest(),
         "langgraph_sha256": hashlib.sha256(b["request_bytes"].encode()).hexdigest()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="+", choices=CONFIGS, default=CONFIGS)
    parser.add_argument("--out", type=Path, default=ROOT / "hw3/results/main")
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument("--budget-usd", type=float, default=2)
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    rows = load_rows(ROOT / "hw3/data/questions.jsonl")[:args.limit]
    root = args.out.resolve()
    if args.report:
        render_report(root, rows, root)
        return
    root.mkdir(parents=True, exist_ok=True)
    if args.limit < 1:
        raise ValueError("Positive question limit required")
    local = dotenv_values(ROOT / ".env")
    cheap = local.get("CHEAP_MODEL") or "openai/gpt-4o-mini"
    strong = local.get("STRONG_MODEL") or "anthropic/claude-sonnet-4.6"
    rates = json.loads((ROOT / "hw3/model_rates.json").read_text())
    sources = ["hw3/agents.py", "hw3/client.py", "hw3/run.py", "hw3/evaluation.py",
               "hw3/snapshot_tools.py", "freshqa/agent.py", "freshqa/tools.py", "freshqa/dataset.py",
               "hw2/lexical_baseline.py", "hw2/prepare_corpus.py", "hw3/data/corpus.jsonl"]
    signature = {"questions": hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest(),
                 "cheap": cheap, "strong": strong, "max_steps": 6, "question_calls": 24,
                 "sources": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources}}
    manifest = root / "manifest.json"
    if manifest.exists():
        if not args.resume or json.loads(manifest.read_text()) != signature:
            raise ValueError("Use --resume with exactly the same data, code and model settings")
    else:
        save(manifest, signature)
    if (root / "inflight_question.json").exists():
        raise RuntimeError("Interrupted question exists: inspect ledger before continuing")
    diagnostic = ROOT / "hw3/results/online_diagnostic/diagnostic.json"
    prior_cost = json.loads(diagnostic.read_text())["known_cost_usd"] if diagnostic.exists() else 0
    client = MeteredClient(local.get("OPENROUTER_API_KEY"), root / "calls.jsonl",
                           budget_usd=args.budget_usd - prior_cost, rates=rates)
    registry = SnapshotRegistry()
    try:
        probe(client, registry, root, cheap)
        for config in args.configs:
            runner, selected = choose_runner(root, rows) if config in {"critic", "plan"} else (None, None)
            for i, row in enumerate(rows, 1):
                target = root / config / (row["id"] + ".json")
                if target.exists():
                    continue
                client.set_scope(config, row["id"], max_calls=24)
                save(root / "inflight_question.json", {"config": config, "id": row["id"]})
                before = len(client.records)
                kwargs = dict(client=client, registry=registry, model=strong if config == "strong_direct" else cheap)
                if config == "strong_direct":
                    result = run_agent(row["question"], **kwargs, tool_names=[])
                elif config == "bare":
                    result = run_bare(row["question"], **kwargs)
                elif config == "langgraph":
                    result = run_graph(row["question"], **kwargs)
                else:
                    result = (run_critic if config == "critic" else run_planned)(row["question"], runner=runner, **kwargs)
                calls = client.records[before:]
                result.update(id=row["id"], question=row["question"], config=config, model=kwargs["model"],
                              executor=selected, call_records=calls, measured=True,
                              correct=score(row, result["final"]))
                for field in ("cost_usd", "prompt_tokens", "completion_tokens"):
                    result[field] = sum(c[field] for c in calls) if all(c.get(field) is not None for c in calls) else None
                result["steps"] = len(calls)
                save(target, result)
                (root / "inflight_question.json").unlink()
                print(f"{config} {i}/{len(rows)} {row['id']} {result['status']} correct={result['correct']} total=${client.spent_usd:.6f}", flush=True)
                if result["status"] in {"error", "cost_unknown", "budget_exceeded"}:
                    raise RuntimeError("Stopped after recorded failure; inspect result and ledger")
                if (root / "STOP").exists():
                    render_report(root, rows, root)
                    print("Stopped after saving the current question; remove STOP and use --resume", flush=True)
                    return
            render_report(root, rows, root)
        print(f"Known total including prior diagnostic/pilot/probes: ${client.spent_usd + prior_cost:.8f}", flush=True)
    finally:
        client.close()
        registry.close()


if __name__ == "__main__":
    main()
