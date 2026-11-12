"""Demonstrate a shared one-call limit across planner and executor."""
import argparse
import json
from pathlib import Path

from dotenv import dotenv_values

from .agents import run_graph, run_planned
from .client import MeteredClient
from .run import ROOT, save
from .snapshot_tools import SnapshotRegistry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "hw3/results/main")
    args = parser.parse_args()
    root = args.out.resolve()
    target = root / "limit_demo.json"
    if target.exists():
        print(target.read_text())
        return
    if (root / "inflight_question.json").exists() or (root / "calls.inflight.json").exists():
        raise RuntimeError("Finish or reconcile the active experiment before running the demo")
    local = dotenv_values(ROOT / ".env")
    rates = json.loads((ROOT / "hw3/model_rates.json").read_text())
    prior = json.loads((ROOT / "hw3/results/online_diagnostic/diagnostic.json").read_text())["known_cost_usd"]
    client = MeteredClient(local.get("OPENROUTER_API_KEY"), root / "calls.jsonl", budget_usd=2 - prior, rates=rates)
    registry = SnapshotRegistry()
    question = "On what date did Ukraine boycott the closing ceremony of the 2026 Winter Paralympics?"
    try:
        client.set_scope("limit_demo", "planner-exhausts-one-call", max_calls=1)
        before = len(client.records)
        result = run_planned(question, runner=run_graph, client=client, registry=registry,
                             model=local.get("CHEAP_MODEL") or "openai/gpt-4o-mini")
        calls = client.records[before:]
        result.update(question=question, max_calls=1, actual_http_calls=len(calls), call_records=calls,
                      limit_triggered=result["status"] == "budget_exceeded" and len(calls) == 1,
                      explanation="The planner consumes the only permitted call; the executor stops before its HTTP request.")
        save(target, result)
        print(json.dumps({k: result[k] for k in ("status", "final", "actual_http_calls", "limit_triggered", "cost_usd")}, ensure_ascii=False))
        if not result["limit_triggered"]:
            raise RuntimeError("Saved actual result does not demonstrate the intended limit")
    finally:
        client.close()
        registry.close()


if __name__ == "__main__":
    main()
