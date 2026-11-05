"""Import measured HW1 baselines for unchanged questions without making model calls."""
import json
import math
from pathlib import Path

from freshqa.dataset import final_text, load_rows
from freshqa.experiment import dataset_hash
from hw2.prepare_corpus import ROOT


# These are the two configurations compared in the original HW1 conclusion.
BASELINES = {
    "strong_no_tools": ("strong_no_tools", "anthropic/claude-sonnet-4.6", []),
    "cheap_live_search": ("cheap_search_read", "openai/gpt-4o-mini",
                          ["web_search", "page_find", "calculator", "python_exec"]),
}


def valid_usd(value):
    return type(value) in (float, int) and math.isfinite(value) and value >= 0


def _import_trace(path, question, config, historical, model, max_steps):
    trace = json.loads(path.read_text(encoding="utf-8"))
    expected = {"config": historical, "model": model, "id": question["id"],
                "question": question["question"], "expected": question["answer"],
                "provider": "openrouter", "cost_basis": "provider_reported"}
    if trace.get("measured") is not True or any(trace.get(k) != v for k, v in expected.items()):
        raise ValueError(f"Трейс расходится с манифестом или вопросом ДЗ‑1: {path}")
    events = [e for e in trace["events"] if e["type"] == "llm"]
    if not (1 <= len(events) == trace["steps"] <= max_steps):
        raise ValueError(f"Число вызовов модели не соответствует трейсу: {path}")
    for event in events:
        if (not valid_usd(event.get("cost_usd")) or not event.get("request_id")
                or any(type(event.get(k)) != int or event[k] < 0
                       for k in ("prompt_tokens", "completion_tokens"))):
            raise ValueError(f"Нет проверяемых расходов и токенов вызова: {path}")

    total = math.fsum(e["cost_usd"] for e in events)
    if any(not valid_usd(trace.get(k)) or not math.isclose(total, trace[k], abs_tol=1e-12, rel_tol=0)
           for k in ("cost_usd", "known_cost_usd")):
        raise ValueError(f"Сумма подтверждённых расходов расходится с трейсом: {path}")
    # max_steps and repeated_tool_call can still end in a real final answer.
    last = events[-1]
    message = last.get("message", {})
    if (trace.get("status") not in {"ok", "max_steps", "repeated_tool_call", "tool_limit"}
            or last.get("finish_reason") != "stop" or message.get("role") != "assistant"
            or message.get("tool_calls") or not isinstance(message.get("content"), str)
            or not message["content"].strip() or final_text(message["content"]) != trace["final"]):
        raise ValueError(f"Нет завершённого финального ответа: {path}")
    return {
        "config": config, "prior_config": historical, "origin": "hw1", "id": trace["id"],
        "question": trace["question"], "final": trace["final"], "status": trace["status"],
        "final_message": message, "finish_reason": last["finish_reason"],
        "model": model, "provider": trace["provider"], "cost_usd": trace["cost_usd"],
        "measured": trace["measured"], "cost_basis": trace["cost_basis"],
        "steps": len(events), "prompt_tokens": sum(e["prompt_tokens"] for e in events),
        "completion_tokens": sum(e["completion_tokens"] for e in events), "retrieved_ids": [],
        "trace_path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        "recorded_at": trace["recorded_at"],
    }


def load_prior_runs(questions, *, trace_dir=ROOT / "traces", dataset=ROOT / "data" / "fresh.jsonl"):
    """Return completed records for matching answerable questions, in input order per baseline."""
    trace_dir = Path(trace_dir).resolve()
    signature = json.loads((trace_dir / "manifest.json").read_text(encoding="utf-8"))["signature"]
    old_rows = load_rows(dataset)
    if len(old_rows) != signature["n"] or dataset_hash(old_rows) != signature["dataset_sha256"]:
        raise ValueError("Набор ДЗ‑1 расходится с манифестом запуска.")
    by_id = {r["id"]: r for r in old_rows}
    questions = list(questions)
    if len(by_id) != len(old_rows) or len({q["id"] for q in questions}) != len(questions):
        raise ValueError("Повторяющийся идентификатор вопроса.")
    result = []
    for config, (historical, model, tools) in BASELINES.items():
        if signature["configs"][historical] != {"model": model, "tools": tools} or signature["max_steps"] != 6:
            raise ValueError(f"Модель, инструменты или лимит шагов изменены для {historical}.")
        for question in questions:
            old = by_id.get(question["id"])
            if (old is None or question.get("answerable") is False or old["question"] != question["question"]):
                continue
            path = trace_dir / historical / (old["id"] + ".json")
            if path.is_file():
                result.append(_import_trace(path, old, config, historical, model, 1 if not tools else 6))
    return result
