"""Portable, offline replay of the measured HW2 completions."""
import gzip
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile

from freshqa.experiment import HARD_FAILURE_STATUSES
from hw2.benchmark import CONFIGS, RAG_CONFIGS
from hw2.prior_runs import valid_usd


def pick(record, names):
    return {name: record[name] for name in names if name in record}


BILLING = ("request_id", "cost_usd", "prompt_tokens", "completion_tokens")
FIELDS = ("id", "config", "question", "model", "origin", "measured", "provider", "cost_basis",
          "status", "final", "cost_usd", "prompt_tokens", "completion_tokens", "steps",
          "retrieved_ids", "recorded_at")


def public_record(record):
    """Export the answer, visible evidence and receipts, excluding transport/host metadata."""
    result = pick(record, FIELDS)
    if record["config"] in RAG_CONFIGS:
        rag = record["rag_result"]
        result["rag_result"] = pick(rag, ("answer", "quote", "status", "retrieved_ids"))
        source = rag["source"]
        result["rag_result"]["source"] = pick(source, ("id", "url", "section", "page")) if source else None
        result["rag_result"]["usage"] = pick(rag["usage"], BILLING)
    else:
        events = []
        for item in record["events"]:
            if item["type"] == "tool":
                events.append(pick(item, ("type", "step", "name", "output")))
            else:
                event = pick(item, ("type", "step", "finish_reason", *BILLING))
                event["message"] = pick(item["message"], ("role", "content", "tool_calls"))
                events.append(event)
        result["events"] = events
    return result


def complete_records(new, prior, rows, signature):
    """Validate identity and coverage, including known per-call charges, before reuse."""
    questions = {r["id"]: r["question"] for r in rows}
    expected = {(name, qid) for name in CONFIGS for qid in questions}
    if len(rows) != len(questions):
        raise ValueError("Повтор ID в выборке.")
    combined = {(r["config"], r["id"]): r for r in prior}
    if len(combined) != len(prior):
        raise ValueError("Повтор записи в исходных измерениях.")
    receipts = set()
    for r in new:
        key = (r["config"], r["id"])
        if (key in combined or key not in expected or r["question"] != questions[key[1]]
                or r["model"] != signature["models"][key[0]]):
            raise ValueError(f"Дублирующаяся или несовместимая запись снимка: {key}")
        if (r["origin"] != "hw2" or r["measured"] is not True or r["provider"] != "openrouter"
                or r["cost_basis"] != "provider_reported" or not valid_usd(r["cost_usd"])
                or r["status"] in HARD_FAILURE_STATUSES):
            raise ValueError(f"Нужен завершённый оплаченный или нулевой результат ДЗ‑2: {key}")
        if r["config"] in RAG_CONFIGS:
            calls = [] if r["status"] == "no_hits" else [r["rag_result"]["usage"]]
        else:
            calls = [item for item in r["events"] if item["type"] == "llm"]
        for call in calls:
            rid = call.get("request_id")
            if (not isinstance(rid, str) or not rid or rid in receipts or not valid_usd(call.get("cost_usd"))
                    or any(type(call.get(k)) != int or call[k] < 0 for k in BILLING[2:])):
                raise ValueError(f"Не подтверждены расходы на отдельный вызов: {key}")
            receipts.add(rid)
        if (len(calls) != r["steps"] or (not calls and r["status"] != "no_hits")
                or not math.isclose(math.fsum(c["cost_usd"] for c in calls), r["cost_usd"], abs_tol=1e-10)
                or any(sum(c[k] for c in calls) != r[k] for k in BILLING[2:])):
            raise ValueError(f"Итог записи расходится с вызовами модели: {key}")
        combined[key] = r
    if combined.keys() != expected:
        raise ValueError("Для переносимого снимка нужны все вопросы во всех пяти конфигурациях.")
    return [combined[key] for key in sorted(combined)]


def load_snapshot(path, signature, rows, prior):
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        snapshot = json.load(handle)
    if snapshot["version"] != 1 or snapshot["signature"] != signature:
        raise ValueError("Данные, модели или протокол снимка отличаются от текущего запуска.")
    return complete_records(snapshot["records"], prior, rows, signature)


def save_snapshot(path, manifest, signature, rows, prior, records):
    if manifest["signature"] != signature:
        raise ValueError("Манифест журнала расходится с текущим протоколом.")
    new = [public_record(r) for r in records if r["origin"] == "hw2"]
    complete_records(new, prior, rows, signature)
    snapshot = {"version": 1, "signature": signature, "records": new}
    data = json.dumps(snapshot, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        pending = Path(handle.name)
    try:
        pending.write_bytes(gzip.compress(data, mtime=0))
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)
