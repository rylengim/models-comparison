"""Run the five HW2 conditions. Only benchmark() with an explicit client sends requests."""

import fcntl
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

from freshqa.agent import run_agent
from freshqa.client import BudgetExceeded, CostUnknownError, OpenRouterClient, OpenRouterError
from freshqa.experiment import HARD_FAILURE_STATUSES, dataset_hash, write_json
from freshqa.tools import ToolRegistry
from hw2.rag import rag_answer


CONFIGS = ("strong_no_tools", "cheap_live_search", "cheap_rag", "cheap_agent_kb", "medium_rag")
RAG_CONFIGS = {"cheap_rag", "medium_rag"}
TOOLS = {
    "strong_no_tools": [],
    "cheap_live_search": ["web_search", "page_find", "calculator", "python_exec"],
    "cheap_agent_kb": ["knowledge_base"],
}
ROOT = Path(__file__).resolve().parents[1]


def _signature(rows, models, max_steps):
    # A fresh clone with the same corpus, code and model IDs can resume a run.
    sources = ("freshqa/agent.py", "freshqa/tools.py", "freshqa/client.py", "hw2/rag.py",
               "hw2/knowledge_base.py", "hw2/chunk_corpus.py", "hw2/embeddings.py",
               "hw2/vector_store.py", "hw2/lexical_baseline.py", "hw2/evaluate_retrieval.py",
               "hw2/benchmark.py", "hw2/benchmark_client.py", "hw2/data/corpus.jsonl")
    return {"format": 1, "dataset_sha256": dataset_hash(rows), "models": dict(models),
            "max_steps": max_steps, "tools": TOOLS,
            "protocol_sha256": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in sources}}


def _read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _known_cost(cost):
    return type(cost) in (int, float) and math.isfinite(cost) and cost >= 0


def _retrieved_ids(events):
    ids = []
    for event in events:
        if event["type"] == "tool" and event["name"] == "knowledge_base":
            output = json.loads(event["output"])
            if isinstance(output, list):
                ids.extend(hit["id"] for hit in output)
    return list(dict.fromkeys(ids))


def _measure(name, question, model, client, kb, registry, max_steps):
    if name in RAG_CONFIGS:
        answer = rag_answer(question, client=client, model=model, knowledge_base=kb)
        usage = answer["usage"]
        no_calls = answer["status"] == "no_hits"
        return {"final": answer["answer"], "status": answer["status"], "rag_result": answer,
                "retrieved_ids": answer["retrieved_ids"], "steps": int(not no_calls),
                "cost_usd": 0 if no_calls else usage.get("cost_usd"),
                "prompt_tokens": usage.get("prompt_tokens", 0 if no_calls else None),
                "completion_tokens": usage.get("completion_tokens", 0 if no_calls else None)}
    trace = run_agent(question, model=model, registry=registry, client=client,
                      tool_names=TOOLS[name], max_steps=max_steps)
    calls = [e for e in trace["events"] if e["type"] == "llm"]
    for key in ("prompt_tokens", "completion_tokens"):
        trace[key] = sum(e[key] for e in calls) if all(key in e for e in calls) else None
    trace["retrieved_ids"] = _retrieved_ids(trace["events"])
    return trace


def _save(root, record, row):
    from hw2.benchmark_report import classify

    outcome = classify(row, record)
    answerable = row.get("answerable", True)
    record = {**record, "outcome": outcome, "answerable": answerable,
              "correct": outcome in {"correct_answer", "correct_refusal"} or (outcome == "no_hits" and not answerable),
              "refused": outcome in {"correct_refusal", "false_refusal", "no_hits"}}
    write_json(root / record["config"] / f"{record['id']}.json", record)


def _check_record(record, row, models):
    if (record.get("question") != row["question"] or record["model"] != models[record["config"]]
            or record["origin"] not in {"hw1", "hw2"}):
        raise ValueError(f"Трейс расходится с вопросом или конфигурацией: {row['id']}")


def benchmark(rows, *, out_dir, models, client=None, knowledge_base=None, prior_records=(),
              config_names=CONFIGS, max_steps=6, resume=False, registry=None):
    """Import verified HW1 rows, optionally measure missing rows, return True on completion.

    models maps all five config names to provider model IDs. No client means
    import-only. budget_usd applies to NEW HW2 calls, including resumed charges;
    clients check this threshold between requests, so the last call may exceed it.
    """
    rows = list(rows)
    by_id = {r["id"]: r for r in rows}
    names = tuple(dict.fromkeys(config_names))
    if not rows or len(by_id) != len(rows) or any(not re.fullmatch(r"[a-zA-Z0-9_-]+", k) for k in by_id):
        raise ValueError("Нужен непустой набор вопросов с уникальными безопасными ID.")
    if set(models) != set(CONFIGS) or not names or set(names) - models.keys():
        raise ValueError("Проверьте имена пяти конфигураций и IDs моделей.")
    if not all(isinstance(m, str) and m.strip() for m in models.values()) or type(max_steps) != int or max_steps < 1:
        raise ValueError("Нужны непустые ID моделей и положительный max_steps.")
    if client is not None and (getattr(client, "accounting_uncertain", False)
                               or getattr(client, "cost_basis", "provider_reported") != "provider_reported"):
        raise ValueError("Для активного запуска нужен клиент с подтвержденной стоимостью.")
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "inflight.json"
    manifest = root / "manifest.json"
    blocked = "Незавершённый или финансово неопределённый запрос. Перед продолжением нужна сверка списаний."
    # One writer through manifest checks, ledger restoration and the last result commit.
    with (root / ".benchmark.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Этот каталог уже занят работающим benchmark.") from None
        if marker.exists():
            raise RuntimeError(blocked)
        signature = _signature(rows, models, max_steps)
        if manifest.exists():
            if not resume or _read(manifest)["signature"] != signature:
                raise ValueError("Для прежнего прогона нужны resume и совпадающие данные, код и настройки.")
        elif any(root.glob("*/*.json")):
            raise ValueError("Найдены записи без манифеста. Выберите новый каталог.")
        else:
            write_json(manifest, {"signature": signature, "created_at": datetime.now(timezone.utc).isoformat()})

        existing = {}
        new_spend = 0.0
        measured = isinstance(client, OpenRouterClient)
        for path in sorted(root.glob("*/*.json")):
            record = _read(path)
            key = (path.parent.name, path.stem)
            if key != (record["config"], record["id"]) or key[0] not in models or key[1] not in by_id:
                raise ValueError(f"Посторонняя запись в каталоге прогона: {path.name}")
            _check_record(record, by_id[key[1]], models)
            if not _known_cost(record["cost_usd"]) or record["status"] in {"error", "cost_unknown"}:
                raise RuntimeError(blocked)
            if record["origin"] == "hw2":
                if client is not None and measured != record["measured"]:
                    raise ValueError("Для тестового и измеренного прогона нужны отдельные каталоги.")
                new_spend += record["cost_usd"]
            existing[key] = record

        for prior in prior_records:
            name, qid = prior["config"], prior["id"]
            if name not in names:
                continue
            row = by_id[qid]
            _check_record(prior, row, models)
            if (prior["origin"] != "hw1" or name not in {"strong_no_tools", "cheap_live_search"}
                    or (name == "cheap_live_search" and max_steps != 6) or prior["measured"] is not True
                    or prior.get("cost_basis") != "provider_reported"
                    or not _known_cost(prior["cost_usd"]) or prior["status"] in HARD_FAILURE_STATUSES):
                raise ValueError(f"Непроверенная архивная запись: {name}/{qid}")
            if (name, qid) not in existing:
                _save(root, prior, row)
                existing[name, qid] = prior

        pending = [(name, row) for name in names for row in rows if (name, row["id"]) not in existing]
        partial = any(r["status"] == "budget_exceeded" and r["config"] in names for r in existing.values())
        if client is None or not pending:
            return not pending and not partial
        if knowledge_base is None and any(name in RAG_CONFIGS or name == "cheap_agent_kb" for name, _ in pending):
            raise ValueError("Для выбранных конфигураций требуется открытая локальная база знаний.")
        if registry is None:
            registry = ToolRegistry(knowledge_search=knowledge_base.search if knowledge_base is not None else None)
        client.spent_usd = max(client.spent_usd, new_spend)
        for name, row in pending:
            if client.spent_usd >= client.budget_usd:
                return False
            before = client.spent_usd
            write_json(marker, {"id": row["id"], "config": name, "known_new_spend_usd": before,
                                "started_at": datetime.now(timezone.utc).isoformat()})
            # A crash, unhandled exception or Ctrl-C leaves the marker for manual reconciliation.
            try:
                record = _measure(name, row["question"], models[name], client, knowledge_base, registry, max_steps)
            except BudgetExceeded:
                # Mandatory RAG has one generation. A pre-send rejection incurs no charge.
                if client.spent_usd != before:
                    raise RuntimeError(blocked) from None
                marker.unlink()
                return False
            except OpenRouterError as error:
                uncertain = isinstance(error, CostUnknownError) or getattr(client, "accounting_uncertain", False)
                record = {"status": "cost_unknown" if uncertain else "error", "final": "unknown",
                          "error": str(error), "cost_usd": None if uncertain else client.spent_usd - before,
                          "steps": 1, "retrieved_ids": [], "prompt_tokens": None, "completion_tokens": None}
            if (record["status"] == "budget_exceeded" and record["steps"] == 0
                    and record["cost_usd"] == 0 and client.spent_usd == before):
                marker.unlink()
                return False
            record.update(id=row["id"], question=row["question"], model=models[name], config=name,
                          provider=getattr(client, "provider", "openrouter" if measured else "injected"),
                          measured=measured, origin="hw2", cost_basis="provider_reported",
                          recorded_at=datetime.now(timezone.utc).isoformat())
            cost = record["cost_usd"]
            if (not _known_cost(cost) or getattr(client, "accounting_uncertain", False)
                    or not math.isclose(cost, client.spent_usd - before, abs_tol=1e-9)):
                record.update(status="cost_unknown", cost_usd=None)
                _save(root, record, row)
                raise RuntimeError(blocked)
            _save(root, record, row)
            marker.unlink()
            if record["status"] in HARD_FAILURE_STATUSES:
                return False
        return not partial
