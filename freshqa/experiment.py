"""Reproducible experiment configuration, resumable traces and aggregation."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .dataset import final_text, score_row

LABELS = {
    "strong_no_tools": "Сильная, без инструментов",
    "cheap_no_tools": "Дешёвая, без инструментов",
    "cheap_search": "Дешёвая, поиск",
    "cheap_search_read": "Дешёвая, поиск + чтение",
    "medium_search_read": "Средняя, поиск + чтение",
    "cheap_extra_search": "Дешёвая, Wiki + второй поиск + чтение",
    "cascade": "Каскад: дешёвая → средняя",
}
REQUIRED_CONFIGS = list(LABELS)[:5]
HARD_FAILURE_STATUSES = {"error", "cost_unknown", "budget_exceeded"}


def configurations():
    cheap = os.getenv("CHEAP_MODEL", "openai/gpt-4o-mini")
    medium = os.getenv("MEDIUM_MODEL", "anthropic/claude-haiku-4.5")
    strong = os.getenv("STRONG_MODEL", "anthropic/claude-sonnet-4.6")
    full = ["web_search", "page_find", "calculator", "python_exec"]
    return {
        "strong_no_tools": {"model": strong, "tools": []},
        "cheap_no_tools": {"model": cheap, "tools": []},
        "cheap_search": {"model": cheap, "tools": ["web_search", "calculator", "python_exec"]},
        "cheap_search_read": {"model": cheap, "tools": full},
        "medium_search_read": {"model": medium, "tools": full},
        "cheap_extra_search": {"model": cheap, "tools": full + ["web_search_extra"]},
        "cascade": {"model": cheap, "fallback_model": medium, "tools": full + ["web_search_extra"]},
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def dataset_hash(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def aggregate(traces, total):
    n = len(traces)
    right = sum(bool(t["correct"]) for t in traces)
    cost = sum(t["cost_usd"] for t in traces) if n and all(t["cost_usd"] is not None for t in traces) else None
    complete = (n == total and n > 0 and cost is not None
                and all(t.get("status") not in HARD_FAILURE_STATUSES for t in traces))
    return dict(n=n, total=total, complete=complete, correct=right,
                accuracy=right / n if n else None,
                cost_total=cost, cost_per_task=cost / n if cost is not None else None,
                cost_per_correct=cost / right if cost is not None and right else None,
                mean_steps=sum(t["steps"] for t in traces) / n if n else None,
                mean_seconds=sum(t["elapsed_s"] for t in traces) / n if n else None)


def run_cascade(question, *, cheap, medium, client, registry, tool_names, max_steps, run=None):
    if run is None:
        from .agent import run_agent as run
    first = run(question, model=cheap, client=client, registry=registry,
                tool_names=tool_names, max_steps=max_steps)
    if first["status"] in HARD_FAILURE_STATUSES:
        return first
    if final_text(first["final"]).lower() not in {"unknown", "неизвестно", ""} and first["status"] == "ok":
        return first
    second = run(question, model=medium, client=client, registry=registry,
                 tool_names=tool_names, max_steps=max_steps)
    costs = [first["cost_usd"], second["cost_usd"]]
    return {**second, "cost_usd": sum(costs) if all(c is not None for c in costs) else None,
            "known_cost_usd": sum(t.get("known_cost_usd", t["cost_usd"] or 0) for t in [first, second]),
            "steps": first["steps"] + second["steps"],
            "elapsed_s": first["elapsed_s"] + second["elapsed_s"],
            "events": first["events"] + second["events"], "stages": [first, second]}


def benchmark(rows, *, trace_dir, config_names, client, max_steps=6, resume=False):
    from .agent import run_agent
    from .tools import ToolRegistry

    if getattr(client, "cost_basis", "provider_reported") != "provider_reported":
        raise ValueError("Основной benchmark требует подтверждённой стоимости. Для Groq используйте groq_check.")
    configs = configurations()
    invalid = set(config_names) - configs.keys()
    if invalid:
        raise ValueError(f"Неизвестные конфигурации: {sorted(invalid)}")
    root = Path(trace_dir)
    inflight_path = root / "inflight.json"
    interrupted_message = ("Обнаружен незавершённый запрос (inflight.json). "
                           "Сверьте списания в OpenRouter до продолжения; автоматический повтор заблокирован.")
    if inflight_path.exists():
        raise ValueError(interrupted_message)
    code = b"".join(p.read_bytes() for p in sorted(Path(__file__).parent.glob("*.py")))
    signature = {"dataset_sha256": dataset_hash(rows), "n": len(rows), "configs": configs,
                 "max_steps": max_steps, "code_sha256": hashlib.sha256(code).hexdigest()}
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not resume:
            raise ValueError("Каталог содержит прогон. Используйте --resume или новый --traces.")
        if old["signature"] != signature:
            raise ValueError("Данные, код или настройки изменились. Нужен новый --traces.")
    else:
        if list(root.glob("*/*.json")):
            raise ValueError("Найдены трейсы без manifest.json; выберите другой каталог.")
        write_json(manifest_path, {"signature": signature, "created_at": datetime.now(timezone.utc).isoformat()})
    previous = [json.loads(p.read_text(encoding="utf-8")) for p in root.glob("*/*.json")]
    if any(t["cost_usd"] is None for t in previous):
        raise ValueError("В прогоне есть неизвестные списания. Сверьте их в OpenRouter перед новым прогоном.")
    # The supplied client may already account for additional requests. Restoring
    # a directory's ledger must never erase those verified charges.
    client.spent_usd = max(client.spent_usd, sum(t["cost_usd"] for t in previous))
    for name in config_names:
        cfg = configs[name]
        registry = ToolRegistry(extra_search=name in {"cheap_extra_search", "cascade"})
        for index, row in enumerate(rows, 1):
            path = root / name / f"{row['id']}.json"
            if path.exists():
                continue
            if client.spent_usd >= client.budget_usd:
                return False
            # Create exclusively BEFORE inference. A crash or Ctrl-C leaves this
            # marker behind, preventing an unaccounted paid request from being
            # silently retried. Do not remove it in a finally block.
            try:
                with inflight_path.open("x", encoding="utf-8") as stream:
                    json.dump({"config": name, "id": row["id"], "known_spent_usd": client.spent_usd,
                               "started_at": datetime.now(timezone.utc).isoformat()}, stream)
            except FileExistsError:
                raise ValueError(interrupted_message) from None
            if name == "cascade":
                trace = run_cascade(row["question"], cheap=cfg["model"], medium=cfg["fallback_model"],
                                    client=client, registry=registry, tool_names=cfg["tools"], max_steps=max_steps)
            else:
                trace = run_agent(row["question"], model=cfg["model"], client=client, registry=registry,
                                  tool_names=cfg["tools"], max_steps=max_steps)
            # Gold data are added only after inference; they never enter run_agent.
            trace.update(id=row["id"], question=row["question"], expected=row["answer"],
                         correct=score_row(row, trace["final"]), config=name,
                         model=cfg["model"], measured=True,
                         recorded_at=datetime.now(timezone.utc).isoformat())
            write_json(path, trace)
            inflight_path.unlink()
            print(f"{name} {index}/{len(rows)}: {trace['status']}; spent=${client.spent_usd:.4f}", flush=True)
            if trace["status"] in HARD_FAILURE_STATUSES:
                return False
    return not any(t.get("status") in HARD_FAILURE_STATUSES for t in previous
                   if t.get("config") in config_names)
