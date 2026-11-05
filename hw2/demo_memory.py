"""Two sessions with a fictional user. Replay is offline; --live runs a new experiment."""
import argparse
import fcntl
import hashlib
import json
import math
from contextlib import closing
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urlsplit
from uuid import uuid4

import requests

from freshqa.tools import ToolRegistry
from hw2.benchmark_client import BenchmarkClient, PRICE_CAPS
from hw2.knowledge_base import KnowledgeBase
from hw2.memory_agent import EXTRACT_RULES, FACT_KEYS, MemorySession, answer_turn
from hw2.memory_comparison import write_comparison
from hw2.memory_store import COLLECTION, EpisodeJournal, FactStore
from hw2.prepare_corpus import ROOT
from hw2.run_benchmark import project_key


MODEL = "openai/gpt-4o-mini"
RESULTS = ROOT / "hw2" / "results" / "memory"
RUN_DIR = ROOT / "tmp" / "hw2-memory"
SESSIONS = (
    ("session-1", (
        "I prefer Tuesday for reminders. For my notes, what day should you use?",
        "I use kilometers for distances. What unit should a route card show?",
        "For a walking route, suggest a short heading for the section about bridges. Give just one heading.",
        "What is the distance unit I asked for? Reply with just the unit.",
        "Use the knowledge_base tool to look up the number of job cuts Statistics Canada announced on January 13, 2026.",
        "Give me one short title for a note about walking to the train station.",
    )),
    ("session-2", (
        "What is my preferred reminder day? Reply with just the weekday.",
        "What units should you use for distances in my route notes?",
        "I'm drafting a route card. Suggest a single short heading for the section about water stops.",
        "I have changed my preference. I now prefer Friday for reminders. Which day should you use from now on?",
        "Give me one short heading for the section on train connections in a route card.",
        "Write the weekday from my most recent correction, with no extra words.",
    )),
)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as f:
        json.dump(data, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write("\n")
        pending = Path(f.name)
    pending.replace(path)


class RecordedCalls:
    """Save one receipt per call position so a stopped paid experiment can be resumed."""
    def __init__(self, client, directory):
        self.client = client
        self.directory = directory
        self.position = 0

    def __getattr__(self, name):
        return getattr(self.client, name)

    def complete(self, *, model, messages, tools=None):
        self.position += 1
        payload = {"model": model, "messages": messages, "tools": tools}
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        receipt = self.directory / f"{self.position:03d}.json"
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved["digest"] != digest:
                raise ValueError("Изменился запрос. Сохранённое измерение относится к другому контексту.")
            return saved["response"]
        marker = self.directory / "pending.json"
        with marker.open("x") as f:
            json.dump({"position": self.position, "digest": digest}, f)
        response = self.client.complete(**payload)
        write_json(receipt, {"digest": digest, "response": response})
        marker.unlink()
        return response


class LocalClient:
    """llama.cpp on this machine reports token counts; requests have no API charge."""
    spent_usd = 0.0
    provider = "local_llama_cpp"

    def __init__(self, uri):
        url = urlsplit(uri)
        if (url.scheme != "http" or url.hostname not in {"127.0.0.1", "::1", "localhost"}
                or url.username or url.password or url.query or url.fragment):
            raise ValueError("Укажите локальный HTTP-адрес llama-server.")
        self.url = uri.rstrip("/") + "/v1/chat/completions"
        self.session = requests.Session()
        self.session.trust_env = False

    def close(self):
        self.session.close()

    def complete(self, *, model, messages, tools=None):
        body = {"model": model, "messages": messages, "temperature": 0, "seed": 1, "max_tokens": 384}
        if tools:
            body["tools"] = tools
        elif messages[0]["content"] == EXTRACT_RULES:
            schema = {"type": "object", "properties": {key: {"type": "string"} for key in FACT_KEYS},
                      "required": list(FACT_KEYS), "additionalProperties": False}
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "memory", "strict": True, "schema": schema}}
        response = self.session.post(self.url, json=body, timeout=(3, 180), allow_redirects=False)
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        return {"message": choice["message"], "finish_reason": choice["finish_reason"],
                "request_id": data["id"], "cost_usd": 0.0,
                **{k: data["usage"][k] for k in ("prompt_tokens", "completion_tokens")}}


def run_sessions(client, model, out, run_dir, *, local=False):
    run_id = "fictional-" + uuid4().hex
    journal = EpisodeJournal(run_dir / "episodes.jsonl", run_id)
    full_history = []
    sessions = []
    with FactStore(run_id) as store, KnowledgeBase() as kb:
        registry = ToolRegistry(knowledge_search=kb.search)
        for session_id, questions in SESSIONS:
            agent = MemorySession(session_id, store=store, journal=journal, client=client,
                                  registry=registry, model=model)
            turns = []
            for question in questions:
                turn = agent.ask(question)
                baseline = answer_turn(question, history=full_history, facts=[], client=client,
                                       registry=registry, model=model)
                full_history.extend([{"role": "user", "content": question},
                                     {"role": "assistant", "content": turn["answer"]}])
                turns.append({**turn, "full_answer": baseline["answer"], "full_usage": baseline["usage"],
                              "full_receipts": baseline["receipts"], "full_tool_calls": baseline["tool_calls"],
                              "full_status": baseline["status"]})
                print(f"{session_id} · {len(turns)}/{len(questions)}", flush=True)
            sessions.append({"session_id": session_id, "turns": turns, **agent.finish()})
        final_lookup = store.search("Which day should I get reminders?", limit=2)
    demo = {"model": model, "window_pairs": 1, "fact_limit": 2, "collection": COLLECTION,
            "price_basis": "Local CPU, API charges" if local else "OpenRouter usage.cost",
            "final_lookup": final_lookup, "sessions": sessions}
    validate_demo(demo)
    write_json(out / "demo.json", demo)
    return demo


def validate_demo(demo):
    first, second = demo["sessions"]
    for session, (expected_id, questions) in zip((first, second), SESSIONS, strict=True):
        if session["session_id"] != expected_id or [t["question"] for t in session["turns"]] != list(questions):
            raise ValueError("Сохранённый диалог отличается от сценария.")
        history = []
        for turn in session["turns"]:
            if turn["history"] != history:
                raise ValueError("Окно не совпадает с предыдущей парой текущей сессии.")
            history = [{"role": "user", "content": turn["question"]},
                       {"role": "assistant", "content": turn["answer"]}]
    for session, expected_day in ((first, "Tuesday"), (second, "Friday")):
        days = [f["value"].casefold() for f in session["facts_after"] if f["key"] == "reminder_day"]
        if days != [expected_day.casefold()]:
            raise ValueError("Проверка замены факта не пройдена.")
    updated = [f for f in demo["final_lookup"] if f["key"] == "reminder_day"]
    if len(updated) != 1 or updated[0]["value"].casefold() != "friday" or updated[0]["session_id"] != "session-2":
        raise ValueError("Поиск после завершения второй сессии не нашёл обновлённый факт.")
    recall = second["turns"][0]
    if (not any(f["key"] == "reminder_day" and f["value"].casefold() == "tuesday" for f in recall["facts"])
            or "tuesday" not in recall["answer"].casefold()):
        raise ValueError("Агент не вспомнил предпочтение из первой сессии.")


def compare(demo, out):
    turns, extractions = [], []
    for s in demo["sessions"]:
        for t in s["turns"]:
            turns.append({"session_id": s["session_id"], "question": t["question"],
                          "full": t["full_usage"], "window": t["usage"]})
        extractions.append({"session_id": s["session_id"], **s["extraction_usage"]})
    return write_comparison(turns, extractions, out, model=demo["model"], price_basis=demo["price_basis"],
                            reference_rates=PRICE_CAPS[MODEL] if demo["price_basis"].startswith("Local CPU") else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--local-server", help="HTTP-адрес llama-server на этом устройстве")
    parser.add_argument("--local-model", default="Qwen2.5-1.5B-Instruct-Q4_K_M")
    modes.add_argument("--live", action="store_true", help="Сделать платный прогон вымышленного диалога через OpenRouter")
    parser.add_argument("--budget-usd", type=float, help="Предел расходов именно на этот эксперимент с памятью")
    parser.add_argument("--run-dir", type=Path, default=RUN_DIR, help="Локальный каталог журнала и квитанций")
    parser.add_argument("--out", type=Path, default=RESULTS)
    args = parser.parse_args()
    source = args.out / "demo.json"
    if source.is_file() or not (args.live or args.local_server):
        source = source if source.is_file() else RESULTS / "demo.json"
        demo = json.loads(source.read_text())
        validate_demo(demo)
    else:
        if args.live and not (args.budget_usd and math.isfinite(args.budget_usd) and args.budget_usd > 0):
            parser.error("--live требует положительного конечного --budget-usd")
        args.run_dir.mkdir(parents=True, exist_ok=True)
        calls = args.run_dir / "calls"
        calls.mkdir(exist_ok=True)
        with (args.run_dir / "lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            if (calls / "pending.json").exists():
                parser.error("Открытая попытка в журнале. Перед продолжением сверьте расходы.")
            if args.live:
                spent = math.fsum(json.loads(p.read_text())["response"]["cost_usd"] for p in calls.glob("[0-9]*.json"))
                client = BenchmarkClient(project_key(ROOT / ".env"), budget_usd=args.budget_usd, spent_usd=spent)
                model = MODEL
            else:
                client = LocalClient(args.local_server)
                model = args.local_model
            with closing(client):
                demo = run_sessions(RecordedCalls(client, calls), model, args.out, args.run_dir, local=bool(args.local_server))
    report = compare(demo, args.out)
    for label in ("full", "window_answers", "extractions", "window"):
        print(label, report["totals"][label])


if __name__ == "__main__":
    main()
