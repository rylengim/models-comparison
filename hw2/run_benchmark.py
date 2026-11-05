"""Replay the HW2 benchmark offline, or collect missing measurements with explicit live settings."""
import argparse
import json
from contextlib import ExitStack
from io import StringIO
from pathlib import Path

from dotenv import dotenv_values

from freshqa.dataset import load_rows
from hw2.benchmark import CONFIGS, _signature, benchmark
from hw2.benchmark_report import load_records, render_report
from hw2.benchmark_snapshot import load_snapshot, save_snapshot
from hw2.chunk_corpus import chunks
from hw2.prepare_corpus import DATA, ROOT
from hw2.prior_runs import load_prior_runs


MODELS = {
    "strong_no_tools": "anthropic/claude-sonnet-4.6",
    "cheap_live_search": "openai/gpt-4o-mini",
    "cheap_rag": "openai/gpt-4o-mini",
    "cheap_agent_kb": "openai/gpt-4o-mini",
    "medium_rag": "anthropic/claude-haiku-4.5",
}
TRACES = ROOT / "tmp" / "hw2-benchmark"
REPORT = DATA.parent / "results" / "benchmark"


def project_key(path):
    """Read the one specified credential from the selected local file."""
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            key, sep, _ = line.strip().removeprefix("export ").partition("=")
            if sep and key.strip() == "OPENROUTER_API_KEY":
                value = dotenv_values(stream=StringIO(line), interpolate=False).get("OPENROUTER_API_KEY")
                if value and value.strip():
                    return value.strip()
    raise ValueError("В выбранном локальном файле отсутствует OPENROUTER_API_KEY.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Заполнить оставшиеся строки вызовами OpenRouter")
    parser.add_argument("--budget-usd", type=float, help="Общий предел новых расходов ДЗ‑2, включая предыдущие попытки")
    parser.add_argument("--key-file", type=Path, default=ROOT / ".env",
                        help="Файл ключа OpenRouter, привязанного к этому устройству (только для --live)")
    parser.add_argument("--traces", type=Path, default=TRACES)
    parser.add_argument("--out", type=Path, default=REPORT)
    parser.add_argument("--snapshot", type=Path,
                        help="Переносимые измерения ДЗ‑2. По умолчанию используется снимок из --out или из результатов проекта.")
    args = parser.parse_args()
    rows = load_rows(DATA / "questions.jsonl") + load_rows(DATA / "unanswerable.jsonl")
    prior = load_prior_runs(rows)
    chunks_path = DATA / "chunks_structure.jsonl"
    snapshot = args.snapshot or args.out / "measured.json.gz"
    if args.snapshot is None and not snapshot.is_file() and (REPORT / "measured.json.gz").is_file():
        snapshot = REPORT / "measured.json.gz"
    signature = _signature(rows, MODELS, max_steps=6)
    manifest_path = args.traces / "manifest.json"
    if snapshot.exists():
        if (args.traces / "inflight.json").exists():
            parser.error("В локальном журнале есть незавершённый запрос. Сначала сверьте списания.")
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest["signature"] != signature:
                parser.error("Локальный журнал относится к другому набору или протоколу.")
        records = load_snapshot(snapshot, signature, rows, prior)
        complete = True
        print("Используются сохранённые измерения ДЗ‑1 и ДЗ‑2.")
    else:
        # First check the ledger without opening Milvus or loading a credential.
        options = dict(out_dir=args.traces, models=MODELS, prior_records=prior, resume=True)
        complete = benchmark(rows, **options)
        if args.live and not complete:
            if not (args.budget_usd and 0 < args.budget_usd < float("inf")):
                parser.error("Для новых вызовов нужен положительный конечный --budget-usd")
            if not chunks_path.is_file():
                parser.error("Сначала выполните python3 -m hw2.chunk_corpus.")
            from hw2.benchmark_client import BenchmarkClient
            from hw2.knowledge_base import KnowledgeBase

            with ExitStack() as resources:
                kb = resources.enter_context(KnowledgeBase())
                client = BenchmarkClient(project_key(args.key_file), budget_usd=args.budget_usd)
                resources.callback(client.close)
                complete = benchmark(rows, knowledge_base=kb, client=client, **options)
        records = load_records(args.traces)
        if complete:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            save_snapshot(snapshot, manifest, signature, rows, prior, records)
    # Corpus text and offsets are versioned in the run signature. Recreate the
    # index-free passages from that corpus when replaying in a clean checkout.
    passages = (c for doc in load_rows(DATA / "corpus.jsonl") for c in chunks(doc, "structure"))
    render_report(rows, records, args.out, chunks=passages)
    print(f"Измерено {len(records)} из {len(rows) * len(CONFIGS)} ответов. Отчёт: {args.out}")
    return 0 if complete or not args.live else 2


if __name__ == "__main__":
    raise SystemExit(main())
