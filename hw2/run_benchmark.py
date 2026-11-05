"""Compare HW2 answer configurations; importing the existing HW1 measurements is the default."""
import argparse
from contextlib import ExitStack
from io import StringIO
from pathlib import Path

from dotenv import dotenv_values

from freshqa.dataset import load_rows
from hw2.benchmark import CONFIGS, benchmark
from hw2.benchmark_report import load_records, render_report
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
    args = parser.parse_args()
    if args.live and not (args.budget_usd and 0 < args.budget_usd < float("inf")):
        parser.error("--live требует положительного конечного --budget-usd")
    rows = load_rows(DATA / "questions.jsonl") + load_rows(DATA / "unanswerable.jsonl")
    prior = load_prior_runs(rows)
    chunks_path = DATA / "chunks_structure.jsonl"
    if args.live and not chunks_path.is_file():
        parser.error("Сначала выполните python3 -m hw2.chunk_corpus.")

    with ExitStack() as resources:
        client = kb = None
        if args.live:
            from hw2.benchmark_client import BenchmarkClient
            from hw2.knowledge_base import KnowledgeBase

            kb = resources.enter_context(KnowledgeBase())
            client = BenchmarkClient(project_key(args.key_file), budget_usd=args.budget_usd)
            resources.callback(client.close)
        complete = benchmark(rows, out_dir=args.traces, models=MODELS, prior_records=prior,
                             config_names=CONFIGS, knowledge_base=kb, client=client, resume=True)

    records = load_records(args.traces)
    chunks = load_rows(chunks_path) if chunks_path.is_file() else []
    render_report(rows, records, args.out, chunks=chunks)
    print(f"Измерено {len(records)} из {len(rows) * len(CONFIGS)} ответов. Отчёт: {args.out}")
    return 0 if complete or not args.live else 2


if __name__ == "__main__":
    raise SystemExit(main())
