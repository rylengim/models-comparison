"""CLI: python -m freshqa --help."""
import argparse
import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path

from dotenv import load_dotenv

from .dataset import load_rows, validate_rows
from .experiment import LABELS, REQUIRED_CONFIGS, benchmark, configurations, write_json
from .reporting import render_report


def get_client(budget):
    from .client import OpenRouterClient
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise ValueError("Нет OPENROUTER_API_KEY. Заполните .env по .env.example. Реальный прогон не выполнен.")
    return OpenRouterClient(key, budget_usd=budget)


def main(argv=None):
    load_dotenv(Path.cwd() / ".env")
    parser = argparse.ArgumentParser(description="Домашняя работа: свежие вопросы и агент")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="Проверка JSONL без модели и ключа")
    validate.add_argument("data", nargs="?", default="data/fresh.jsonl")
    schemas = sub.add_parser("schemas", help="Показать схемы инструментов")
    schemas.add_argument("--extra-search", action="store_true")
    report = sub.add_parser("report", help="Построить таблицу/график из сохранённых трейсов")
    report.add_argument("--data", default="data/fresh.jsonl")
    report.add_argument("--traces", default="traces")
    report.add_argument("--out", default="reports")
    run = sub.add_parser("benchmark", help="Реальный платный эксперимент, нужен ключ")
    run.add_argument("--data", default="data/fresh.jsonl")
    run.add_argument("--traces", default="traces")
    run.add_argument("--out", default="reports")
    run.add_argument("--configs", nargs="+", choices=list(LABELS), default=REQUIRED_CONFIGS)
    run.add_argument("--bonuses", action="store_true", help="Все семь конфигураций")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--limit", type=int, help="Только smoke; используйте отдельные --traces и --out")
    run.add_argument("--budget-usd", type=float, default=float(os.getenv("BUDGET_USD", "5")))
    run.add_argument("--max-steps", type=int, default=int(os.getenv("MAX_STEPS", "6")))
    ask = sub.add_parser("ask", help="Один вопрос агенту")
    ask.add_argument("question")
    ask.add_argument("--model", default=configurations()["cheap_search_read"]["model"])
    ask_tools = ask.add_mutually_exclusive_group()
    ask_tools.add_argument("--extra-search", action="store_true")
    ask_tools.add_argument("--knowledge-base", action="store_true",
                           help="Дать агенту только локальный поиск ДЗ‑2. Ответ модели использует обычный API.")
    ask.add_argument("--output", default="tmp/answer_trace.json")
    ask.add_argument("--budget-usd", type=float, default=float(os.getenv("BUDGET_USD", "5")))
    args = parser.parse_args(argv)
    try:
        if args.command == "schemas":
            from .tools import ToolRegistry
            names = configurations()["cheap_extra_search" if args.extra_search else "cheap_search_read"]["tools"]
            print(json.dumps(ToolRegistry(args.extra_search).schemas(names), ensure_ascii=False, indent=2))
            return 0
        if args.command == "ask":
            from .agent import run_agent
            from .tools import ToolRegistry
            client = get_client(args.budget_usd)
            with ExitStack() as resources:
                if args.knowledge_base:
                    from hw2.knowledge_base import KnowledgeBase
                    kb = resources.enter_context(KnowledgeBase())
                    registry = ToolRegistry(knowledge_search=kb.search)
                    tools = ["knowledge_base"]
                else:
                    registry = ToolRegistry(args.extra_search)
                    tools = configurations()["cheap_extra_search" if args.extra_search else "cheap_search_read"]["tools"]
                result = run_agent(args.question, model=args.model, client=client,
                                   registry=registry, tool_names=tools)
            write_json(args.output, result)
            print(f"FINAL: {result['final']}\nstatus={result['status']}; trace={args.output}")
            return 0 if result["status"] == "ok" else 2
        rows = load_rows(args.data)
        problems = validate_rows(rows)
        if problems:
            print("\n".join(problems), file=sys.stderr)
            return 1
        if args.command == "validate":
            print(f"Записей: {len(rows)}; проблем формата и дословного вхождения: 0.")
            print("Это не проверка существования цитат в интернете и не замер свежести Sonnet.")
            return 0
        if args.command == "report":
            render_report(rows, args.traces, args.out)
            print(f"Отчёт: {args.out}/report.md")
            return 0
        if args.limit is not None:
            if args.limit < 1:
                raise ValueError("--limit должен быть положительным")
            if args.traces == "traces" or args.out == "reports":
                raise ValueError("Для --limit укажите отдельные --traces и --out, чтобы не смешать smoke и полный прогон.")
            rows = rows[:args.limit]
        if args.max_steps < 1:
            raise ValueError("--max-steps должен быть положительным")
        client = get_client(args.budget_usd)
        done = benchmark(rows, trace_dir=args.traces,
                         config_names=list(LABELS) if args.bonuses else args.configs,
                         client=client, max_steps=args.max_steps, resume=args.resume)
        render_report(rows, args.traces, args.out)
        print(f"{'Готово' if done else 'Прогон остановлен'}; отчёт: {args.out}/report.md")
        return 0 if done else 2
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
