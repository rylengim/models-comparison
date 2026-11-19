"""CLI: python -m freshqa --help."""
import argparse
import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path

from dotenv import load_dotenv

from .dataset import load_rows, validate_rows
from .experiment import BONUS_CONFIGS, LABELS, REQUIRED_CONFIGS, benchmark, configurations, write_json
from .reporting import DEFAULT_REPORT_OUT, render_report, validate_report_output


def get_client(budget):
    from .client import OpenRouterClient
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise ValueError("Нет OPENROUTER_API_KEY. Заполните .env по .env.example. Реальный прогон не выполнен.")
    return OpenRouterClient(key, budget_usd=budget)


def run_hw4_ask(args, parser):
    import asyncio

    if args.openrouter:
        from hw4 import mcp_openrouter

        if args.budget_usd is None:
            parser.error("Для --openrouter укажите лимит на вопрос через --budget-usd.")
        key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not key:
            mode = "--no-tools" if args.no_tools else "--mcp"
            raise ValueError(f"Для {mode} --openrouter задайте OPENROUTER_API_KEY в окружении.")
        model = args.model if args.model is not None else mcp_openrouter.DEFAULT_MODEL
        if args.no_tools:
            default_trace = mcp_openrouter.DEFAULT_NO_TOOLS_TRACE
            default_ledger = mcp_openrouter.DEFAULT_NO_TOOLS_LEDGER
            operation = mcp_openrouter.ask_no_tools
        else:
            default_trace = mcp_openrouter.DEFAULT_TRACE
            default_ledger = mcp_openrouter.DEFAULT_LEDGER
            operation = mcp_openrouter.ask_openrouter
        output = Path(args.output) if args.output is not None else default_trace
        ledger = (args.ledger if args.ledger is not None else default_ledger).resolve()
        ledger_files = (ledger, ledger.with_suffix(".inflight.json"), ledger.with_suffix(".lock"))
        options = {"api_key": key, "model": model}
    else:
        from hw4.mcp_ask import ask_publications
        from hw4.spending import DEFAULT_LEDGER

        if args.model is not None:
            parser.error("Чтобы выбрать модель с --mcp, добавьте --openrouter.")
        model = "stub"
        output = Path(args.output) if args.output is not None else Path("tmp/mcp_answer_trace.json")
        ledger = args.ledger if args.ledger is not None else DEFAULT_LEDGER
        ledger_files = (ledger,)
        operation = ask_publications
        options = {}

    trace_files = {output.resolve(), output.with_suffix(".tmp").resolve()}
    if trace_files.intersection(path.resolve() for path in ledger_files):
        raise ValueError("Укажите разные пути для трейса и файлов учёта расходов.")
    result = asyncio.run(operation(args.question, budget_usd=args.budget_usd, ledger_path=ledger, **options))
    write_json(output, result)
    print(f"FINAL: {result['final']}")
    if result.get("quote"):
        print(f"QUOTE: {result['quote']}")
    if result["source_url"]:
        print(f"SOURCE: {result['source_url']}")
    print(f"model={model}; status={result['status']}; trace={output}")
    cost = result["cost_usd"] if args.openrouter else result["simulated_cost_usd"]
    label = "Расходы" if args.openrouter else "Условные расходы"
    if cost is None:
        print("Стоимость вызова не подтверждена.", file=sys.stderr)
    else:
        print(f"{label} на вопрос: ${cost:.6f} из ${result['question_budget_usd']:.6f}.")
    day = result["daily_spend"]
    daily_cost = day.get("cost_usd" if args.openrouter else "simulated_usd") if day else None
    if daily_cost is None:
        print("Дневной счётчик расходов недоступен.", file=sys.stderr)
    else:
        print(f"{label} за {day['date']} (МСК): ${daily_cost:.6f}.")
    return 0 if result["status"] == "ok" else 2


def main(argv=None):
    parser = argparse.ArgumentParser(description="Домашняя работа: свежие вопросы и агент")
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser("validate", help="Проверка JSONL без модели и ключа")
    validate.add_argument("data", nargs="?", default="data/fresh.jsonl")
    schemas = sub.add_parser("schemas", help="Показать схемы инструментов")
    schema_tools = schemas.add_mutually_exclusive_group()
    schema_tools.add_argument("--extra-search", action="store_true")
    schema_tools.add_argument("--official-sources", action="store_true")
    report = sub.add_parser("report", help="Построить таблицу/график из сохранённых трейсов")
    report.add_argument("--data", default="data/fresh.jsonl")
    report.add_argument("--traces", default="traces")
    report.add_argument("--out", default=DEFAULT_REPORT_OUT)
    run = sub.add_parser("benchmark", help="Реальный платный эксперимент, нужен ключ")
    run.add_argument("--data", default="data/fresh.jsonl")
    run.add_argument("--traces", default="traces")
    run.add_argument("--out", default=DEFAULT_REPORT_OUT)
    run.add_argument("--configs", nargs="+", choices=list(LABELS), default=REQUIRED_CONFIGS)
    run.add_argument("--bonuses", action="store_true", help="Исходные семь конфигураций; официальный поиск включается отдельно через --configs cheap_official_sources")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--limit", type=int, help="Только smoke; используйте отдельные --traces и --out")
    run.add_argument("--budget-usd", type=float)
    run.add_argument("--max-steps", type=int)
    ask = sub.add_parser("ask", help="Один вопрос агенту")
    ask.add_argument("question")
    ask.add_argument("--model")
    ask_tools = ask.add_mutually_exclusive_group()
    ask_tools.add_argument("--extra-search", action="store_true")
    ask_tools.add_argument("--official-sources", action="store_true")
    ask_tools.add_argument("--knowledge-base", action="store_true",
                           help="Дать агенту только локальный поиск ДЗ‑2. Ответ модели использует обычный API.")
    ask_tools.add_argument("--mcp", action="store_true", help="Поиск ДЗ 4 через Docker MCP. По умолчанию используется заглушка.")
    ask_tools.add_argument("--no-tools", action="store_true", help="Один платный запрос к модели без инструментов.")
    ask.add_argument("--openrouter", action="store_true", help="Платная модель OpenRouter с --mcp или --no-tools")
    ask.add_argument("--output")
    ask.add_argument("--budget-usd", metavar="USD", help="Денежный лимит. С MCP-заглушкой расходы условные.")
    ask.add_argument("--ledger", type=Path, help="Файл учёта расходов для --mcp или --no-tools.")
    args = parser.parse_args(argv)
    try:
        if args.command in {"report", "benchmark"}:
            validate_report_output(args.out)
        if args.command == "ask":
            if args.no_tools and not args.openrouter:
                ask.error("Для --no-tools добавьте --openrouter.")
            if args.mcp or args.no_tools:
                return run_hw4_ask(args, ask)
            if args.ledger is not None:
                ask.error("--ledger доступен с --mcp или --no-tools.")
            if args.openrouter:
                ask.error("--openrouter доступен с --mcp или --no-tools.")
        if args.command == "ask" and args.budget_usd is not None:
            args.budget_usd = float(args.budget_usd)
        # Offline validation, schema inspection and reports never read credentials.
        if args.command in {"ask", "benchmark"}:
            load_dotenv(Path.cwd() / ".env")
            if args.budget_usd is None:
                args.budget_usd = float(os.getenv("BUDGET_USD", "5"))
            if args.command == "benchmark" and args.max_steps is None:
                args.max_steps = int(os.getenv("MAX_STEPS", "6"))
            if args.command == "ask" and args.model is None:
                args.model = configurations()["cheap_search_read"]["model"]
        if args.command == "schemas":
            from .tools import ToolRegistry
            config = "cheap_official_sources" if args.official_sources else (
                "cheap_extra_search" if args.extra_search else "cheap_search_read")
            names = configurations()[config]["tools"]
            registry = ToolRegistry(args.extra_search, official_sources=args.official_sources)
            print(json.dumps(registry.schemas(names), ensure_ascii=False, indent=2))
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
                    registry = ToolRegistry(args.extra_search, official_sources=args.official_sources)
                    config = "cheap_official_sources" if args.official_sources else (
                        "cheap_extra_search" if args.extra_search else "cheap_search_read")
                    tools = configurations()[config]["tools"]
                result = run_agent(args.question, model=args.model, client=client,
                                   registry=registry, tool_names=tools)
            output = args.output if args.output is not None else "tmp/answer_trace.json"
            write_json(output, result)
            print(f"FINAL: {result['final']}\nstatus={result['status']}; trace={output}")
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
            if args.traces == "traces" or args.out in {"reports", DEFAULT_REPORT_OUT}:
                raise ValueError("Для --limit укажите отдельные --traces и --out, чтобы не смешать smoke и полный прогон.")
            rows = rows[:args.limit]
        if args.max_steps < 1:
            raise ValueError("--max-steps должен быть положительным")
        client = get_client(args.budget_usd)
        done = benchmark(rows, trace_dir=args.traces,
                         config_names=BONUS_CONFIGS if args.bonuses else args.configs,
                         client=client, max_steps=args.max_steps, resume=args.resume)
        render_report(rows, args.traces, args.out)
        print(f"{'Готово' if done else 'Прогон остановлен'}; отчёт: {args.out}/report.md")
        return 0 if done else 2
    except (ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
