"""Small live Groq diagnostic, separate from the required Sonnet experiment."""
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from .agent import run_agent
from .dataset import load_rows, score_row, validate_rows
from .experiment import write_json
from .tools import ToolRegistry

PROFILES = {
    "cheap_no_tools": ("openai/gpt-oss-20b", []),
    "cheap_search_read": ("openai/gpt-oss-20b", ["web_search", "page_find", "calculator", "python_exec"]),
    "medium_search_read": ("openai/gpt-oss-120b", ["web_search", "page_find", "calculator", "python_exec"]),
}


def run_check(rows, *, client, out_dir, max_steps=4):
    from .groq_client import PRICING, PRICING_SOURCE, PRICING_VERIFIED_AT, REQUEST_SETTINGS

    out = Path(out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Каталог диагностики не пуст. Выберите новый --out; платные вызовы автоматически не повторяются.")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "manifest.json", {
        "experiment_kind": "diagnostic_sample", "provider": "groq", "question_ids": [r["id"] for r in rows],
        "profiles": PROFILES, "max_steps": max_steps, "request_settings": REQUEST_SETTINGS,
        "pricing": PRICING, "pricing_source": PRICING_SOURCE, "pricing_verified_at": PRICING_VERIFIED_AT,
        "cost_basis": "list_price_estimate", "actual_billed_cost_known": False,
        "started_at": datetime.now(timezone.utc).isoformat(),
    })
    results = []
    stopped = False
    for profile, (model, tools) in PROFILES.items():
        for row in rows:
            inflight = out / "inflight.json"
            write_json(inflight, {"profile": profile, "id": row["id"],
                                 "known_estimate_usd": client.budget_used_usd})
            result = run_agent(row["question"], model=model, client=client,
                               registry=ToolRegistry(), tool_names=tools, max_steps=max_steps)
            result.update(id=row["id"], question=row["question"], expected=row["answer"],
                          correct=score_row(row, result["final"]), profile=profile, model=model,
                          experiment_kind="diagnostic_sample", measured=True)
            write_json(out / "traces" / f"{profile}-{row['id']}.json", result)
            inflight.unlink()
            results.append(result)
            _summary(out, results)
            print(f"{profile}/{row['id']}: {result['status']}; correct={result['correct']}; "
                  f"steps={result['steps']}; estimate=${client.budget_used_usd:.6f}", flush=True)
            if result["status"] in {"error", "cost_unknown", "budget_exceeded"}:
                stopped = True
                break
        if stopped:
            break
    return results


def _summary(out, results):
    summary = {
        "experiment_kind": "diagnostic_sample", "provider": "groq", "attempted": len(results),
        "sonnet_freshness_checked": False, "cost_usd": None,
        "known_estimated_cost_usd": sum(r.get("known_estimated_cost_usd", 0) for r in results),
        "cost_basis": "list_price_estimate", "results": [
            {k: r.get(k) for k in ("id", "profile", "model", "final", "expected", "correct", "status", "steps",
                                  "elapsed_s", "cost_usd", "estimated_cost_usd")} for r in results],
    }
    write_json(out / "summary.json", summary)
    lines = ["# Groq: диагностический запуск", "",
             "Это небольшой реальный запуск для проверки интеграции, не итоговый замер на 145 вопросах.",
             "Sonnet 4.6 через Groq недоступен; обязательный критерий свежести здесь не проверяется.",
             "Цена ниже — оценка по опубликованным токенным тарифам без скидок, а не списание со счёта.", "",
             "| Профиль | Вопрос | Ответ | Эталон | Верно | Шаги | Оценка USD | Статус |",
             "|---|---|---|---|---|---:|---:|---|"]
    for r in results:
        def cell(value):
            return str(value).replace("|", "\\|").replace("\n", " ")[:180]
        estimate = r.get("estimated_cost_usd")
        price = "неизвестна" if estimate is None else f"{estimate:.6f}"
        trace = f"traces/{r['profile']}-{r['id']}.json"
        lines.append(f"| {r['profile']} | [{r['id']}]({trace}) | {cell(r['final'])} | {cell(r['expected'])} | "
                     f"{'да' if r['correct'] else 'нет'} | {r['steps']} | {price} | {r['status']} |")
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    load_dotenv(Path.cwd() / ".env")
    parser = argparse.ArgumentParser(description="Проверить Groq отдельно от обязательного эксперимента")
    parser.add_argument("--data", default="data/fresh.jsonl")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--out", default="reports/groq-check")
    parser.add_argument("--budget-usd", type=float, default=.10)
    parser.add_argument("--max-steps", type=int, default=4)
    args = parser.parse_args(argv)
    try:
        if not 1 <= args.limit <= 10 or not 1 <= args.max_steps <= 8:
            raise ValueError("Для диагностики --limit должен быть 1–10, --max-steps — 1–8.")
        rows = load_rows(args.data)
        errors = validate_rows(rows)
        if errors:
            raise ValueError("\n".join(errors))
        from .groq_client import GroqClient
        client = GroqClient(os.getenv("GROQ_API_KEY", ""), budget_usd=args.budget_usd)
        results = run_check(rows[:args.limit], client=client, out_dir=args.out, max_steps=args.max_steps)
        print(f"Диагностика: {args.out}/README.md; фактическое списание API не сообщает.")
        return 2 if any(r["status"] in {"error", "cost_unknown", "budget_exceeded"} for r in results) else 0
    except (ValueError, OSError) as error:
        print(str(error))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
