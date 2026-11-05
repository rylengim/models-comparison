"""Score measured HW2 traces, keeping incomplete coverage and unknown charges visible."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from freshqa.dataset import final_text, load_rows, score_row
from hw2.prepare_corpus import DATA, normalize_evidence


LABELS = {
    "strong_no_tools": "Сильная без инструментов",
    "cheap_live_search": "Дешёвая с живым поиском",
    "cheap_rag": "Дешёвая с обязательным RAG",
    "cheap_agent_kb": "Дешёвый агент с базой",
    "medium_rag": "Средняя с обязательным RAG",
}
AGENT_FINAL = {"ok", "max_steps", "repeated_tool_call", "tool_limit"}
ERRORS = {"error", "cost_unknown", "budget_exceeded", "execution_error"}


def classify(row, record):
    """Independently score the response; a technical fallback is never an explicit refusal."""
    status = record["status"]
    if status in ERRORS:
        return "execution_error"
    if status == "no_hits":
        return "no_hits"
    answer = record["final"]
    if not isinstance(answer, str) or not final_text(answer):
        return "invalid_output"
    if status in AGENT_FINAL:
        last = next((e for e in reversed(record.get("events", [])) if e.get("type") == "llm"), {})
        message = last.get("message", record.get("final_message"))
        reason = last.get("finish_reason", record.get("finish_reason"))
        if (not isinstance(message, dict) or message.get("role") != "assistant" or message.get("tool_calls")
                or reason not in (None, "stop") or not isinstance(message.get("content"), str)
                or final_text(message["content"]) != final_text(answer)):
            return "invalid_output"
    elif status not in {"answered", "insufficient_evidence"}:
        return "invalid_output"
    refusal = final_text(answer).splitlines()[0].strip().rstrip(".!").casefold() in {"unknown", "неизвестно"}
    if status == "insufficient_evidence" and not refusal:
        return "invalid_output"
    if status == "answered" and refusal:
        return "invalid_output"
    if refusal:
        return "false_refusal" if row.get("answerable", True) else "correct_refusal"
    if row.get("answerable", True) and score_row(row, answer):
        return "correct_answer"
    return "wrong_answer"


def load_records(trace_dir):
    records = []
    for path in sorted(Path(trace_dir).glob("*/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if (record.get("config"), record.get("id")) != (path.parent.name, path.stem):
            raise ValueError(f"Имя трейса расходится с его содержимым: {path}")
        records.append(record)
    return records


def summarize(rows, records):
    gold = {row["id"]: row for row in rows}
    if not gold or len(gold) != len(rows):
        raise ValueError("Нужна непустая выборка с уникальными ID.")
    positive_total = sum(bool(row.get("answerable", True)) for row in rows)
    seen = set()
    details = {name: [] for name in LABELS}
    for record in records:
        name, qid = record["config"], record["id"]
        if name not in details or qid not in gold or record["question"] != gold[qid]["question"]:
            raise ValueError(f"Трейс не соответствует выборке: {name}/{qid}")
        if record.get("measured") is not True or (name, qid) in seen:
            raise ValueError(f"Требуется одно реальное измерение для {name}/{qid}")
        seen.add((name, qid))
        price = record.get("cost_usd")
        if price is not None:
            if type(price) not in (int, float) or not math.isfinite(price) or price < 0:
                raise ValueError(f"Некорректная цена {name}/{qid}")
            if record.get("cost_basis") != "provider_reported" and not (
                record["status"] == "no_hits" and price == 0
            ):
                price = None
        if record.get("origin") not in {"hw1", "hw2"}:
            raise ValueError(f"Неизвестное происхождение {name}/{qid}")
        known = price
        if price is None and record.get("cost_basis") == "provider_reported":
            known = record.get("known_cost_usd")
            if known is not None and (type(known) not in (int, float) or not math.isfinite(known) or known < 0):
                raise ValueError(f"Некорректная известная часть расходов {name}/{qid}")
        details[name].append({
            "id": qid, "model": record["model"], "origin": record["origin"],
            "status": record["status"], "final": record["final"], "cost_usd": price, "known_cost_usd": known,
            "answerable": bool(gold[qid].get("answerable", True)), "outcome": classify(gold[qid], record),
        })

    summary = {}
    for name, items in details.items():
        models = {item["model"] for item in items}
        if len(models) > 1:
            raise ValueError(f"Разные модели в конфигурации {name}")
        counts = Counter(item["outcome"] for item in items)
        n = len(items)
        positives = sum(item["answerable"] for item in items)
        empty_positive = sum(i["outcome"] == "no_hits" and i["answerable"] for i in items)
        empty_negative = counts["no_hits"] - empty_positive
        right = counts["correct_answer"] + counts["correct_refusal"] + empty_negative
        priced = [i for i in items if i["cost_usd"] is not None]
        total_cost = math.fsum(i["cost_usd"] for i in priced) if n and len(priced) == n else None
        summary[name] = {
            "model": next(iter(models), None), "n": n, "target": len(rows), "positive_n": positives,
            "negative_n": n - positives, "complete": n == len(rows) and not counts["execution_error"],
            "correct": right, "accuracy": right / n if n else None, "correct_answer": counts["correct_answer"],
            "correct_refusal": counts["correct_refusal"], "false_refusal": counts["false_refusal"],
            "no_hits_positive": empty_positive, "no_hits_negative": empty_negative,
            "invalid_output": counts["invalid_output"], "execution_error": counts["execution_error"],
            "cost_total": total_cost, "cost_per_question": total_cost / n if total_cost is not None else None,
            "cost_per_correct": total_cost / right if total_cost is not None and right else None,
            "unknown_charges": n - len(priced),
            "known_cost_by_origin": {origin: math.fsum(i["known_cost_usd"] for i in items
                                                       if i["origin"] == origin and i["known_cost_usd"] is not None)
                                     for origin in ("hw1", "hw2")},
            "count_by_origin": dict(Counter(i["origin"] for i in items)),
        }
    return {"target": len(rows), "positive_target": positive_total, "negative_target": len(rows) - positive_total,
            "configs": summary, "questions": details}


def _visible_context(record, by_id):
    passages, missing = [], []
    for chunk_id in dict.fromkeys(record.get("retrieved_ids", [])):
        if chunk_id in by_id:
            passages.append(by_id[chunk_id])
        else:
            missing.append(chunk_id)
    for event in record.get("events", []):
        if event.get("type") != "tool":
            continue
        try:
            body = json.loads(event["output"])
        except (KeyError, TypeError, ValueError):
            continue
        if isinstance(body, list) and event.get("name") == "knowledge_base":
            passages.extend(x for x in body if isinstance(x, dict) and "text" in x)
        if isinstance(body, dict):
            passages.extend({"url": r.get("url"), "text": r.get("snippet", "")} for r in body.get("results", []))
            passages.extend({"url": body.get("url"), "text": t} for t in body.get("excerpts", []))
    unresolved = set(missing) - {p.get("id") for p in passages}
    return passages, bool(unresolved)


def failure_examples(report, rows, records, chunks):
    """Use the context actually delivered to the model, never a fresh retrieval."""
    configs = report["configs"]
    if not all(v["complete"] for v in configs.values()):
        return {"best_config": None, "counts": {}, "examples": {}}
    name = min(configs, key=lambda key: (-configs[key]["accuracy"], configs[key]["cost_per_correct"]
                                         if configs[key]["cost_per_correct"] is not None else math.inf))
    by_id = {chunk["id"]: chunk for chunk in chunks}
    gold = {row["id"]: row for row in rows}
    runs = {rec["id"]: rec for rec in records if rec["config"] == name}
    counts, examples = Counter(), {}
    for item in report["questions"][name]:
        if not item["answerable"] or item["outcome"] == "correct_answer":
            continue
        row, rec = gold[item["id"]], runs[item["id"]]
        if rec.get("origin") == "hw1" and name == "cheap_live_search" and Path(row["id"]).name == row["id"]:
            # Resolve only this run's known local baseline. The trace_path field is not a file selector.
            path = DATA.parents[1] / "traces" / "cheap_search_read" / f"{row['id']}.json"
            if path.is_file():
                archived = json.loads(path.read_text(encoding="utf-8"))
                if (archived.get("id"), archived.get("question"), archived.get("final")) == (
                    row["id"], row["question"], rec["final"]
                ):
                    rec = {**rec, "events": archived["events"]}
        passages, unresolved = _visible_context(rec, by_id)
        needle = " " + normalize_evidence(row["evidence"]) + " "
        match = next((p for p in passages if p.get("url") == row["url"]
                      and needle in " " + normalize_evidence(p.get("text", "")) + " "), None)
        if name == "strong_no_tools":
            category = "without_tools"
        elif match:
            category = "answer_failed"
        elif unresolved or (name == "cheap_live_search" and "events" not in rec):
            category = "context_unavailable"
        else:
            category = "evidence_missing"
        counts[category] += 1
        if category not in examples:
            example_passage = match or next(iter(passages), None)
            examples[category] = {"id": row["id"], "question": row["question"], "actual": item["final"],
                                  "expected": row["answer"], "evidence": row["evidence"], "source": row["url"],
                                  "outcome": item["outcome"], "context": example_passage}
    return {"best_config": name, "counts": dict(counts), "examples": examples}


def _money_chart(configs, path):
    from matplotlib.figure import Figure
    from matplotlib.ticker import PercentFormatter

    figure = Figure(figsize=(9, 4.8), layout="constrained")
    ax = figure.subplots()
    points = 0
    for name, result in configs.items():
        if result["cost_per_question"] is None:
            continue
        n, target = result["n"], result["target"]
        label = f"{LABELS[name]} ({n}/{target})"
        ax.scatter(result["cost_per_question"], result["accuracy"], s=80,
                   marker="o" if result["complete"] else "^", label=label)
        points += 1
    ax.set(title="Качество и цена ответа на измеренных вопросах",
           xlabel="Расходы API на вопрос, USD", ylabel="Доля верных результатов", ylim=(-0.04, 1.04))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(alpha=0.2)
    if points:
        ax.legend(loc="best", fontsize=8)
    else:
        ax.text(.5, .5, "Пока нет измерений с известной ценой", transform=ax.transAxes, ha="center")
    figure.savefig(path, dpi=150)


def _ratio(numerator, denominator):
    return f"{numerator}/{denominator}" if denominator else "—"


def _usd(value):
    return f"{value:.6f}" if value is not None else "—"


def render_report(rows, records, output_dir, *, chunks=()):
    records = list(records)
    report = summarize(rows, records)
    report["failures"] = failure_examples(report, rows, records, chunks)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                                     encoding="utf-8")
    _money_chart(report["configs"], out / "money_chart.png")
    lines = ["# Ответы агента и расходы", "",
             f"Вопросов с ответом — {report['positive_target']}, вопросов для проверки отказа — "
             f"{report['negative_target']}.", "",
             "| Конфигурация | Покрытие | Измерено | Верных | Точность | USD/вопрос | USD/верный |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for name, v in report["configs"].items():
        state = "полное" if v["complete"] else ("частичное" if v["n"] else "ожидает запуска")
        accuracy = f"{v['accuracy']:.1%}" if v["accuracy"] is not None else "—"
        lines.append(f"| {LABELS[name]} | {state} | {v['n']}/{v['target']} | {_ratio(v['correct'], v['n'])} | "
                     f"{accuracy} | {_usd(v['cost_per_question'])} | {_usd(v['cost_per_correct'])} |")
    lines += ["", "Для каждой строки знаменатель точности и цены вопроса — число обработанных вопросов. "
              "Технические ошибки и отклонённый формат учитываются в числе обработанных и дают ноль верных. "
              "На вопросах без ответа засчитывается явный отказ или пустая выдача с ответом unknown. "
              "Цена верного результата равна сумме подтверждённых расходов по строке, делённой на число верных. "
              "Прочерк означает отсутствие данных, неизвестную цену либо ноль верных результатов.",
              "", "При разном покрытии цифры относятся к разным подвыборкам. На графике треугольники обозначают "
              "частичные строки, круги — полные.",
              "", "![Цена и качество](money_chart.png)", "", "## Отказы на обработанных вопросах", "",
              "| Конфигурация | Верных ответов на положительных | Отказ модели на отрицательных | "
              "Ложный отказ модели на положительных | Пустая выдача (+ / −) | "
              "Всего ложных отказов системы | Ошибка формата / выполнения |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for name, v in report["configs"].items():
        n = v["positive_n"]
        empty = f"{v['no_hits_positive']} / {v['no_hits_negative']}" if v["n"] else "—"
        errors = f"{v['invalid_output']} / {v['execution_error']}" if v["n"] else "—"
        lines.append(f"| {LABELS[name]} | {_ratio(v['correct_answer'], n)} | "
                     f"{_ratio(v['correct_refusal'], v['negative_n'])} | {_ratio(v['false_refusal'], n)} | "
                     f"{empty} | {_ratio(v['false_refusal'] + v['no_hits_positive'], n)} | {errors} |")
    lines += ["", "## Происхождение расходов", "",
              "| Конфигурация | Архив ДЗ‑1, ответов / USD | Новые результаты ДЗ‑2, ответов / USD | Неизвестных цен |",
              "|---|---:|---:|---:|"]
    for name, v in report["configs"].items():
        cols = []
        for origin in ("hw1", "hw2"):
            n = v["count_by_origin"].get(origin, 0)
            cols.append(f"{n} / {_usd(v['known_cost_by_origin'][origin])}" if n else "—")
        lines.append(f"| {LABELS[name]} | {cols[0]} | {cols[1]} | {v['unknown_charges'] if v['n'] else '—'} |")
    lines += ["", "Архивные расходы учитываются один раз при сравнении конфигураций. "
              "Столбцы происхождения суммируют только известные списания. "
              "При неизвестной цене хотя бы одного вызова средние цены всей строки оставлены пустыми."]
    analysis = report["failures"]
    if analysis["best_config"]:
        lines += ["", "## Ошибки на вопросах с ответом", "",
                  f"Лучший полный результат по точности у «{LABELS[analysis['best_config']]}». "
                  "При равенстве выбран меньший расход на верный результат."]
        titles = {"evidence_missing": "Цитата отсутствовала в контексте", "answer_failed": "Цитата была в контексте",
                  "without_tools": "Режим без инструментов", "context_unavailable": "Контекст недоступен"}
        for key, ex in analysis["examples"].items():
            lines += ["", f"### {titles[key]} ({analysis['counts'][key]} случаев)", "",
                      f"Пример {ex['id']}. {ex['question']}", "",
                      f"Ожидалось **{ex['expected']}**, получено **{ex['actual']}**. Эталонная цитата — "
                      f"«{ex['evidence']}». [Источник]({ex['source']})."]
            context = ex["context"]
            if context:
                snippet = " ".join(context["text"].split())[:550]
                lines.append(f"Фрагмент из выдачи — «{snippet}».")
            if key == "evidence_missing":
                lines.append("Следующая проверка — повторить поиск с именем или датой события и "
                             "сравнить позицию фрагмента с эталонной цитатой.")
            elif key == "answer_failed":
                lines.append("Следующая проверка — дать генератору найденный фрагмент отдельно и "
                             "проверить, совпадают ли предмет вопроса, единицы и даты в ответе и цитате.")
    else:
        lines += ["", "Для выбора лучшей конфигурации и разбора её ошибок нужно завершить все пять строк."]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows(DATA / "questions.jsonl") + load_rows(DATA / "unanswerable.jsonl")
    chunk_file = DATA / "chunks_structure.jsonl"
    render_report(rows, load_records(args.traces), args.out, chunks=load_rows(chunk_file) if chunk_file.exists() else ())


if __name__ == "__main__":
    main()
