"""Build reports only from saved, explicitly measured traces."""
import csv
import json
from pathlib import Path

from .dataset import is_correct, score_row
from .experiment import LABELS, REQUIRED_CONFIGS, aggregate, configurations, dataset_hash, write_json


def freshness(traces, total, model):
    finished = len(traces) == total and total > 0 and all(t["status"] == "ok" for t in traces)
    sonnet = model == "anthropic/claude-sonnet-4.6"
    right = sum(t["correct"] for t in traces)
    return {"complete": finished, "required_model": sonnet, "n": len(traces), "total": total,
            "correct": right, "accuracy": right / len(traces) if traces else None,
            "passed": right / total <= .2 if finished and sonnet else None,
            "guessed_ids": [t["id"] for t in traces if t["correct"]]}


def render_report(rows, trace_dir="traces", out_dir="reports"):
    root, out = Path(trace_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    configs = configurations()
    manifest = root / "manifest.json"
    if manifest.exists():
        meta = json.loads(manifest.read_text(encoding="utf-8"))
        if meta["signature"]["dataset_sha256"] != dataset_hash(rows):
            raise ValueError("Хэш датасета не совпадает с трейсами; нельзя смешивать замеры.")
        configs = meta["signature"]["configs"]
    gold = {r["id"]: r for r in rows}
    grouped = {name: [] for name in LABELS}
    trace_paths = {}
    for path in sorted(root.glob("*/*.json")):
        if not manifest.exists():
            raise ValueError("Есть трейсы без manifest.json")
        trace = json.loads(path.read_text(encoding="utf-8"))
        if trace.get("measured") is not True or trace.get("id") not in gold:
            raise ValueError(f"Посторонний или тестовый трейс: {path}")
        if trace["config"] != path.parent.name or trace["id"] != path.stem:
            raise ValueError(f"Трейс не соответствует пути: {path}")
        trace["correct"] = score_row(gold[trace["id"]], trace["final"])
        grouped[trace["config"]].append(trace)
        trace_paths[(trace["config"], trace["id"])] = path
    summary = {name: aggregate(items, len(rows)) for name, items in grouped.items()}
    fresh = freshness(grouped["strong_no_tools"], len(rows), configs["strong_no_tools"]["model"])
    write_json(out / "report.json", summary)
    write_json(out / "freshness.json", fresh)
    with (out / "report.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["config", "model", *next(iter(summary.values())).keys()])
        writer.writeheader()
        for name, values in summary.items():
            writer.writerow({"config": name, "model": configs[name]["model"], **values})

    def fmt(value, percent=False):
        return "—" if value is None else (f"{value:.1%}" if percent else f"{value:.4f}")

    lines = ["# Результаты эксперимента", "",
             "Числа рассчитываются только из реальных трейсов. `—` означает отсутствие замера, а не ноль.", "",
             "| Конфигурация | Завершено | Верных | $/задачу | $/верный | Шагов | Секунд |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, values in summary.items():
        lines.append(f"| {LABELS[name]} | {values['n']}/{len(rows)} | {fmt(values['accuracy'], True)} | "
                     f"{fmt(values['cost_per_task'])} | {fmt(values['cost_per_correct'])} | "
                     f"{fmt(values['mean_steps'])} | {fmt(values['mean_seconds'])} |")
    lines += ["", "Незавершённые строки описывают только обработанные вопросы и не подходят для итогового сравнения.",
              "При нуле верных $/верный не определена. При неизвестном списании цена не рассчитывается.",
              "", "## Проверка свежести", "",
              "| Модель | Проверено | Верных | Критерий ≤20% |", "|---|---:|---:|---|",
              f"| {configs['strong_no_tools']['model']} | {fresh['n']}/{len(rows)} | {fmt(fresh['accuracy'], True)} | "
              + ("пройден" if fresh["passed"] is True else "НЕ пройден" if fresh["passed"] is False else "не проверен") + " |",
              "", f"Угаданные id: {', '.join(fresh['guessed_ids']) or 'нет результатов / нет угаданных'}.", "",
              "![Цена и качество](money_chart.png)", "", "## Вывод", ""]
    if not all(summary[name]["complete"] for name in REQUIRED_CONFIGS):
        lines += ["Полного прогона пяти обязательных конфигураций пока нет. Тезис курса не проверен; "
                  "вывод о превосходстве дешёвого агента делать нельзя. Нужен OPENROUTER_API_KEY и полный запуск."]
    else:
        strong, cheap = summary["strong_no_tools"], summary["cheap_search_read"]
        if strong["cost_per_task"] is None or cheap["cost_per_task"] is None:
            lines += ["Все вопросы обработаны, но стоимость части запросов неизвестна. Сравнение цены требует сверки списаний."]
        else:
            wins = cheap["accuracy"] >= strong["accuracy"] and cheap["cost_per_task"] < strong["cost_per_task"]
            lines += [f"На этом прогоне тезис {'подтвердился' if wins else 'не подтвердился'} по критерию "
                      "«не ниже точность и ниже средняя цена задачи». "
                      f"Дешёвый агент: {cheap['accuracy']:.1%}, ${cheap['cost_per_task']:.4f}/задачу; "
                      f"сильная без инструментов: {strong['accuracy']:.1%}, ${strong['cost_per_task']:.4f}/задачу. "
                      "Один прогон не измеряет статистическую устойчивость; причины различий нужно сверить по трейсaм."]
    text = "\n".join(lines) + "\n"
    (out / "report.md").write_text(text, encoding="utf-8")
    _chart(summary, out)
    _failure_candidates(grouped, trace_paths, out)
    readme = Path("README.md")
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    if readme.exists() and out.resolve() == Path("reports").resolve():
        original = readme.read_text(encoding="utf-8")
        if start in original and end in original:
            embedded = "\n".join("#" + line if line.startswith("#") else line for line in text.splitlines())
            embedded = embedded.replace("](money_chart.png)", "](reports/money_chart.png)")
            readme.write_text(original.split(start, 1)[0] + start + "\n\n" + embedded + "\n\n" + end
                              + original.split(end, 1)[1], encoding="utf-8")
    return summary, fresh


def _chart(summary, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    measured = [(k, v) for k, v in summary.items() if v["complete"] and v["cost_per_task"] is not None]
    ax.set(xlabel="Средняя стоимость задачи, USD", ylabel="Доля верных ответов, %", ylim=(-3, 103))
    ax.set_title("Свежие вопросы: цена и качество")
    ax.grid(alpha=.2)
    for name, values in measured:
        ax.scatter(values["cost_per_task"], 100 * values["accuracy"], s=70, label=LABELS[name])
    if measured:
        ax.legend(loc="best", fontsize=8)
    else:
        has_attempts = any(v["n"] for v in summary.values())
        title = "Нет полных замеров с известной ценой" if has_attempts else "Замеры ещё не выполнены"
        detail = ("Есть неполные прогоны или неизвестные списания\nПодробнее в таблице отчёта" if has_attempts
                  else "Нужен API-ключ и полный прогон моделей\nТестовые данные на график не нанесены")
        ax.text(.5, .55, title, transform=ax.transAxes, ha="center", fontsize=18)
        ax.text(.5, .44, detail,
                transform=ax.transAxes, ha="center", fontsize=11)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(out / "money_chart.png", dpi=160)
    plt.close(fig)


def _failure_candidates(grouped, paths, out):
    import os
    failures = [t for name in ("cheap_search_read", "cheap_extra_search", "medium_search_read", "cheap_search")
                for t in grouped[name] if not t["correct"]]
    lines = ["# Кандидаты для разбора провалов", "",
             "Автоматическая предварительная классификация: перед сдачей прочитайте два трейса и уточните причину.", ""]
    if not failures:
        lines += ["Реальных провалов пока нет: измерения не выполнены либо агент не ошибся. "
                  "Два вымышленных случая не подставляются вместо экспериментальных."]
    for trace in failures[:2]:
        tool_events = [e for e in trace["events"] if e.get("type") == "tool"]
        found = any(is_correct(trace["expected"], e.get("output", "")) for e in tool_events)
        reason = ("Эталон встречается в наблюдениях, но итоговый ответ его не содержит."
                  if found else "Эталон не обнаружен в наблюдениях; проверьте запросы, ошибки источников и чтение страницы.")
        relative = os.path.relpath(paths[(trace["config"], trace["id"])], out)
        lines += [f"- [{trace['config']}/{trace['id']}]({relative}): {reason} "
                  f"Ответ: `{trace['final']}`; эталон: `{trace['expected']}`; статус: `{trace['status']}`."]
    (out / "failure_candidates.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
