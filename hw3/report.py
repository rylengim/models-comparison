"""Rebuild the HW3 tables and chart from saved HTTP-accounted result records."""
from __future__ import annotations

import csv
import json
import math
import os
from collections import Counter
from pathlib import Path

from .evaluation import score

LABELS = {
    'bare': 'Голый цикл',
    'langgraph': 'LangGraph',
    'critic': 'Критик с опорой',
    'plan': 'План + исполнитель',
    'strong_direct': 'Сильная модель без инструментов',
}
_FAILURES = {'error', 'cost_unknown', 'budget_exceeded', 'pending', 'output_truncated'}


def _valid_number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _sum_known(items, field):
    values = [item.get(field) for item in items]
    return sum(values) if values and all(_valid_number(v) for v in values) else None


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _load_records(root, gold):
    grouped = {name: [] for name in LABELS}
    request_ids = set()
    for config in LABELS:
        for path in sorted((root / config).glob('*.json')):
            record = json.loads(path.read_text(encoding='utf-8'))
            qid = record.get('id')
            if qid not in gold or record.get('config') != config or path.stem != qid:
                raise ValueError(f'Record identity/config mismatch: {path.name}')
            if record.get('question') != gold[qid].get('question'):
                raise ValueError(f'Record question mismatch: {config}/{qid}')
            if record.get('measured') is not True:
                raise ValueError(f'Unmeasured record cannot enter report: {config}/{qid}')
            calls = record.get('call_records')
            if not isinstance(calls, list):
                raise ValueError(f'call_records must be a list: {config}/{qid}')
            for call in calls:
                if not isinstance(call, dict):
                    raise ValueError(f'Invalid HTTP ledger entry: {config}/{qid}')
                if call.get('config') != config or call.get('question_id') != qid:
                    raise ValueError(f'HTTP call scope mismatch: {config}/{qid}')
                request_id = call.get('request_id')
                if request_id:
                    if request_id in request_ids:
                        raise ValueError(f'Duplicate request_id in records: {config}/{qid}')
                    request_ids.add(request_id)
            for field in ('cost_usd', 'prompt_tokens', 'completion_tokens'):
                ledger_total = _sum_known(calls, field) if calls else 0
                if ledger_total is None:
                    record[field] = None
                elif not _valid_number(record.get(field)) or not math.isclose(
                    ledger_total, record[field], abs_tol=1e-10, rel_tol=1e-7
                ):
                    label = 'cost' if field == 'cost_usd' else field
                    raise ValueError(f'HTTP ledger {label} mismatch: {config}/{qid}')
                else:
                    record[field] = ledger_total
            if record.get('steps') != len(calls):
                raise ValueError(f'HTTP ledger calls mismatch: {config}/{qid}')
            record['steps'] = len(calls)
            record['correct'] = score(gold[qid], record.get('final'))
            grouped[config].append(record)
    return grouped


def _summarize(config, records, gold):
    n, total = len(records), len(gold)
    correct = sum(record['correct'] for record in records)
    cost = _sum_known(records, 'cost_usd')
    prompt = _sum_known(records, 'prompt_tokens')
    completion = _sum_known(records, 'completion_tokens')
    calls = _sum_known(records, 'steps')
    complete = (n == total and n > 0 and
                not any(record.get('status') in _FAILURES for record in records))
    summary = dict(
        config=config, label=LABELS[config], n=n, total=total, complete=complete,
        correct=correct, accuracy=correct / n if n else None,
        calls_per_question=calls / n if n and calls is not None else None,
        prompt_tokens_per_question=prompt / n if n and prompt is not None else None,
        completion_tokens_per_question=completion / n if n and completion is not None else None,
        tokens_per_question=(prompt + completion) / n if n and prompt is not None and completion is not None else None,
        cents_per_question=100 * cost / n if n and cost is not None else None,
        cents_per_correct=100 * cost / correct if correct and cost is not None else None,
        cost_usd=cost,
        status_counts=dict(Counter(record.get('status', 'missing') for record in records)),
    )
    if config == 'critic':
        pairs = [r for r in records if isinstance(r.get('initial_answer'), str)]
        summary.update(
            paired_n=len(pairs),
            fixed=sum(not score(gold[r['id']], r['initial_answer']) and r['correct'] for r in pairs),
            broken=sum(score(gold[r['id']], r['initial_answer']) and not r['correct'] for r in pairs),
        )
    return summary


def _chart(summary, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = ['#5436A3', '#00838F', '#C43C3C', '#2471A3', '#F09000']
    fig, ax = plt.subplots(figsize=(9, 5.5), layout='constrained')
    visible = 0
    for row, color in zip(summary, colors):
        if row['complete'] and row['cents_per_question'] is not None:
            ax.scatter(row['cents_per_question'], 100 * row['accuracy'], s=100,
                       c=color, label=f"{row['label']} ({row['correct']}/{row['n']})", edgecolors='white')
            visible += 1
    ax.set(xlabel='Центов на вопрос', ylabel='Доля верных ответов, %', ylim=(-3, 103),
           title='ДЗ 3: фактическая цена и качество')
    ax.grid(alpha=.2)
    if visible:
        ax.legend(loc='best', fontsize=9)
    else:
        ax.text(.5, .5, 'Пока нет полных замеров с известной стоимостью',
                transform=ax.transAxes, ha='center', va='center')
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _fmt(value, percent=False):
    if value is None:
        return '—'
    return f'{value:.1%}' if percent else f'{value:.4f}'.rstrip('0').rstrip('.')


def render_report(result_dir: Path, rows: list[dict], output_dir: Path) -> list[dict]:
    """Regrade saved results without modifying them; never invent missing costs."""
    root, out = Path(result_dir), Path(output_dir)
    gold = {row['id']: row for row in rows}
    if not rows or len(gold) != len(rows):
        raise ValueError('Expected nonempty rows with unique IDs')
    grouped = _load_records(root, gold)
    summary = [_summarize(name, records, gold) for name, records in grouped.items()]
    out.mkdir(parents=True, exist_ok=True)
    _write_json(out / 'summary.json', summary)
    fields = list(dict.fromkeys(key for row in summary for key in row))
    with (out / 'summary.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    candidates = []
    for config, records in grouped.items():
        for record in [r for r in records if not r['correct']][:10]:
            row = gold[record['id']]
            candidates.append(dict(config=config, id=row['id'], question=row.get('question'),
                                   answer=record.get('final'), expected=row['answer'],
                                   answer_aliases=row.get('answer_aliases', []), evidence=row.get('evidence'),
                                   url=row.get('url'), status=record.get('status'), audit_status='pending'))
    _write_json(out / 'audit_candidates.json', candidates)
    _chart(summary, out / 'cost_quality.png')
    lines = [
        '# ДЗ 3: результаты измерений', '',
        'Таблица пересчитана из сохранённых ответов и HTTP-учёта. Цена включает неверные ответы. '
        '`—` означает отсутствие данных или неопределённую метрику; цена верного ответа при нуле верных не равна нулю.', '',
        '| Конфигурация | Вопросов | Верных | Доля верных | Вызовов/вопрос | Токенов/вопрос | ¢/вопрос | ¢/верный | Состояние |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|',
    ]
    for row in summary:
        lines.append(f"| {row['label']} | {row['n']}/{row['total']} | {row['correct']} | "
                     f"{_fmt(row['accuracy'], True)} | {_fmt(row['calls_per_question'])} | "
                     f"{_fmt(row['tokens_per_question'])} | {_fmt(row['cents_per_question'])} | "
                     f"{_fmt(row['cents_per_correct'])} | {'полный' if row['complete'] else 'неполный'} |")
    lines += ['', 'Неполные конфигурации не участвуют в итоговом сравнении и не наносятся на график. '
              'Входные и выходные токены отдельно сохранены в `summary.csv`.', '',
              '![Цена вопроса против доли верных](cost_quality.png)', '',
              '## Критик: исправил и испортил', '']
    critic = next(row for row in summary if row['config'] == 'critic')
    lines += [f"Пар с сохранённым исходным ответом: {critic['paired_n']}. "
              f"Исправлено: **{critic['fixed']}**, испорчено: **{critic['broken']}**. "
              'Сравниваются исходный и окончательный ответы одного прогона критика. '
              'Независимый запуск базовой конфигурации не используется для этого счёта.', '',
              '## Проверка проверки', '']
    if (out / 'manual_audit.md').is_file():
        lines += ['Результат ручной проверки: [полные ответы, вердикты и правки](manual_audit.md). '
                  'После добавления подтверждённых алиасов отчёт пересчитывается из исходных ответов.']
    else:
        lines += ['**Ручная проверка ещё не выполнена.** В `audit_candidates.json` сохранены до десяти '
                  'незачтённых полных ответов каждой конфигурации с эталоном и опорой. '
                  'Нужно прочитать их, проверить спорные источники и записать решения в `manual_audit.md`.']
    lines += ['', 'Оценщик наследует проверки дат, чисел и явных алиасов из ДЗ 1; '
              'нормализует диакритику и отдельные числительные zero–twenty, допускает число без слова people в вопросе how many; '
              'отвергает явное отрицание и неразрешённые альтернативы. Это консервативная эвристика, '
              'а не понимание смысла; неоднозначные ответы требуют ручного аудита.', '',
              '## Первый запрос и программный лимит', '']
    for filename, label in [('framework_probe.json', 'Побайтовое сравнение первого запроса'),
                            ('limit_demo.json', 'Демонстрация остановки по лимиту')]:
        path = root / filename
        if path.exists():
            relative = os.path.relpath(path, out)
            lines.append(f'- [{label}]({relative}).')
        else:
            lines.append(f'- {label}: результат ещё не сохранён.')
    lines += ['', '## Выводы', '']
    if not all(row['complete'] and row['cost_usd'] is not None for row in summary):
        lines += ['Пять полных сопоставимых конфигураций с известной стоимостью ещё не собраны. '
                  'Вывод о пользе и окупаемости паттернов пока не делается.']
    else:
        baselines = summary[:2]
        best = max(baselines, key=lambda row: (row['accuracy'], -row['cents_per_question']))
        lines += [f"Среди двух исполнений лучшая доля верных у «{best['label']}»: "
                  f"{best['correct']}/{best['n']} при {_fmt(best['cents_per_question'])} ¢/вопрос. "
                  'При равной точности выбран более дешёвый вариант.']
        for row in summary[2:4]:
            delta = row['correct'] - best['correct']
            ratio = row['cents_per_question'] / best['cents_per_question'] if best['cents_per_question'] else None
            lines.append(f"«{row['label']}»: изменение числа верных {delta:+d}; "
                         f"цена вопроса {_fmt(ratio)}× от выбранной базы. "
                         + ('Прироста качества на этом наборе не получено.' if delta <= 0 else
                            'Есть наблюдаемый прирост; цену одного верного ответа следует сравнивать по таблице.'))
        lines += ['На 40 вопросах один ответ меняет долю верных на 2,5 п.п.; '
                  'один прогон не доказывает статистическую устойчивость различий.']
    (out / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return summary
