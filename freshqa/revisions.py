"""Offline P0 re-evaluation of the accepted HW1–3 archive; never resume inference."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
import os
from pathlib import Path

from .dataset import METRIC_VERSION, load_rows, score_row
from .experiment import REQUIRED_CONFIGS, dataset_hash
from .reporting import render_report as report_hw1
from hw2.benchmark_report import render_report as report_hw2
from hw2.benchmark_snapshot import load_snapshot
from hw2.prior_runs import load_prior_runs
from hw2.rag import PARSER_VERSION, validate_response
from hw2.reparse import reparse_record
from hw3.audited_report import apply_overrides
from hw3.evaluation import score as score_hw3
from hw3.report import _load_records, render_report as report_hw3

ROOT = Path(__file__).resolve().parents[1]
BASELINE_COMMIT = '061b823c3ac8286d03c1f10fcffff93d907d1e0c'
# Accepted portable snapshot, independently checked against BASELINE_COMMIT.
# Pin the historical generation protocol here; do not relax benchmark resume checks.
SNAPSHOT_SHA256 = '3886477df16dd8b41394b58fbcbc2b70d93319635e3e840f657a465e283e95eb'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def input_hashes(root):
    paths = [root / p for p in ('data/fresh.jsonl', 'traces/manifest.json', 'reports/report.json',
             'hw2/data/questions.jsonl', 'hw2/data/unanswerable.jsonl', 'hw2/data/corpus.jsonl',
             'hw2/data/chunks_structure.jsonl', 'hw2/results/benchmark/measured.json.gz',
             'hw2/results/benchmark/report.json', 'hw2/results/benchmark/format_diagnostic.json',
             'hw3/data/questions.jsonl', 'hw3/results/main/grading_overrides.json',
             'hw3/results/main/audited/summary.json', 'hw3/results/main/manifest.json')]
    paths += sorted((root / 'traces').glob('*/*.json'))
    from hw3.report import LABELS
    for name in LABELS:
        paths += sorted((root / 'hw3/results/main' / name).glob('*.json'))
    return {str(p.relative_to(root)): sha(p) for p in paths}


def validate_output(root, out):
    root, out = root.resolve(), out.resolve()
    revisions = root / 'reports/revisions'
    if (root.is_relative_to(out) or out == revisions
            or (out.is_relative_to(root) and not out.is_relative_to(revisions))):
        raise ValueError('Revision output must be separate from accepted inputs and reports')
    # Existing symlinks must not redirect individual report writes into source artifacts.
    if out.exists() and any(p.is_symlink() for p in out.rglob('*')):
        raise ValueError('Revision output must not contain symlinks')


def comparison(hw, old, new, cost_key):
    items = []
    for name, current in new.items():
        previous = old[name]
        if previous['n'] != current['n'] or not math.isclose(
                previous[cost_key], current[cost_key], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f'Coverage or spending changed in {hw}/{name}')
        items.append(dict(homework=hw, config=name, n=current['n'], old_correct=previous['correct'],
                          new_correct=current['correct'], old_cost_usd=previous[cost_key],
                          new_cost_usd=current[cost_key]))
    return items


def parser_replay(records, diagnostic, chunks, gold):
    assessments = [reparse_record(r) for r in records if r['config'] in {'cheap_rag', 'medium_rag'}]
    by_id = {c['id']: c for c in chunks}
    samples = []
    for sample in diagnostic['samples']:
        # Only content was saved for these separate calls, not role/finish_reason.
        # The synthetic envelope allows content/span checking; it proves no envelope properties.
        result = validate_response({'message': {'role': 'assistant', 'content': sample['raw_content']},
                                    **sample['usage']}, [by_id[qid] for qid in sample['retrieved_ids']])
        samples.append(dict(config=sample['config'], id=diagnostic['id'],
                            separate_from_benchmark=True, envelope_verified=False,
                            status=result['status'], format_valid=result['format_valid'],
                            validation_error=result['validation_error'], answer=result['answer'],
                            fact_correct=score_row(gold[diagnostic['id']], result['answer']),
                            historical_usage=sample['usage']))
    return dict(parser_version=PARSER_VERSION, full_benchmark_reparsed=all(
                    a['replay_status'] in {'reparsed', 'no_model_call'} for a in assessments),
                coverage={name: dict(Counter(a['replay_status'] for a in assessments if a['config'] == name))
                          for name in ('cheap_rag', 'medium_rag')},
                records=assessments, separate_diagnostic=samples)


def build_revision(root=ROOT, out=ROOT / 'reports/revisions/2026-11-15'):
    root, out = Path(root).resolve(), Path(out).resolve()
    validate_output(root, out)
    before = input_hashes(root)
    rows1 = load_rows(root / 'data/fresh.jsonl')
    gold1 = {r['id']: r for r in rows1}
    summary1, _ = report_hw1(rows1, root / 'traces', out / 'hw1')
    # Keep new text artifacts consistent with the repository's LF convention.
    csv_path = out / 'hw1/report.csv'
    csv_path.write_text(csv_path.read_text(encoding='utf-8'), encoding='utf-8', newline='\n')
    grades1 = []
    for config in REQUIRED_CONFIGS:
        for path in sorted((root / 'traces' / config).glob('*.json')):
            r = read(path)
            grades1.append(dict(config=config, id=r['id'], metric_version=METRIC_VERSION,
                                old_correct=r['correct'], correct=score_row(gold1[r['id']], r['final']),
                                final=r['final'], expected=gold1[r['id']]['answer'], cost_usd=r['cost_usd']))
    write(out / 'hw1/grades.json', grades1)
    write(out / 'hw1/changed_grades.json', [r for r in grades1 if r['old_correct'] != r['correct']])
    subgroups = {}
    for config in REQUIRED_CONFIGS:
        subgroups[config] = {}
        for group, custom in [('original_125', False), ('custom_20', True)]:
            items = [r for r in grades1 if r['config'] == config
                     and (int(r['id'].split('-')[1]) >= 125) == custom]
            subgroups[config][group] = {'n': len(items), 'correct': sum(r['correct'] for r in items)}
    write(out / 'hw1/subgroups.json', subgroups)

    rows2 = load_rows(root / 'hw2/data/questions.jsonl') + load_rows(root / 'hw2/data/unanswerable.jsonl')
    snapshot_path = root / 'hw2/results/benchmark/measured.json.gz'
    if sha(snapshot_path) != SNAPSHOT_SHA256:
        raise ValueError('Not the accepted HW2 snapshot; verify its historical protocol separately')
    with gzip.open(snapshot_path, 'rt', encoding='utf-8') as stream:
        signature2 = json.load(stream)['signature']
    if signature2['dataset_sha256'] != dataset_hash(rows2) or signature2['protocol_sha256'][
            'hw2/data/corpus.jsonl'] != sha(root / 'hw2/data/corpus.jsonl'):
        raise ValueError('HW2 frozen questions or corpus changed')
    prior = load_prior_runs(rows2, trace_dir=root / 'traces', dataset=root / 'data/fresh.jsonl')
    records2 = load_snapshot(snapshot_path, signature2, rows2, prior)
    chunks = load_rows(root / 'hw2/data/chunks_structure.jsonl')
    summary2 = report_hw2(rows2, records2, out / 'hw2', chunks=chunks)
    old2 = read(root / 'hw2/results/benchmark/report.json')
    old_outcomes = {(c, r['id']): r['outcome'] for c, items in old2['questions'].items() for r in items}
    grades2 = [dict(r, config=c, metric_version=METRIC_VERSION, old_outcome=old_outcomes[c, r['id']],
                    parser_assessment='historical_status_retained' if c.endswith('_rag') else 'not_applicable')
               for c, items in summary2['questions'].items() for r in items]
    write(out / 'hw2/grades.json', grades2)
    write(out / 'hw2/changed_grades.json', [r for r in grades2 if r['old_outcome'] != r['outcome']])
    replay = parser_replay(records2, read(root / 'hw2/results/benchmark/format_diagnostic.json'),
                           chunks, {r['id']: r for r in rows2})
    write(out / 'hw2/parser_replay.json', replay)

    result3 = root / 'hw3/results/main'
    rows3 = apply_overrides(load_rows(root / 'hw3/data/questions.jsonl'), read(result3 / 'grading_overrides.json'))
    gold3 = {r['id']: r for r in rows3}
    (out / 'hw3').mkdir(parents=True, exist_ok=True)
    audit_link = os.path.relpath(result3 / 'manual_audit.md', out / 'hw3')
    (out / 'hw3/manual_audit.md').write_text(
        f'# Принятая ручная проверка\n\n[Исходный аудит]({audit_link}). Эталон исправлен одинаково во всех режимах.\n', encoding='utf-8')
    summary3 = report_hw3(result3, rows3, out / 'hw3')
    old3 = {r['config']: r for r in read(result3 / 'audited/summary.json')}
    grades3 = []
    for config, items in _load_records(result3, gold3).items():
        for r in items:
            grade = dict(config=config, id=r['id'], metric_version=METRIC_VERSION,
                         evaluator='hw3_guards_plus_' + METRIC_VERSION, final=r['final'],
                         correct=r['correct'],
                         cost_usd=r['cost_usd'])
            if config == 'critic':
                grade.update(initial_answer=r['initial_answer'], initial_correct=score_hw3(gold3[r['id']], r['initial_answer']))
            grades3.append(grade)
    write(out / 'hw3/grades.json', grades3)
    # The accepted audited report has totals, not all per-answer grades. Its capped
    # failure list cannot reconstruct them. Compare totals explicitly, without guessing.
    changes3 = [r['config'] for r in summary3 if old3[r['config']]['correct'] != r['correct']]

    compared = comparison('hw1', read(root / 'reports/report.json'),
                          {c: summary1[c] for c in REQUIRED_CONFIGS}, 'cost_total')
    compared += comparison('hw2', old2['configs'], summary2['configs'], 'cost_total')
    compared += comparison('hw3', old3, {r['config']: r for r in summary3}, 'cost_usd')
    notes = {
        'hw1': 'Все 725 ответов оценены заново; исходные ответы и списания сохранены.',
        'hw2': 'Это пересчёт фактов по сохранённым финальным ответам. Исторические статусы RAG '
               'сохранены: у всех 228 RAG-записей отсутствует raw-ответ. Таблица НЕ является замером '
               'парсера v2 и не измеряет качество Haiku после исправления. '
               'Отдельная диагностика и покрытие восстановления: [parser_replay.json](parser_replay.json).',
        'hw3': 'Использован принятый исправленный эталон и сохранены дополнительные guards ДЗ 3. '
               'Все пять итогов совпали с принятым аудированным отчётом. '
               'Разбор критика: [вердикты и источники](../critic_audit.md).',
    }
    for hw, note in notes.items():
        report = out / hw / 'report.md'
        title, body = report.read_text(encoding='utf-8').split('\n', 1)
        report.write_text(f'{title}\n\n> Офлайн-ревизия, метрика `{METRIC_VERSION}`. '
                          f'Новых вызовов и расходов нет. {note}\n' + body, encoding='utf-8')
    if before != input_hashes(root):
        raise ValueError('Accepted inputs changed during offline re-evaluation')
    summary = dict(metric_version=METRIC_VERSION, parser_version=PARSER_VERSION, new_model_calls=0,
                   new_cost_usd=0, input_hashes_verified=True, comparison=compared,
                   hw3_changed_config_totals=len(changes3), full_rag_benchmark_reparsed=replay['full_benchmark_reparsed'])
    write(out / 'summary.json', summary)
    code = ['freshqa/dataset.py', 'freshqa/quantities.py', 'freshqa/revisions.py', 'freshqa/reporting.py',
            'hw2/rag.py', 'hw2/reparse.py', 'hw2/benchmark_snapshot.py', 'hw2/benchmark_report.py',
            'hw2/memory_agent.py', 'hw3/evaluation.py', 'hw3/report.py',
            'hw3/audited_report.py', 'hw2/prior_runs.py', 'freshqa/experiment.py']
    write(out / 'provenance.json', dict(baseline_commit=BASELINE_COMMIT, metric_version=METRIC_VERSION,
          parser_version=PARSER_VERSION, generation_signatures={'hw1': read(root / 'traces/manifest.json')['signature'],
                                                               'hw2': signature2,
                                                               'hw3': read(result3 / 'manifest.json')},
          input_sha256=before, evaluation_code_sha256={p: sha(root / p) for p in code},
          new_model_calls=0, new_cost_usd=0, historical_rag_statuses_retained=True))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'reports/revisions/2026-11-15')
    args = parser.parse_args()
    summary = build_revision(ROOT, args.out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
