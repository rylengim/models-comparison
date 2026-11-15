"""Offline source audit: observed tool failures, final answer state and metric stay separate."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

from .dataset import METRIC_VERSION, final_text, load_rows, score_row
from .experiment import REQUIRED_CONFIGS, dataset_hash

ROOT = Path(__file__).resolve().parents[1]
AUDIT_VERSION = 'source-audit-v1'
DIAGNOSTIC_VERSION = 'source-diagnostics-v1'
DEFAULT_OUT = ROOT / 'reports/revisions/2026-11-15-sources'
STAGES = {'validation', 'search', 'resolve', 'fetch', 'extract', 'control'}
DIAGNOSTIC_KEYS = {'version', 'stage', 'outcome', 'error_kind', 'http_status', 'retryable'}
NETWORK_TOOLS = {'web_search', 'web_search_extra', 'official_search', 'page_find'}
HTTP_ERRORS = {'legacy_http_error', 'http_error'}
TRANSPORT_ERRORS = HTTP_ERRORS | {'legacy_read_timeout', 'legacy_connect_timeout', 'legacy_request_error',
                                  'timeout', 'tls_error', 'connection_error', 'request_error', 'dns_error'}


def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _legacy(stage, outcome, kind=None):
    return dict(version=DIAGNOSTIC_VERSION, stage=stage, outcome=outcome,
                error_kind=kind, http_status=None, retryable=None)


def _recorded_diagnostic(value):
    valid = (isinstance(value, dict) and set(value) == DIAGNOSTIC_KEYS
             and value['version'] == DIAGNOSTIC_VERSION and isinstance(value['stage'], str) and value['stage'] in STAGES
             and isinstance(value['outcome'], str) and value['outcome'] in {'ok', 'empty', 'error'}
             and (value['error_kind'] is None or isinstance(value['error_kind'], str)
                  and bool(value['error_kind'].strip()))
             and (value['http_status'] is None or type(value['http_status']) is int
                  and 100 <= value['http_status'] <= 599)
             and (value['retryable'] is None or type(value['retryable']) is bool))
    if not valid or (value['outcome'] == 'error') != (value['error_kind'] is not None):
        raise ValueError('Malformed recorded source diagnostic')
    if value['outcome'] != 'error' and value['http_status'] is not None and value['http_status'] >= 400:
        raise ValueError('Recorded source diagnostic contradicts its HTTP failure status')
    return dict(value)


def observe_tool(event, event_index):
    """Normalize saved observations without reconstructing unavailable HTTP facts."""
    try:
        payload = json.loads(event['output'])
    except (ValueError, TypeError, KeyError):
        payload = None
    result = dict(event_index=event_index, tool=event['name'], urls=[], diagnostic_origin='legacy_inferred')
    arguments = event.get('arguments')
    if isinstance(arguments, dict) and isinstance(arguments.get('url'), str) and arguments['url']:
        result['urls'].append(dict(origin='argument', url=arguments['url']))
    if not isinstance(payload, dict):
        result['diagnostic'] = _legacy('control', 'error', 'legacy_unparseable_output')
        return result
    if isinstance(payload.get('url'), str) and payload['url']:
        result['urls'].append(dict(origin='page_result', url=payload['url']))
    if isinstance(payload.get('results'), list):
        result['result_count'] = len(payload['results'])
        for hit in payload['results']:
            if isinstance(hit, dict) and isinstance(hit.get('url'), str) and hit['url']:
                result['urls'].append(dict(origin='search_result', url=hit['url']))
    if isinstance(payload.get('excerpts'), list):
        result['excerpt_count'] = len(payload['excerpts'])
    if type(payload.get('keywords_matched')) is bool:
        result['keywords_matched'] = payload['keywords_matched']
    if 'diagnostic' in payload:
        diagnostic = _recorded_diagnostic(payload['diagnostic'])
        if payload.get('error') is not None and diagnostic['outcome'] != 'error':
            raise ValueError('Recorded source diagnostic contradicts its error payload')
        result.update(diagnostic=diagnostic, diagnostic_origin='recorded')
        return result
    error = payload.get('error')
    if payload.get('truncated') is True:
        diagnostic = _legacy('control', 'error', 'output_truncated')
    elif error is not None:
        if isinstance(error, str) and error.startswith('Source request failed:'):
            kind = {'HTTPError': 'legacy_http_error', 'ReadTimeout': 'legacy_read_timeout',
                    'ConnectTimeout': 'legacy_connect_timeout'}.get(error.split(':', 1)[1].strip(),
                                                                   'legacy_request_error')
            diagnostic = _legacy('fetch', 'error', kind)
        elif error == 'Invalid tool arguments' or error == 'invalid_arguments':
            diagnostic = _legacy('validation', 'error', 'invalid_arguments')
        elif isinstance(error, str) and error in {'repeated_tool_call', 'tool_not_allowed', 'tool_limit'}:
            diagnostic = _legacy('control', 'error', error)
        else:
            diagnostic = _legacy('control', 'error', 'legacy_tool_error')
    elif 'results' in payload and isinstance(payload['results'], list):
        diagnostic = _legacy('search', 'ok' if payload['results'] else 'empty')
    elif type(payload.get('found')) is bool:
        diagnostic = _legacy('extract', 'ok' if payload['found'] else 'empty')
    elif event['name'] in NETWORK_TOOLS:
        diagnostic = _legacy('control', 'error', 'legacy_unclassified_output')
    else:
        diagnostic = _legacy('control', 'ok')
    result['diagnostic'] = diagnostic
    return result


def answer_state(row, output):
    if not isinstance(output, str) or not final_text(output):
        return 'missing'
    actual = final_text(output)
    first_line = actual.splitlines()[0].strip()
    refusal = re.match(r'^(?:unknown|неизвестно)(?:\s*$|\s*[^\w\s])', first_line, re.I)
    if refusal:
        # Same exact-name exception as metric v2; a mention in an explanation is insufficient.
        names = [row['answer'], *row.get('answer_aliases', [])]
        exact_name = any(actual.casefold() == str(name).strip().casefold() for name in names)
        if not (exact_name and re.search(r'[^\W\d_]', actual[refusal.end():])):
            return 'abstention'
    return 'answer'


def _summary(records):
    tools, kinds, codes, diagnostic_origins = Counter(), Counter(), Counter(), Counter()
    for record in records:
        tools.update(record['tool_counts'])
        kinds.update(record['technical_issues'])
        codes.update(record['observed_http_statuses'])
        diagnostic_origins.update(record['diagnostic_origins'])
    cost = [r['historical_cost_usd'] for r in records]
    return dict(n=len(records), correct=sum(r['correct'] for r in records),
                answer_states=dict(Counter(r['answer_state'] for r in records)),
                observed_run_statuses=dict(Counter(r['observed_run_status'] for r in records)),
                tool_events=dict(tools), technical_issues=dict(kinds),
                traces_with_technical_issues=sum(r['technical_issue_count'] > 0 for r in records),
                traces_with_http_errors=sum(r['http_error_count'] > 0 for r in records),
                traces_with_transport_errors=sum(r['transport_error_count'] > 0 for r in records),
                observed_http_statuses=dict(codes), diagnostic_origins=dict(diagnostic_origins),
                technical_x_correct=dict(Counter(
                    ('issues_' if r['technical_issue_count'] else 'no_issues_')
                    + ('correct' if r['correct'] else 'incorrect') for r in records)),
                transport_x_correct=dict(Counter(
                    ('issues_' if r['transport_error_count'] else 'no_issues_')
                    + ('correct' if r['correct'] else 'incorrect') for r in records)),
                historical_cost_usd=math.fsum(cost) if all(v is not None for v in cost) else None,
                unknown_historical_charges=sum(v is None for v in cost))


def audit_traces(rows, trace_dir):
    """Read all five required configurations with full identity and coverage checks."""
    trace_dir = Path(trace_dir)
    gold = {r['id']: r for r in rows}
    if not rows or len(gold) != len(rows) or any(not re.fullmatch(r'fresh-\d+', qid) for qid in gold):
        raise ValueError('Expected unique fresh-N rows')
    signature = _read(trace_dir / 'manifest.json').get('signature', {})
    configs = signature.get('configs', {})
    if (signature.get('dataset_sha256') != dataset_hash(rows) or signature.get('n') != len(rows)
            or any(not isinstance(configs.get(c), dict) or not isinstance(configs[c].get('model'), str)
                   for c in REQUIRED_CONFIGS)):
        raise ValueError('Dataset/configuration manifest mismatch')
    records, own_details = [], []
    for config in REQUIRED_CONFIGS:
        paths = sorted((trace_dir / config).glob('*.json'))
        if {p.stem for p in paths} != gold.keys():
            raise ValueError(f'Incomplete or extra trace coverage: {config}')
        for path in paths:
            trace = _read(path)
            row = gold[path.stem]
            if (trace.get('id') != row['id'] or trace.get('config') != config
                    or trace.get('question') != row['question'] or trace.get('expected') != row['answer']
                    or trace.get('model') != configs[config]['model'] or trace.get('measured') is not True):
                raise ValueError(f'Trace identity mismatch: {config}/{path.stem}')
            if (not isinstance(trace.get('events'), list) or not isinstance(trace.get('status'), str)
                    or any(not isinstance(e, dict) for e in trace['events'])):
                raise ValueError(f'Invalid trace structure: {config}/{path.stem}')
            cost = trace.get('cost_usd')
            if cost is not None and (type(cost) not in (int, float) or not math.isfinite(cost) or cost < 0):
                raise ValueError(f'Invalid historical cost: {config}/{path.stem}')
            observations = []
            for index, event in enumerate(trace['events']):
                if event.get('type') == 'tool':
                    if not isinstance(event.get('name'), str) or not event['name']:
                        raise ValueError(f'Missing tool name: {config}/{path.stem}')
                    observations.append(observe_tool(event, index))
            issues = Counter(o['diagnostic']['error_kind'] for o in observations if o['diagnostic']['outcome'] == 'error')
            group = 'custom_20' if int(row['id'].split('-')[1]) >= 125 else 'original_125'
            record = dict(config=config, id=row['id'], group=group, metric_version=METRIC_VERSION,
                          correct=isinstance(trace.get('final'), str) and score_row(row, trace['final']),
                          answer_state=answer_state(row, trace.get('final')),
                          observed_run_status=trace['status'], historical_cost_usd=cost,
                          tool_counts=dict(Counter(o['tool'] for o in observations)),
                          technical_issues=dict(issues), technical_issue_count=sum(issues.values()),
                          http_error_count=sum(n for kind, n in issues.items() if kind in HTTP_ERRORS),
                          transport_error_count=sum(n for kind, n in issues.items() if kind in TRANSPORT_ERRORS),
                          observed_http_statuses=dict(Counter(str(o['diagnostic']['http_status']) for o in observations
                                                   if o['diagnostic']['http_status'] is not None)),
                          diagnostic_origins=dict(Counter(o['diagnostic_origin'] for o in observations)))
            records.append(record)
            if group == 'custom_20':
                own_details.append(dict(config=config, id=row['id'], question=row['question'],
                                        final=trace.get('final'), source_events=observations,
                                        trace_path=f'traces/{config}/{path.name}'))
    groups = {group: {config: _summary([r for r in records if r['config'] == config
                                      and (group == 'all_145' or r['group'] == group)])
                      for config in REQUIRED_CONFIGS} for group in ('all_145', 'original_125', 'custom_20')}
    return dict(version=AUDIT_VERSION, metric_version=METRIC_VERSION, generation_signature=signature,
                new_model_calls=0, new_network_calls=0, new_cost_usd=0,
                groups=groups, records=records, own_details=own_details,
                interpretation='Technical issues, final answer state and metric correctness are separate observations; no causal attribution.')


def validate_output(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    revisions = root / 'reports/revisions'
    if (root.is_relative_to(out) or out == revisions
            or out.is_relative_to(root) and not out.is_relative_to(revisions)):
        raise ValueError('Audit output must be separate from accepted project inputs')
    if out.exists() and any(p.is_symlink() for p in out.rglob('*')):
        raise ValueError('Audit output must not contain symlinks')
    if (out / 'audit.json').exists():
        if _read(out / 'audit.json').get('version') != AUDIT_VERSION:
            raise ValueError('Refusing to overwrite an unrelated audit')
    elif (out / 'audit.md').exists():
        raise ValueError('Refusing to overwrite an unrelated audit document')


def _markdown(report):
    lines = ['# ДЗ 1: источники и независимые признаки ошибок', '',
             f'Офлайн-аудит `{AUDIT_VERSION}`, метрика `{METRIC_VERSION}`. Новых вызовов, сетевых запросов и списаний нет.', '',
             'Исходные ответы, расходы, статусы и эталоны сохранены. Технический сбой не означает неверный факт; '
             'успешный инструмент не доказывает достаточность источника. `max_steps` не означает ошибку ответа.', '']
    for group, label in [('all_145', 'Все 145 вопросов'), ('original_125', 'Исходные 125 вопросов'),
                         ('custom_20', 'Собственные 20 вопросов')]:
        lines += [f'## {label}', '', '| Режим | Вопросов | Верно | Отказов | Любые технические ошибки, трейсов | HTTP/транспорт, трейсов | Событий инструментов |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for name, row in report['groups'][group].items():
            lines.append(f"| `{name}` | {row['n']} | {row['correct']} | {row['answer_states'].get('abstention', 0)} | "
                         f"{row['traces_with_technical_issues']} | {row['traces_with_transport_errors']} | {sum(row['tool_events'].values())} |")
        lines.append('')
    lines += ['## Наблюдаемые ошибки на собственных вопросах', '',
              '| Режим | HTTPError без кода | ReadTimeout | Аргументы | Повтор |', '|---|---:|---:|---:|---:|']
    for name, row in report['groups']['custom_20'].items():
        errors = row['technical_issues']
        lines.append(f"| `{name}` | {errors.get('legacy_http_error', 0)} | {errors.get('legacy_read_timeout', 0)} | "
                     f"{errors.get('invalid_arguments', 0)} | {errors.get('repeated_tool_call', 0)} |")
    mini = report['groups']['custom_20']['cheap_search_read']
    lines += ['', 'События включают отклонённые аргументы и заблокированные повторы. Это не число реальных HTTP-запросов. '
              'Столбец HTTP/транспорт учитывает только явно наблюдаемые классы HTTPError, тайм-аута, соединения и аналогичных '
              'сбоев доступа; ошибки аргументов, извлечения и повторов считаются отдельно. '
              f"На собственных вопросах у Mini с чтением {mini['traces_with_technical_issues']} трейсов с любой технической ошибкой, "
              f"из них {mini['traces_with_transport_errors']} с HTTP/транспортной.", '',
              'У старых ошибок `HTTPError` отсутствуют HTTP-коды и причины. Им не присваиваются 403/429, rate limit '
              'или anti-bot challenge. Успешной старой выдаче также не приписывается HTTP 200. '
              'Отсутствующие сведения остаются `null`; более поздняя проверка доступности не восстанавливает прежний ответ.', '',
              'У новых результатов сохраняется исходная структурированная диагностика. Невалидная диагностика '
              'отклоняется; она не подменяется успешным legacy-разбором.', '',
              '## Подробные наблюдения собственных вопросов', '',
              f"Ниже указаны точные пути к исходным трейсaм. В `audit.json` для всех {len(report['own_details'])} результатов сохранены "
              'индексы событий (с нуля), фактически переданные URL и URL полученных результатов. '
              'Эталонные URL в наблюдения поиска не добавляются.', '',
              '| Режим | ID | Ответ | Метрика | Ошибок инструментов | Трейс |', '|---|---|---|---|---:|---|']
    by_key = {(r['config'], r['id']): r for r in report['records']}
    for detail in report['own_details']:
        r = by_key[detail['config'], detail['id']]
        lines.append(f"| `{r['config']}` | `{r['id']}` | {r['answer_state']} | {'верно' if r['correct'] else 'неверно'} | "
                     f"{r['technical_issue_count']} | `{detail['trace_path']}` |")
    lines += ['', 'Состояния `answer` / `abstention` / `missing`, исходный статус исполнения и правильность '
              'сохранены отдельными полями. Пересечение ошибок и неверных ответов показывает совместное наблюдение, '
              'а не доказанную причину. Метрика остаётся эвристической.', '',
              f"Полные оценки всех {len(report['records'])} ответов, сводки, {len(report['own_details'])} подробных наблюдений, сохранённая сигнатура генерации "
              'и контрольные суммы находятся в [audit.json](audit.json).']
    return '\n'.join(lines) + '\n'


def build_audit(root=ROOT, out=DEFAULT_OUT):
    root, out = Path(root).resolve(), Path(out).resolve()
    validate_output(root, out)
    inputs = [root / 'data/fresh.jsonl', root / 'traces/manifest.json']
    for config in REQUIRED_CONFIGS:
        inputs += sorted((root / 'traces' / config).glob('*.json'))
    before = {str(path.relative_to(root)): _sha(path) for path in inputs}
    report = audit_traces(load_rows(root / 'data/fresh.jsonl'), root / 'traces')
    if any(_sha(root / path) != digest for path, digest in before.items()):
        raise ValueError('Inputs changed during source audit')
    report.update(input_hashes_verified=True, input_sha256=before,
                  code_sha256={p: _sha(ROOT / p) for p in ('freshqa/source_audit.py', 'freshqa/dataset.py', 'freshqa/experiment.py',
                                                           'freshqa/quantities.py')})
    out.mkdir(parents=True, exist_ok=True)
    for filename, content in [('audit.json', json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n'),
                              ('audit.md', _markdown(report))]:
        target = out / filename
        temporary = out / (filename + '.tmp')
        temporary.write_text(content, encoding='utf-8')
        temporary.replace(target)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    report = build_audit(ROOT, args.out)
    print(f"Audited {len(report['records'])} saved answers; new model/network calls: 0. {args.out / 'audit.md'}")


if __name__ == '__main__':
    main()
