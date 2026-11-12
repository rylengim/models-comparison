import json
from pathlib import Path

import pytest

from hw3.report import render_report


def write_trace(root, config, qid='one', answer='Alice', initial=None, cost=.01):
    record = dict(id=qid, question='Who won?', config=config, model='test-model',
                  final=answer, status='ok', measured=True, steps=1, prompt_tokens=10,
                  completion_tokens=2, cost_usd=cost, correct=False,
                  call_records=[dict(question_id=qid, config=config, cost_usd=cost,
                                     prompt_tokens=10, completion_tokens=2,
                                     status_code=200, request_id=f'{config}-{qid}')])
    if initial is not None:
        record['initial_answer'] = initial
    path = root / config / f'{qid}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record))
    return path


def test_report_regrades_outputs_counts_all_costs_and_pairs_critic(tmp_path):
    root, out = tmp_path / 'results', tmp_path / 'report'
    rows = [dict(id='one', question='Who won?', answer='Alice', evidence='Alice won', url='https://example.com')]
    trace_path = write_trace(root, 'critic', initial='Bob')
    before = trace_path.read_bytes()
    summary = render_report(root, rows, out)
    critic = next(row for row in summary if row['config'] == 'critic')
    assert critic['correct'] == 1
    assert critic['accuracy'] == 1
    assert critic['cents_per_question'] == pytest.approx(1)
    assert critic['tokens_per_question'] == 12
    assert critic['fixed'] == 1 and critic['broken'] == 0
    assert trace_path.read_bytes() == before
    assert (out / 'cost_quality.png').stat().st_size > 1000
    assert 'Ручная проверка ещё не выполнена' in (out / 'report.md').read_text()
    assert not next(row for row in summary if row['config'] == 'bare')['complete']


def test_partial_unknown_cost_cannot_support_comparison(tmp_path):
    rows = [dict(id=id, question='Who won?', answer='Alice') for id in ('one', 'two')]
    write_trace(tmp_path / 'results', 'bare', answer='Bob', cost=None)
    summary = render_report(tmp_path / 'results', rows, tmp_path / 'out')
    bare = summary[0]
    assert bare['n'] == 1 and bare['total'] == 2 and not bare['complete']
    assert bare['cost_usd'] is None and bare['cents_per_correct'] is None
    candidates = json.loads((tmp_path / 'out' / 'audit_candidates.json').read_text())
    assert candidates[0]['answer'] == 'Bob'
    assert candidates[0]['expected'] == 'Alice'


def test_audit_candidates_follow_dataset_order_not_lexical_filenames(tmp_path):
    rows = [dict(id=qid, question='Who won?', answer='Alice') for qid in ('q2', 'q10', 'q3')]
    for row in rows:
        write_trace(tmp_path / 'results', 'bare', qid=row['id'], answer='unknown')
    render_report(tmp_path / 'results', rows, tmp_path / 'out')
    candidates = json.loads((tmp_path / 'out' / 'audit_candidates.json').read_text())
    assert [row['id'] for row in candidates] == ['q2', 'q10', 'q3']


def test_report_refuses_wrong_question_or_duplicate_request_ids(tmp_path):
    rows = [dict(id='one', question='Different question', answer='Alice')]
    write_trace(tmp_path / 'results', 'bare')
    with pytest.raises(ValueError, match='question'):
        render_report(tmp_path / 'results', rows, tmp_path / 'out')


def test_duplicate_provider_request_cannot_be_charged_twice(tmp_path):
    rows = [dict(id='one', question='Who won?', answer='Alice')]
    root = tmp_path / 'results'
    first = write_trace(root, 'bare')
    second = write_trace(root, 'langgraph')
    data = json.loads(second.read_text())
    data['call_records'][0]['request_id'] = json.loads(first.read_text())['call_records'][0]['request_id']
    second.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Duplicate request_id'):
        render_report(root, rows, tmp_path / 'out')


def test_ledger_mismatch_blocks_report_instead_of_hiding_spend(tmp_path):
    rows = [dict(id='one', question='Who won?', answer='Alice')]
    root = tmp_path / 'results'
    path = write_trace(root, 'bare')
    data = json.loads(path.read_text())
    data['cost_usd'] = .001
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='ledger cost mismatch'):
        render_report(root, rows, tmp_path / 'out')


def test_report_requires_explicit_measured_and_http_ledger(tmp_path):
    rows = [dict(id='one', question='Who won?', answer='Alice')]
    root = tmp_path / 'results'
    path = write_trace(root, 'bare')
    record = json.loads(path.read_text())
    record.pop('measured')
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='Unmeasured'):
        render_report(root, rows, tmp_path / 'out')
    record['measured'] = True
    record.pop('call_records')
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='call_records'):
        render_report(root, rows, tmp_path / 'out')


def test_empty_ledger_requires_exact_zero_totals(tmp_path):
    rows = [dict(id='one', question='Who won?', answer='Alice')]
    root = tmp_path / 'results'
    path = write_trace(root, 'bare')
    record = json.loads(path.read_text())
    record['call_records'] = []
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='ledger'):
        render_report(root, rows, tmp_path / 'out')
    record.update(cost_usd=0, prompt_tokens=0, completion_tokens=0, steps=0,
                  status='budget_exceeded', final='unknown')
    path.write_text(json.dumps(record))
    summary = render_report(root, rows, tmp_path / 'out')
    assert summary[0]['cost_usd'] == 0
    assert summary[0]['calls_per_question'] == 0
    assert not summary[0]['complete']
