"""The revision is a read-only replay of accepted measurements, not a new run."""
import json
from pathlib import Path
import socket

import pytest

from freshqa.revisions import build_revision, validate_output

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('relative', ['.', 'reports', 'reports/revisions', 'reports/groq-check',
                                     'data/new', 'traces/new', 'hw2/results/new', 'hw3/results/new',
                                     'reports/revisions/2026-11-15',
                                     'reports/revisions/2026-11-15/hw1',
                                     'reports/revisions/2026-11-15-sources'])
def test_revision_rejects_output_over_accepted_inputs(relative):
    with pytest.raises(ValueError):
        validate_output(ROOT, ROOT / relative)


def test_frozen_cross_homework_replay_keeps_coverage_costs_and_missing_raw_visible(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Offline revision attempted network or credentials')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket, 'create_connection', forbidden)
    import dotenv
    monkeypatch.setattr(dotenv, 'load_dotenv', forbidden)
    summary = build_revision(ROOT, tmp_path / 'revision')
    assert summary['new_model_calls'] == 0
    assert summary['new_cost_usd'] == 0
    assert summary['input_hashes_verified'] is True
    for hw, count in [('hw1', 725), ('hw2', 570), ('hw3', 200)]:
        grades = json.loads((tmp_path / f'revision/{hw}/grades.json').read_text())
        assert len(grades) == count
        assert len({(r['config'], r['id']) for r in grades}) == count
        assert all(r['metric_version'] == summary['metric_version'] for r in grades)
    replay = json.loads((tmp_path / 'revision/hw2/parser_replay.json').read_text())
    assert replay['coverage'] == {'cheap_rag': {'unavailable_raw_response': 114},
                                  'medium_rag': {'unavailable_raw_response': 114}}
    assert replay['full_benchmark_reparsed'] is False
    samples = {s['config']: s for s in replay['separate_diagnostic']}
    assert samples['medium_rag']['status'] == 'answered'
    assert samples['medium_rag']['fact_correct'] is True
    assert samples['cheap_rag']['validation_error'] == 'unsupported_span'
    assert all(s['envelope_verified'] is False for s in samples.values())
    assert summary['hw3_changed_config_totals'] == 0
    assert all(c['old_cost_usd'] == pytest.approx(c['new_cost_usd'])
               for c in summary['comparison'])


def test_current_hw1_revision_grades_match_report_policy_for_truncated_gold(tmp_path, monkeypatch):
    from freshqa.dataset import load_rows

    row = load_rows(ROOT / 'data/fresh.jsonl')[0]
    path = ROOT / 'traces/cheap_search' / f"{row['id']}.json"
    original_read_text = Path.read_text

    def truncated_gold_in_memory(source, *args, **kwargs):
        text = original_read_text(source, *args, **kwargs)
        if source == path:
            trace = json.loads(text)
            trace.update(correct=True, final=row['answer'], status='output_truncated', events=[
                {'type': 'llm', 'finish_reason': 'length',
                 'message': {'content': 'FINAL: ' + row['answer']}}])
            return json.dumps(trace)
        return text

    monkeypatch.setattr(Path, 'read_text', truncated_gold_in_memory)
    out = tmp_path / 'revision'
    summary = build_revision(ROOT, out)
    grades = json.loads((out / 'hw1/grades.json').read_text())
    grade = next(r for r in grades if r['config'] == 'cheap_search' and r['id'] == row['id'])
    assert grade['old_correct'] is True
    assert grade['correct'] is False
    from freshqa.evaluation import GRADING_POLICY_VERSION
    assert grade['grading_policy_version'] == summary['grading_policy_version'] == GRADING_POLICY_VERSION
    report = json.loads((out / 'hw1/report.json').read_text())
    assert report['cheap_search']['correct'] == sum(r['correct'] for r in grades if r['config'] == 'cheap_search')
    provenance = json.loads((out / 'provenance.json').read_text())
    assert provenance['grading_policy_version'] == GRADING_POLICY_VERSION
    assert 'freshqa/evaluation.py' in provenance['evaluation_code_sha256']


def test_revision_function_and_cli_default_to_new_versioned_output(monkeypatch):
    from freshqa import revisions
    from freshqa.evaluation import GRADING_POLICY_VERSION
    import sys

    default = build_revision.__defaults__[1]
    assert default == ROOT / 'reports/revisions' / GRADING_POLICY_VERSION
    seen = []
    monkeypatch.setattr(sys, 'argv', ['freshqa.revisions'])
    monkeypatch.setattr(revisions, 'build_revision', lambda root, out: seen.append(out) or {})
    revisions.main()
    assert seen == [default]
