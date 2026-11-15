import copy
import json
from pathlib import Path
import socket

import pytest

from freshqa.dataset import METRIC_VERSION
from freshqa.experiment import REQUIRED_CONFIGS, dataset_hash
from freshqa.source_audit import audit_traces, build_audit, observe_tool, validate_output


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def archive(tmp_path):
    root = tmp_path / 'project'
    rows = [{'id': 'fresh-0', 'question': 'Who won?', 'answer': 'Alice'},
            {'id': 'fresh-125', 'question': 'Which star?', 'answer': 'S301'}]
    (root / 'data').mkdir(parents=True)
    (root / 'data/fresh.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    signature = {'n': 2, 'dataset_sha256': dataset_hash(rows),
                 'configs': {c: {'model': c + '-model'} for c in REQUIRED_CONFIGS}}
    write(root / 'traces/manifest.json', {'signature': signature})
    for config in REQUIRED_CONFIGS:
        for row in rows:
            write(root / 'traces' / config / (row['id'] + '.json'),
                  dict(id=row['id'], question=row['question'], expected=row['answer'], config=config,
                       model=config + '-model', measured=True, final=row['answer'], status='max_steps',
                       cost_usd=.01, events=[]))
    return root, rows


def event(output, *, name='web_search', arguments=None):
    return dict(type='tool', name=name, arguments=arguments or {'query': 'star'}, output=json.dumps(output))


def test_legacy_error_never_invents_http_code_or_causal_explanation():
    original = event({'error': 'Source request failed: HTTPError'})
    saved = copy.deepcopy(original)
    result = observe_tool(original, 7)
    assert result['event_index'] == 7
    assert result['diagnostic']['error_kind'] == 'legacy_http_error'
    assert result['diagnostic']['http_status'] is None
    assert result['diagnostic']['retryable'] is None
    assert result['diagnostic_origin'] == 'legacy_inferred'
    assert original == saved


def test_new_diagnostic_and_observed_urls_are_preserved_without_a_request():
    diagnostic = dict(version='source-diagnostics-v1', stage='fetch', outcome='error',
                      error_kind='http_error', http_status=429, retryable=True)
    result = observe_tool(event({'error': 'Source request failed', 'diagnostic': diagnostic},
                                name='page_find', arguments={'url': 'https://example.org/article'}), 3)
    assert result['diagnostic'] == diagnostic
    assert result['diagnostic_origin'] == 'recorded'
    assert result['urls'] == [{'origin': 'argument', 'url': 'https://example.org/article'}]


@pytest.mark.parametrize('error,stage,kind', [
    ('Source request failed: ReadTimeout', 'fetch', 'legacy_read_timeout'),
    ('Invalid tool arguments', 'validation', 'invalid_arguments'),
    ('repeated_tool_call', 'control', 'repeated_tool_call'),
])
def test_legacy_control_and_validation_are_separate_from_fetch_errors(error, stage, kind):
    result = observe_tool(event({'error': error}), 1)
    assert result['diagnostic']['stage'] == stage
    assert result['diagnostic']['error_kind'] == kind


def test_empty_search_and_unmatched_page_are_not_network_failures():
    search = observe_tool(event({'source': 'Wikipedia', 'results': []}), 1)
    page = observe_tool(event({'found': True, 'keywords_matched': False, 'excerpts': ['Opening text.']},
                              name='page_find'), 2)
    assert search['diagnostic']['outcome'] == 'empty'
    assert page['diagnostic']['outcome'] == 'ok'
    assert page['keywords_matched'] is False
    assert search['diagnostic']['http_status'] is page['diagnostic']['http_status'] is None


def test_truncated_legacy_observation_is_not_a_successful_source_result():
    result = observe_tool(event({'truncated': True, 'text': '{"source": "Wikipedia"'}), 1)
    assert result['diagnostic']['outcome'] == 'error'
    assert result['diagnostic']['error_kind'] == 'output_truncated'
    assert result['diagnostic']['http_status'] is None


@pytest.mark.parametrize('name', ['web_search', 'web_search_extra', 'official_search', 'page_find'])
def test_unrecognized_network_payload_does_not_claim_success(name):
    result = observe_tool(event({}, name=name), 1)
    assert result['diagnostic']['outcome'] == 'error'
    assert result['diagnostic']['error_kind'] == 'legacy_unclassified_output'
    assert result['diagnostic']['http_status'] is None


def test_recorded_success_cannot_have_an_http_failure_code():
    diagnostic = dict(version='source-diagnostics-v1', stage='fetch', outcome='ok',
                      error_kind=None, http_status=503, retryable=None)
    with pytest.raises(ValueError, match='diagnostic'):
        observe_tool(event({'results': [], 'diagnostic': diagnostic}), 1)


@pytest.mark.parametrize('patch', [{'version': 'unknown'}, {'http_status': True},
                                  {'http_status': 999}, {'retryable': 'yes'}, {'outcome': 'maybe'},
                                  {'stage': []}, {'outcome': []}])
def test_malformed_recorded_diagnostic_is_not_silently_reinterpreted(patch):
    diagnostic = dict(version='source-diagnostics-v1', stage='fetch', outcome='error',
                      error_kind='http_error', http_status=503, retryable=True)
    diagnostic.update(patch)
    with pytest.raises(ValueError, match='diagnostic'):
        observe_tool(event({'error': 'failure', 'diagnostic': diagnostic}), 1)


def test_error_payload_cannot_claim_successful_diagnostic():
    diagnostic = dict(version='source-diagnostics-v1', stage='fetch', outcome='ok',
                      error_kind=None, http_status=200, retryable=None)
    with pytest.raises(ValueError, match='diagnostic'):
        observe_tool(event({'error': 'failure', 'diagnostic': diagnostic}), 1)


def test_nonstring_legacy_error_stays_unclassified_instead_of_crashing():
    result = observe_tool(event({'error': {'message': 'not a known legacy form'}}), 1)
    assert result['diagnostic']['error_kind'] == 'legacy_tool_error'
    assert result['diagnostic']['http_status'] is None


def test_technical_issue_answer_state_and_metric_are_independent(archive):
    root, rows = archive
    p = root / 'traces/cheap_search_read/fresh-125.json'
    record = json.loads(p.read_text())
    record['events'] = [event({'error': 'Source request failed: HTTPError'})]
    write(p, record)
    report = audit_traces(rows, root / 'traces')
    result = next(r for r in report['records'] if r['config'] == 'cheap_search_read' and r['id'] == 'fresh-125')
    assert result['answer_state'] == 'answer'
    assert result['correct'] is True
    assert result['observed_run_status'] == 'max_steps'
    assert result['technical_issue_count'] == 1
    assert report['groups']['custom_20']['cheap_search_read']['technical_x_correct']['issues_correct'] == 1
    assert report['groups']['custom_20']['cheap_search_read']['transport_x_correct']['issues_correct'] == 1
    assert len(report['records']) == 10
    assert len(report['own_details']) == 5


@pytest.mark.parametrize('final,state', [('unknown\n\nS301 was mentioned.', 'abstention'),
                                       ('unknown: S301 was mentioned.', 'abstention'), ('', 'missing')])
def test_final_state_uses_explicit_abstention_boundary(archive, final, state):
    root, rows = archive
    p = root / 'traces/cheap_search_read/fresh-125.json'
    record = json.loads(p.read_text())
    record['final'] = final
    write(p, record)
    report = audit_traces(rows, root / 'traces')
    result = next(r for r in report['records'] if r['config'] == 'cheap_search_read' and r['id'] == 'fresh-125')
    assert result['answer_state'] == state
    assert result['correct'] is False


def test_missing_or_nonstring_final_cannot_receive_credit(archive):
    root, rows = archive
    p = root / 'traces/cheap_search_read/fresh-125.json'
    record = json.loads(p.read_text())
    record['final'] = ['S301']
    write(p, record)
    report = audit_traces(rows, root / 'traces')
    result = next(r for r in report['records'] if r['config'] == 'cheap_search_read' and r['id'] == 'fresh-125')
    assert result['answer_state'] == 'missing'
    assert result['correct'] is False


def test_exact_unknown_title_is_an_answer_not_abstention():
    from freshqa.source_audit import answer_state
    row = {'answer': 'Unknown: Cosmic Time Machine'}
    assert answer_state(row, 'Unknown: Cosmic Time Machine') == 'answer'
    assert answer_state(row, 'unknown: I considered Unknown: Cosmic Time Machine.') == 'abstention'


@pytest.mark.parametrize('problem', ['missing_trace', 'wrong_question', 'wrong_model', 'wrong_expected',
                                    'wrong_id', 'unmeasured', 'extra_trace', 'dataset_hash'])
def test_archive_identity_and_coverage_fail_closed(archive, problem):
    root, rows = archive
    p = root / 'traces/cheap_search_read/fresh-125.json'
    record = json.loads(p.read_text())
    if problem == 'missing_trace':
        p.unlink()
    elif problem == 'extra_trace':
        write(p.with_name('extra.json'), record)
    elif problem == 'dataset_hash':
        rows[0]['question'] = 'Changed question'
    else:
        key, value = {'wrong_question': ('question', 'Changed'), 'wrong_model': ('model', 'Other'),
                      'wrong_expected': ('expected', 'Other'), 'wrong_id': ('id', 'fresh-0'),
                      'unmeasured': ('measured', False)}[problem]
        record[key] = value
        write(p, record)
    with pytest.raises(ValueError):
        audit_traces(rows, root / 'traces')


@pytest.mark.parametrize('relative', ['.', 'reports', 'reports/groq-check', 'reports/revisions',
                                     'data', 'traces/new', 'hw2/results/new'])
def test_output_cannot_overwrite_accepted_project_data(tmp_path, relative):
    with pytest.raises(ValueError):
        validate_output(tmp_path, tmp_path / relative)


def test_build_is_offline_and_preserves_all_archive_bytes(archive, monkeypatch):
    root, _ = archive
    before = {str(p): p.read_bytes() for p in root.rglob('*') if p.is_file()}
    def forbidden(*args, **kwargs):
        raise AssertionError('No network in an offline audit')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket, 'getaddrinfo', forbidden)
    import dotenv
    monkeypatch.setattr(dotenv, 'load_dotenv', forbidden)
    out = root / 'reports/revisions/new-audit'
    report = build_audit(root, out)
    assert all(Path(p).read_bytes() == data for p, data in before.items())
    assert report['input_hashes_verified'] is True
    assert report['metric_version'] == METRIC_VERSION
    assert report['new_model_calls'] == report['new_network_calls'] == 0
    assert (out / 'audit.json').is_file()
    assert (out / 'audit.md').is_file()


def test_accepted_archive_coverage_and_legacy_uncertainty_are_retained(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('No network in an offline audit')
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    monkeypatch.setattr(socket, 'getaddrinfo', forbidden)
    from freshqa.dataset import load_rows
    root = Path(__file__).resolve().parents[1]
    report = audit_traces(load_rows(root / 'data/fresh.jsonl'), root / 'traces')
    assert len(report['records']) == 725
    assert len(report['own_details']) == 100
    assert len({(r['config'], r['id']) for r in report['records']}) == 725
    own = report['groups']['custom_20']['cheap_search_read']
    assert own['n'] == 20 and own['correct'] == 2
    assert own['technical_issues']['legacy_http_error'] == 40
    assert own['traces_with_technical_issues'] == 13
    assert own['traces_with_http_errors'] == own['traces_with_transport_errors'] == 12
    assert own['observed_http_statuses'] == {}
    assert own['tool_events'] == {'web_search': 89, 'page_find': 6}
    observations = [e for r in report['own_details'] for e in r['source_events']]
    assert all(e['diagnostic']['http_status'] is None for e in observations)
    assert not any(u['origin'] == 'argument' for e in observations for u in e['urls'])
