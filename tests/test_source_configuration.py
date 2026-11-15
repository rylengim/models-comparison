"""Opt-in sources do not alter archived configuration coverage or hidden costs."""
import json
from types import SimpleNamespace

import pytest

from freshqa.experiment import BONUS_CONFIGS, REQUIRED_CONFIGS, configurations, dataset_hash, benchmark, write_json
from freshqa.reporting import render_report


def one_row():
    return {'id': 'one', 'source': 'fresh', 'question': 'Which observation was announced?',
            'answer': 'Example', 'evidence': 'Example was announced.', 'url': 'https://example.org/gold-only'}


def test_original_five_and_seven_bonus_configs_keep_their_meaning():
    assert REQUIRED_CONFIGS == ['strong_no_tools', 'cheap_no_tools', 'cheap_search',
                                'cheap_search_read', 'medium_search_read']
    assert BONUS_CONFIGS == REQUIRED_CONFIGS + ['cheap_extra_search', 'cascade']
    config = configurations()['cheap_official_sources']
    assert config['model'] == configurations()['cheap_search_read']['model']
    assert config['tools'] == configurations()['cheap_search_read']['tools'] + ['official_search']


def test_new_source_scope_only_reaches_the_requested_benchmark_configuration(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr('freshqa.tools.ToolRegistry', lambda **kw: kw)
    def fake_run(question, *, registry, client, **kwargs):
        seen.append((question, registry, kwargs))
        client.spent_usd += .01
        return dict(final='unknown', status='ok', cost_usd=.01, known_cost_usd=.01,
                    steps=1, elapsed_s=0, events=[])
    monkeypatch.setattr('freshqa.agent.run_agent', fake_run)
    assert benchmark([one_row()], trace_dir=tmp_path, config_names=['cheap_official_sources'],
                     client=SimpleNamespace(spent_usd=0, budget_usd=1))
    question, registry, kwargs = seen[0]
    assert question == one_row()['question']
    assert registry == {'extra_search': False, 'official_sources': True}
    assert not {'answer', 'expected', 'evidence', 'url'} & kwargs.keys()
    assert one_row()['url'] not in str(seen)


def test_old_manifest_report_does_not_invent_a_new_unmeasured_config(tmp_path, monkeypatch):
    rows = [one_row()]
    old_configs = {k: v for k, v in configurations().items() if k != 'cheap_official_sources'}
    traces = tmp_path / 'traces'
    write_json(traces / 'manifest.json', {'signature': {'dataset_sha256': dataset_hash(rows), 'configs': old_configs}})
    monkeypatch.setattr('freshqa.reporting._chart', lambda *args: None)
    summary, _ = render_report(rows, traces, tmp_path / 'report')
    assert set(summary) == set(old_configs)
    assert 'cheap_official_sources' not in (tmp_path / 'report/report.csv').read_text()


def test_report_rejects_an_unknown_manifest_configuration(tmp_path):
    rows = [one_row()]
    configs = {**configurations(), 'unexpected': {'model': 'unknown', 'tools': []}}
    write_json(tmp_path / 'traces/manifest.json', {'signature': {'dataset_sha256': dataset_hash(rows), 'configs': configs}})
    with pytest.raises(ValueError, match='конфигурац'):
        render_report(rows, tmp_path / 'traces', tmp_path / 'report')


def test_offline_schema_command_does_not_read_dotenv_or_construct_a_client(monkeypatch, capsys):
    from freshqa import __main__ as cli
    def forbidden(*args, **kwargs):
        raise AssertionError('Offline schema inspection must not read credentials or call a model')
    monkeypatch.setattr(cli, 'load_dotenv', forbidden)
    monkeypatch.setattr(cli, 'get_client', forbidden)
    class Registry:
        def __init__(self, extra_search=False, *, official_sources=False):
            assert not extra_search and official_sources
        def schemas(self, names):
            return names
    monkeypatch.setattr('freshqa.tools.ToolRegistry', Registry)
    assert cli.main(['schemas', '--official-sources']) == 0
    assert 'official_search' in json.loads(capsys.readouterr().out)


def test_live_defaults_are_loaded_after_dotenv_and_only_for_live_commands(monkeypatch, tmp_path):
    from freshqa import __main__ as cli
    seen = {}
    def dotenv(_path):
        monkeypatch.setenv('CHEAP_MODEL', 'configured-model')
        monkeypatch.setenv('BUDGET_USD', '1.25')
    monkeypatch.setattr(cli, 'load_dotenv', dotenv)
    monkeypatch.setattr(cli, 'get_client', lambda budget: seen.setdefault('budget', budget))
    monkeypatch.setattr('freshqa.tools.ToolRegistry', lambda *args, **kwargs: kwargs)
    def fake_run(question, **kwargs):
        seen.update(kwargs)
        return {'final': 'unknown', 'status': 'ok'}
    monkeypatch.setattr('freshqa.agent.run_agent', fake_run)
    assert cli.main(['ask', 'Example question', '--official-sources', '--output', str(tmp_path / 'answer.json')]) == 0
    assert seen['budget'] == 1.25
    assert seen['model'] == 'configured-model'
    assert seen['registry']['official_sources'] is True


@pytest.mark.parametrize('selection,expected', [
    (['--bonuses'], BONUS_CONFIGS),
    (['--configs', 'cheap_official_sources'], ['cheap_official_sources']),
])
def test_paid_configuration_selection_never_adds_the_new_mode_implicitly(selection, expected, monkeypatch, tmp_path):
    from freshqa import __main__ as cli
    seen = {}
    monkeypatch.setattr(cli, 'load_dotenv', lambda *args: None)
    monkeypatch.setattr(cli, 'get_client', lambda budget: object())
    monkeypatch.setattr(cli, 'load_rows', lambda path: [one_row()])
    monkeypatch.setattr(cli, 'render_report', lambda *args: None)
    def fake_benchmark(rows, **kwargs):
        seen.update(kwargs)
        return True
    monkeypatch.setattr(cli, 'benchmark', fake_benchmark)
    assert cli.main(['benchmark', '--traces', str(tmp_path / 'traces'),
                     '--out', str(tmp_path / 'report'), *selection]) == 0
    assert seen['config_names'] == expected


@pytest.mark.parametrize('dependency', ['freshqa/official_sources.py', 'freshqa/source_diagnostics.py'])
def test_hw2_generation_signature_changes_when_source_protocol_dependency_changes(dependency, monkeypatch):
    from pathlib import Path
    from hw2.benchmark import ROOT, _signature
    contents = {}
    # Simulated files isolate signature behavior from actual generation or credentials.
    monkeypatch.setattr(Path, 'read_bytes', lambda path: contents.get(str(path), b'unchanged'))
    before = _signature([], {}, max_steps=6)
    contents[str(ROOT / dependency)] = b'changed source behavior'
    after = _signature([], {}, max_steps=6)
    assert before != after
    assert before['dataset_sha256'] == after['dataset_sha256']
