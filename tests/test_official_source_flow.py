"""Search → verified URL read → final answer, with synthetic transport only."""
import copy
import json
import socket
from urllib.parse import quote

import pytest
import requests

from freshqa.agent import run_agent
from freshqa.tools import ToolRegistry


class Response:
    status_code = 200
    headers = {'Content-Type': 'text/html'}
    encoding = 'utf-8'

    def __init__(self, body):
        self.body = body

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self.body.encode()

    def close(self):
        pass


def listing(url, text='Astronomy result'):
    return f'<div class="result"><a class="result__a" href="{url}">Observation</a><p class="result__snippet">{text}</p></div>'


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('No real network')
    monkeypatch.setattr(requests.Session, 'get', forbidden)


def test_official_search_filters_before_limit_and_unwraps_search_links(monkeypatch):
    html = ''.join(listing(f'https://example.org/{i}') for i in range(6))
    html += listing('//duckduckgo.com/l/?uddg=' + quote('https://www.eso.org/public/news/synthetic/', safe=''))
    calls = []
    def get(session, url, **kwargs):
        calls.append((url, kwargs))
        return Response(html)
    monkeypatch.setattr(requests.Session, 'get', get)
    result = json.loads(ToolRegistry(official_sources=True).run('official_search', {'query': 'new nebula'}))
    assert [r['url'] for r in result['results']] == ['https://www.eso.org/public/news/synthetic/']
    assert len(calls) == 1 and calls[0][1]['allow_redirects'] is False
    assert 'site:eso.org' in calls[0][1]['params']['q']
    assert result['diagnostic']['outcome'] == 'ok'


def test_official_search_is_opt_in_and_rejects_question_metadata():
    assert 'error' in json.loads(ToolRegistry().run('official_search', {'query': 'nebula'}))
    result = json.loads(ToolRegistry(official_sources=True).run('official_search',
                       {'query': 'nebula', 'answer': 'SECRET GOLD', 'url': 'https://www.eso.org/right'}))
    assert result['diagnostic']['error_kind'] == 'invalid_arguments'
    schema = ToolRegistry(official_sources=True).schemas(['official_search'])[0]['function']['parameters']
    assert set(schema['properties']) == {'query', 'source'}


def test_new_tool_runs_through_existing_loop_and_reads_found_url(monkeypatch):
    found_url = 'https://www.eso.org/public/news/synthetic/'
    http_calls, model_inputs = [], []
    def get(session, url, **kwargs):
        http_calls.append((url, kwargs))
        if 'duckduckgo' in url:
            return Response(listing(found_url))
        assert kwargs['headers']['Host'] == 'www.eso.org'
        return Response('<main><p>The newly observed object is TEST-NEBULA.</p></main>')
    monkeypatch.setattr(requests.Session, 'get', get)
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *a, **k: [(2, 1, 6, '', ('93.184.216.34', 443))])

    class Client:
        spent_usd = 0
        def complete(self, **kwargs):
            model_inputs.append(copy.deepcopy(kwargs))
            n = len(model_inputs)
            if n == 1:
                name, args = 'official_search', {'query': 'new observed object', 'source': 'eso'}
            elif n == 2:
                observed = json.loads(kwargs['messages'][-1]['content'])
                name, args = 'page_find', {'url': observed['results'][0]['url'], 'keywords': ['object']}
            else:
                observed = json.loads(kwargs['messages'][-2]['content'])
                assert 'TEST-NEBULA' in ' '.join(observed['excerpts'])
                assert kwargs['tools'] is None  # existing final-answer reserve still applies
                return {'message': {'role': 'assistant', 'content': 'FINAL: TEST-NEBULA'},
                        'cost_usd': 0, 'prompt_tokens': 1, 'completion_tokens': 1, 'request_id': 'fake-3'}
            return {'message': {'role': 'assistant', 'content': None, 'tool_calls': [
                {'id': str(n), 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]},
                'cost_usd': 0, 'prompt_tokens': 1, 'completion_tokens': 1, 'request_id': f'fake-{n}'}

    result = run_agent('What is the newly observed object?', model='synthetic-test', client=Client(),
                       registry=ToolRegistry(official_sources=True), tool_names=['official_search', 'page_find'], max_steps=3)
    assert result['final'] == 'TEST-NEBULA' and result['steps'] == 3
    assert len(http_calls) == 2 and http_calls[1][0].startswith('https://93.184.216.34:443/')
    assert all(json.loads(e['output'])['diagnostic']['outcome'] == 'ok'
               for e in result['events'] if e['type'] == 'tool')
    assert 'TEST-NEBULA' not in json.dumps(model_inputs[0])


def test_selected_official_host_still_cannot_resolve_to_private_network(monkeypatch):
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *a, **k: [(2, 1, 6, '', ('10.0.0.1', 443))])
    result = json.loads(ToolRegistry(official_sources=True).run('page_find', {'url': 'https://www.eso.org/article'}))
    assert result['diagnostic']['error_kind'] == 'blocked_url'


def test_bad_or_overlong_links_do_not_hide_valid_results_or_get_rewritten(monkeypatch):
    long_url = 'https://www.eso.org/' + 'a' * 2100
    good_url = 'https://www.eso.org/article'
    html = listing('https://[') + listing(long_url) + listing(good_url)
    monkeypatch.setattr(requests.Session, 'get', lambda *a, **k: Response(html))
    result = json.loads(ToolRegistry(official_sources=True).run('official_search', {'query': 'nebula'}))
    assert [r['url'] for r in result['results']] == [good_url]
