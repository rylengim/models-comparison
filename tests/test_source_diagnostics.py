"""Offline source failures preserve what was observed, without inventing causes."""
import json
import socket

import pytest
import requests

from freshqa.tools import MAX_RESPONSE_BYTES, ToolRegistry


class Response:
    def __init__(self, body, status=200, content_type='application/json'):
        self.body = body if isinstance(body, str) else json.dumps(body)
        self.status_code = status
        self.headers = {'Content-Type': content_type}
        self.encoding = 'utf-8'
        self.closed = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError('sensitive request URL and headers')

    def iter_content(self, chunk_size):
        yield self.body.encode()

    def close(self):
        self.closed = True


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Unmocked network')
    monkeypatch.setattr(requests.Session, 'get', forbidden)


def source_call(monkeypatch, response, name='web_search', **arguments):
    attempts = []
    def get(session, url, **kwargs):
        attempts.append(url)
        if isinstance(response, Exception):
            raise response
        return response
    monkeypatch.setattr(requests.Session, 'get', get)
    result = json.loads(ToolRegistry(extra_search=True).run(name, arguments or {'query': 'nebula'}))
    assert len(attempts) == 1  # no hidden retries
    assert 'sensitive' not in json.dumps(result)
    return result


@pytest.mark.parametrize('status,retryable', [(403, False), (404, False), (429, True), (503, True)])
def test_http_status_is_kept_without_guessing_why_server_refused(monkeypatch, status, retryable):
    response = Response('', status)
    result = source_call(monkeypatch, response)
    assert result['diagnostic'] == {'version': 'source-diagnostics-v1', 'stage': 'search',
                                    'outcome': 'error', 'error_kind': 'http_error',
                                    'http_status': status, 'retryable': retryable}
    assert response.closed


@pytest.mark.parametrize('error,kind', [(requests.ReadTimeout('sensitive URL'), 'timeout'),
                                      (requests.ConnectionError('sensitive headers'), 'connection_error'),
                                      (requests.exceptions.SSLError('sensitive cert'), 'tls_error')])
def test_transport_errors_remain_distinct_and_do_not_guess_http_status(monkeypatch, error, kind):
    result = source_call(monkeypatch, error)
    assert result['diagnostic']['error_kind'] == kind
    assert result['diagnostic']['http_status'] is None


@pytest.mark.parametrize('body,outcome', [({'query': {'search': []}}, 'empty'),
                                        ({'query': {'search': [{'title': 'Nebula', 'snippet': 'Observed.'}]}}, 'ok')])
def test_search_success_and_empty_are_observations_not_fact_grades(monkeypatch, body, outcome):
    result = source_call(monkeypatch, Response(body))
    assert result['diagnostic']['outcome'] == outcome
    assert result['diagnostic']['http_status'] == 200
    assert 'correct' not in result


@pytest.mark.parametrize('response,kind', [(Response('<html>oops</html>'), 'invalid_response'),
                                         (Response({'query': []}), 'invalid_response'),
                                         (Response({'query': {'search': 'oops'}}), 'invalid_response'),
                                         (Response('x' * (MAX_RESPONSE_BYTES + 1)), 'response_too_large'),
                                         (Response('', 302), 'redirect_blocked')])
def test_response_failures_are_not_reported_as_empty_search(monkeypatch, response, kind):
    result = source_call(monkeypatch, response)
    assert result['diagnostic']['outcome'] == 'error'
    assert result['diagnostic']['error_kind'] == kind
    assert result['diagnostic']['http_status'] == response.status_code
    assert response.closed


def test_dns_failure_has_its_own_stage_and_makes_no_http_request(monkeypatch):
    def failure(*args, **kwargs):
        raise socket.gaierror('sensitive hostname')
    monkeypatch.setattr(socket, 'getaddrinfo', failure)
    result = json.loads(ToolRegistry().run('page_find', {'url': 'https://example.org/article'}))
    assert result['diagnostic']['stage'] == 'resolve'
    assert result['diagnostic']['error_kind'] == 'dns_error'
    assert result['diagnostic']['http_status'] is None
    assert 'sensitive' not in json.dumps(result)


def test_invalid_arguments_and_unsafe_urls_never_become_transport_errors():
    registry = ToolRegistry()
    for args, kind in [({'title': 'Nebula', 'keywords': 'wrong'}, 'invalid_arguments'),
                       ({'url': 'file:///private'}, 'blocked_url')]:
        result = json.loads(registry.run('page_find', args))
        assert result['diagnostic']['stage'] == 'validation'
        assert result['diagnostic']['error_kind'] == kind


def test_page_content_and_unmatched_keywords_are_separate(monkeypatch):
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *a, **k: [(2, 1, 6, '', ('93.184.216.34', 443))])
    result = source_call(monkeypatch, Response('<main><p>Nebula observations</p></main>', content_type='text/html'),
                         'page_find', url='https://example.org/article', keywords=['missing'])
    assert result['found'] and not result['keywords_matched']
    assert result['diagnostic']['outcome'] == 'ok'
    assert result['diagnostic']['stage'] == 'extract'


def test_unsupported_content_is_a_loading_success_but_extraction_failure(monkeypatch):
    monkeypatch.setattr(socket, 'getaddrinfo', lambda *a, **k: [(2, 1, 6, '', ('93.184.216.34', 443))])
    result = source_call(monkeypatch, Response('PDF', content_type='application/pdf'),
                         'page_find', url='https://example.org/article')
    assert result['diagnostic']['stage'] == 'extract'
    assert result['diagnostic']['error_kind'] == 'unsupported_content'
    assert result['diagnostic']['http_status'] == 200


def test_search_challenge_is_not_zero_results(monkeypatch):
    result = source_call(monkeypatch, Response('<form id="challenge-form"></form>', content_type='text/html'),
                         'web_search_extra', query='nebula')
    assert result['diagnostic']['error_kind'] == 'challenge'
    assert result['diagnostic']['http_status'] == 200


def test_valid_text_mentioning_a_scientific_challenge_is_not_blocked(monkeypatch):
    result = source_call(monkeypatch, Response('<p>Scientific challenge. No results.</p>', content_type='text/html'),
                         'web_search_extra', query='nebula')
    assert result['diagnostic']['outcome'] == 'empty'


def test_oversized_tool_result_keeps_a_complete_diagnostic(monkeypatch):
    # Valid long titles become much longer when encoded as URL paths.
    items = [{'title': '星' * 500, 'snippet': 'text' * 225} for _ in range(5)]
    result = source_call(monkeypatch, Response({'query': {'search': items}}))
    assert result['diagnostic']['error_kind'] == 'output_truncated'
    assert result['diagnostic']['stage'] == 'control'
    assert result['diagnostic']['http_status'] == 200
    assert result['truncated'] is True


@pytest.mark.parametrize('body', [{}, {'query': {'pages': []}}, {'query': {'pages': {'1': {}}}}])
def test_malformed_wiki_page_is_not_reported_as_missing(monkeypatch, body):
    result = source_call(monkeypatch, Response(body), 'page_find', title='Nebula')
    assert result['diagnostic']['outcome'] == 'error'
    assert result['diagnostic']['error_kind'] == 'invalid_response'
    assert result['diagnostic']['http_status'] == 200


def test_page_api_error_is_fetch_failure_not_search_failure(monkeypatch):
    result = source_call(monkeypatch, Response({'error': {'info': 'sensitive server detail'}}),
                         'page_find', title='Nebula')
    assert result['diagnostic']['stage'] == 'fetch'
    assert result['diagnostic']['error_kind'] == 'api_error'
    assert result['diagnostic']['http_status'] == 200
