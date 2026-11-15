"""Offline source scope: only broad official hosts, never question-to-answer hints."""
import pytest

from freshqa.official_sources import official_domains, official_query, filter_official_results


def hit(url, **extra):
    return {'title': 'Discovery', 'url': url, 'snippet': 'A published observation.', **extra}


def test_catalog_has_only_the_five_shared_publisher_domains():
    assert set(official_domains()) == {
        'esahubble.org', 'almaobservatory.org', 'esawebb.org', 'chandra.si.edu', 'eso.org'}
    assert official_domains('eso') == ('eso.org',)
    assert all('/' not in domain for domain in official_domains())


def test_query_is_bounded_and_uses_only_explicit_question_terms_and_shared_scope():
    query = official_query('galaxy observations 2026')
    assert 'galaxy observations 2026' in query
    assert all(f'site:{host}' in query for host in official_domains())
    assert official_query('galaxy observations', 'webb') == '(galaxy observations) (site:esawebb.org)'
    assert len(official_query('x' * 500)) <= 700


@pytest.mark.parametrize('query,source', [('', 'all'), ('  ', 'all'), ('x' * 501, 'all'),
                                         (None, 'all'), ('galaxy', 'unlisted')])
def test_invalid_query_or_source_is_not_silently_broadened(query, source):
    with pytest.raises(ValueError):
        official_query(query, source)


@pytest.mark.parametrize('url', [
    'https://www.eso.org.evil.example/article', 'https://evil.eso.org/article',
    'https://www.eso.org@evil.example/article', 'https://user@www.eso.org/article',
    'https://user:password@www.eso.org/article', 'http://127.0.0.1/article',
    'file:///eso.org/article', 'https://www.eso.org:444/article',
    'https://www.eso.org/article\n', 'https://www.eso.org/' + 'x' * 2000,
])
def test_scope_rejects_lookalikes_credentials_nonpublic_schemes_and_invalid_urls(url):
    assert filter_official_results([hit(url)]) == []


def test_filter_accepts_exact_hosts_and_www_aliases_but_checks_selected_source():
    results = [hit('https://www.eso.org/article'), hit('https://esahubble.org/article')]
    assert len(filter_official_results(results)) == 2
    assert filter_official_results(results, 'hubble') == [results[1]]


def test_filter_runs_before_result_limit_and_deduplicates_without_leaking_extra_fields():
    results = [hit(f'https://irrelevant.example/{i}') for i in range(6)]
    valid = hit('https://www.eso.org/article', arbitrary_metadata='do not retain')
    results += [valid, valid, hit('https://www.eso.org/article#section')]
    results += [hit(f'https://esahubble.org/article/{i}') for i in range(10)]
    filtered = filter_official_results(results)
    assert len(filtered) == 5
    assert filtered[0] == hit('https://www.eso.org/article')
    assert len({r['url'] for r in filtered}) == 5


def test_malformed_search_results_do_not_become_source_pages():
    assert filter_official_results([None, {}, {'url': 123}, hit('https://www.eso.org/a', title=[])]) == []
