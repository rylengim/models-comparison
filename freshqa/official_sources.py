"""A shared publisher scope, independent of questions, gold answers and article URLs.

The catalog describes the astronomy collection's publishers, not the article
answering any individual question. Search availability is measured separately.
"""
from typing import Literal
from urllib.parse import urlparse

OfficialSource = Literal['all', 'hubble', 'alma', 'webb', 'chandra', 'eso']
DOMAINS = {
    'hubble': 'esahubble.org',
    'alma': 'almaobservatory.org',
    'webb': 'esawebb.org',
    'chandra': 'chandra.si.edu',
    'eso': 'eso.org',
}


def official_domains(source: OfficialSource = 'all') -> tuple[str, ...]:
    if source == 'all':
        return tuple(DOMAINS.values())
    if not isinstance(source, str) or source not in DOMAINS:
        raise ValueError('Unknown official source scope')
    return (DOMAINS[source],)


def official_query(query: str, source: OfficialSource = 'all') -> str:
    domains = official_domains(source)
    if not isinstance(query, str) or not 1 <= len(query.strip()) <= 500:
        raise ValueError('Search query must contain 1–500 characters')
    return f"({query.strip()}) (" + ' OR '.join(f'site:{domain}' for domain in domains) + ')'


def filter_official_results(results, source: OfficialSource = 'all') -> list[dict]:
    """Apply host scope before the five-result limit; selected URLs are read separately.

    Only exact publishers and their www aliases qualify. Public-IP/TLS validation
    still happens when page_find reads a result; a matching host is not an SSRF bypass.
    """
    allowed = set(official_domains(source))
    selected, seen = [], set()
    for result in results:
        if not isinstance(result, dict) or any(not isinstance(result.get(k), str)
                                              for k in ('url', 'title', 'snippet')):
            continue
        url = result['url']
        if not 1 <= len(url) <= 2000 or any(ord(char) <= 32 or ord(char) == 127 for char in url):
            continue
        try:
            parsed = urlparse(url)
            host = (parsed.hostname or '').lower().rstrip('.').removeprefix('www.')
            if (parsed.scheme not in {'https', 'http'} or parsed.username is not None
                    or parsed.password is not None or parsed.port not in {None, 80, 443}
                    or host not in allowed):
                continue
        except ValueError:
            continue
        url = parsed._replace(fragment='').geturl()
        if url in seen:
            continue
        seen.add(url)
        selected.append({'title': result['title'][:500], 'url': url, 'snippet': result['snippet'][:800]})
        if len(selected) == 5:
            break
    return selected
