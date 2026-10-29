"""Offline behavior/security tests for every agent tool."""
import json
import socket

import pytest
import requests

from freshqa.tools import ToolRegistry


class Response:
    def __init__(self, body, *, status=200, content_type="application/json"):
        self.body = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        self.status_code = status
        self.headers = {"Content-Type": content_type}
        self.encoding = "utf-8"

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size):
        yield self.body

    def close(self):
        pass


@pytest.fixture
def registry():
    return ToolRegistry(extra_search=True)


@pytest.fixture(autouse=True)
def no_unmocked_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Offline tests must mock HTTP requests")
    monkeypatch.setattr(requests.Session, "get", forbidden)


def patch_http(monkeypatch, response):
    calls = []

    def get(_session, url, **kwargs):
        calls.append((url, kwargs))
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(requests.Session, "get", get)
    return calls


def run(registry, name, **kwargs):
    return json.loads(registry.run(name, kwargs))


def test_schemas_expose_described_pydantic_parameters(registry):
    schemas = registry.schemas(["web_search", "calculator"])
    assert [s["function"]["name"] for s in schemas] == ["web_search", "calculator"]
    assert schemas[0]["function"]["parameters"]["properties"]["query"]["description"]
    assert schemas[1]["function"]["parameters"]["additionalProperties"] is False


def test_wikipedia_search_normal(registry, monkeypatch):
    calls = patch_http(monkeypatch, Response({"query": {"search": [
        {"title": "Example", "snippet": "A <span>useful</span> result."}
    ]}}))
    result = run(registry, "web_search", query="example", language="ru")
    assert result["results"][0]["snippet"] == "A useful result."
    assert "ru.wikipedia.org" in result["results"][0]["url"]
    assert calls[0][1]["allow_redirects"] is False


def test_wikipedia_search_empty(registry, monkeypatch):
    patch_http(monkeypatch, Response({"query": {"search": []}}))
    assert run(registry, "web_search", query="missing")["results"] == []


def test_wikipedia_search_network_error(registry, monkeypatch):
    patch_http(monkeypatch, requests.Timeout("unavailable"))
    assert "error" in run(registry, "web_search", query="example")


def test_page_find_reads_beyond_intro(registry, monkeypatch):
    article = "Old introduction.\n\n" + "Historical context. " * 150 + "\n\nIn 2026 the new champion was Alice."
    calls = patch_http(monkeypatch, Response({"query": {"pages": {"1": {
        "title": "Championship", "extract": article
    }}}}))
    result = run(registry, "page_find", title="Championship", keywords=["2026", "Alice"])
    assert "Alice" in " ".join(result["excerpts"])
    assert "exintro" not in calls[0][1]["params"]


def test_page_find_missing_page(registry, monkeypatch):
    patch_http(monkeypatch, Response({"query": {"pages": {"-1": {"missing": ""}}}}))
    assert run(registry, "page_find", title="Missing")["found"] is False


def test_page_find_http_error(registry, monkeypatch):
    patch_http(monkeypatch, Response({}, status=503))
    assert "error" in run(registry, "page_find", title="Example")


def test_page_find_public_html(registry, monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 443))])
    patch_http(monkeypatch, Response("<html><script>bad()</script><main><p>Winner 2026: Alice.</p></main></html>", content_type="text/html"))
    result = run(registry, "page_find", url="https://example.org/results", keywords=["2026"])
    assert "Alice" in " ".join(result["excerpts"])
    assert "bad()" not in " ".join(result["excerpts"])


@pytest.mark.parametrize("url", ["file:///etc/passwd", "http://127.0.0.1", "http://[::1]", "http://169.254.169.254/latest", "http://user:pass@example.org", "http://localhost"])
def test_page_find_rejects_unsafe_urls(registry, url):
    assert "error" in run(registry, "page_find", url=url)


def test_page_find_rejects_private_dns(registry, monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("10.0.0.5", 80))])
    assert "error" in run(registry, "page_find", url="http://private.example")


def test_page_find_rejects_redirect(registry, monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 443))])
    patch_http(monkeypatch, Response("", status=302, content_type="text/html"))
    assert "error" in run(registry, "page_find", url="https://example.org")


def test_calculator_normal(registry):
    assert run(registry, "calculator", expression="(125 - 25) * 1.2 / 3")["result"] == 40


@pytest.mark.parametrize("expression", ["", "1 / 0", "open('/etc/passwd')", "2 ** 1000000000"])
def test_calculator_empty_or_error(registry, expression):
    assert "error" in run(registry, "calculator", expression=expression)


def test_python_exec_normal(registry):
    result = run(registry, "python_exec", code="values = [4, 8, 12]\nprint(sum(values) / len(values))")
    assert result["stdout"].strip() == "8.0"


def test_python_exec_no_output(registry):
    assert run(registry, "python_exec", code="x = 2 + 2")["stdout"] == ""


@pytest.mark.parametrize("code", ["", "import os", "print((1).__class__)", "open('/etc/passwd')", "print(__builtins__)", "print(1 / 0)", "print = 1", "print('x')"])
def test_python_exec_empty_or_error(registry, code):
    assert "error" in run(registry, "python_exec", code=code)


def test_python_timeout_is_reported(registry, monkeypatch):
    import subprocess

    def timed_out(*args, **kwargs):
        raise subprocess.TimeoutExpired("python", 2)

    monkeypatch.setattr(subprocess, "run", timed_out)
    assert "timeout" in run(registry, "python_exec", code="print(1)")["error"].lower()


def test_python_child_does_not_inherit_secrets(registry, monkeypatch):
    import subprocess

    original = subprocess.run
    captured = {}

    def capture(*args, **kwargs):
        captured.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setenv("HOMEWORK_TEST_SECRET", "do-not-inherit")
    monkeypatch.setattr(subprocess, "run", capture)
    assert run(registry, "python_exec", code="print(1)")["stdout"].strip() == "1"
    assert "HOMEWORK_TEST_SECRET" not in captured["env"]
    assert "-I" in captured["args"] if "args" in captured else True


def test_extra_search_normal(registry, monkeypatch):
    html = '<div class="result"><a class="result__a" href="https://example.org">Example</a><div class="result__snippet">Fresh answer.</div></div>'
    patch_http(monkeypatch, Response(html, content_type="text/html"))
    result = run(registry, "web_search_extra", query="fresh answer")
    assert result["results"][0]["url"] == "https://example.org"


def test_extra_search_empty(registry, monkeypatch):
    patch_http(monkeypatch, Response("<html>No results</html>", content_type="text/html"))
    assert run(registry, "web_search_extra", query="missing")["results"] == []


def test_extra_search_error(registry, monkeypatch):
    patch_http(monkeypatch, requests.ConnectionError("offline"))
    assert "error" in run(registry, "web_search_extra", query="example")


def test_extra_search_is_opt_in():
    assert "error" in run(ToolRegistry(), "web_search_extra", query="example")


@pytest.mark.parametrize("name,arguments", [("unknown", {}), ("web_search", {"query": "  "}), ("web_search", {"query": "a", "language": "de"}), ("calculator", {"expression": "2+2", "extra": True})])
def test_registry_returns_validation_errors(registry, name, arguments):
    assert "error" in json.loads(registry.run(name, arguments))


def test_public_page_connects_to_checked_ip(registry, monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", ("93.184.216.34", 443))])
    calls = patch_http(monkeypatch, Response("<p>Public source</p>", content_type="text/html"))
    assert run(registry, "page_find", url="https://example.org/article")["found"]
    assert calls[0][0] == "https://93.184.216.34:443/article"
    assert calls[0][1]["headers"]["Host"] == "example.org"


def test_source_download_limit(registry, monkeypatch):
    patch_http(monkeypatch, Response("x" * 2_000_001))
    assert "limit" in run(registry, "web_search", query="example")["error"]


def test_python_output_limit(registry):
    assert "error" in run(registry, "python_exec", code="print(list(range(5000)))")


def test_malformed_api_response_is_reported(registry, monkeypatch):
    patch_http(monkeypatch, Response([]))
    assert "error" in run(registry, "web_search", query="example")


@pytest.mark.parametrize("address", ["224.0.0.1", "ff02::1", "fec0::1"])
def test_public_page_rejects_non_unicast_or_site_local(registry, monkeypatch, address):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(2, 1, 6, "", (address, 80))])
    patch_http(monkeypatch, Response("<p>Should not be read</p>", content_type="text/html"))
    assert "error" in run(registry, "page_find", url="http://unsafe.example")
