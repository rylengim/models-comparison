import json

import pytest
import requests

from freshqa.tools import ToolRegistry
from hw3.snapshot_tools import SnapshotRegistry


ROWS = [
    {"page": "Mars mission", "url": "https://en.wikipedia.org/wiki/Mars_mission",
     "text": "The Mars mission launched on 15 March. The spacecraft reached Mars in June.",
     "snapshot_date": "2026-09-01"},
    {"page": "Venus mission", "url": "https://en.wikipedia.org/wiki/Venus_mission",
     "text": "The Venus mission launched in May. Venus observations use radar."},
    {"page": "Space agency announcement", "url": "https://example.org/space/mars",
     "text": "The agency confirmed the Mars mission launch on 15 March."},
]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Snapshot tools must never use the network")
    monkeypatch.setattr(requests.Session, "request", fail)


@pytest.fixture
def registry():
    instance = SnapshotRegistry(rows=ROWS)
    yield instance
    instance.close()


def test_all_four_schemas_are_identical_to_hw1(registry):
    original = ToolRegistry()
    try:
        names = ["web_search", "page_find", "calculator", "python_exec"]
        assert registry.schemas(names) == original.schemas(names)
    finally:
        original.http.close()


def test_search_uses_snapshot_and_returns_relevant_excerpt(registry):
    output = json.loads(registry.run("web_search", {"query": "Mars mission"}))
    assert output["source"] == "Wikipedia local snapshot"
    assert output["results"][0]["title"] == "Mars mission"
    assert "15 March" in output["results"][0]["snippet"]
    assert "2026-09-01" in output["results"][0]["note"]


def test_exact_page_and_url_use_same_fulltext(registry):
    by_title = registry.page_find(title="MARS MISSION", keywords=["June"])
    by_url = registry.page_find(url="https://en.wikipedia.org/wiki/Mars_mission", keywords=["June"])
    assert by_title == by_url
    assert by_title["found"] and by_title["keywords_matched"]
    assert "June" in " ".join(by_title["excerpts"])


def test_missing_page_and_unmatched_search_do_not_fall_back_to_web(registry):
    assert registry.page_find(title="nonexistent")["found"] is False
    assert registry.page_find(url="https://example.org/not-saved")["found"] is False
    assert registry.web_search("gibberishunseenword")["results"] == []


def test_russian_requests_are_explicitly_unsupported(registry):
    assert registry.web_search("Mars", language="ru")["results"] == []
    result = registry.page_find(title="Mars mission", language="ru")
    assert result["found"] is False
    assert "English" in result["note"]


def test_gold_fields_never_enter_index_or_tool_output():
    rows = [{**ROWS[0], "question": "secretquestiontoken", "answer": "secretgoldtoken",
             "evidence": "secretevidencetoken"}]
    registry = SnapshotRegistry(rows=rows)
    try:
        assert registry.web_search("secretgoldtoken")["results"] == []
        assert registry.web_search("secretquestiontoken")["results"] == []
        assert registry.web_search("secretevidencetoken")["results"] == []
        output = json.dumps(registry.page_find(title="Mars mission"))
        assert "secret" not in output
    finally:
        registry.close()


def test_numeric_tool_remains_original_and_local(registry):
    assert json.loads(registry.run("calculator", {"expression": "6*7"})) == {"result": 42}


def test_loading_jsonl_file_and_search_limit(tmp_path):
    rows = [dict(page=f"Mars page {i}", url=f"https://example.org/{i}", text="Mars launch") for i in range(8)]
    path = tmp_path / "corpus.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    registry = SnapshotRegistry(corpus_path=path)
    try:
        assert len(registry.web_search("Mars")["results"]) == 5
        assert registry.web_search("Mars") == registry.web_search("Mars")
    finally:
        registry.close()
