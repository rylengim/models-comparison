from datetime import datetime

import pytest

pytest.importorskip("pymilvus")
pytest.importorskip("milvus_lite")
from hw2.embeddings import DIMENSION
from hw2.memory_store import EpisodeJournal, FactStore


def encode_for_test(texts, role):
    """Separate city facts from diet facts without model weights or a network call."""
    return [[float("city" in text.lower()), float("diet" in text.lower()), 0.1]
            + [0.] * (DIMENSION - 3) for text in texts]


def facts_at(path, user="demo"):
    return FactStore(user, uri=path, encoder=encode_for_test)


def test_empty_memory(tmp_path):
    with facts_at(tmp_path / "facts.db") as facts:
        assert facts.search("Which city?") == []
        assert facts.all() == []


def test_finds_relevant_fact_across_sessions_and_isolates_users(tmp_path):
    path = tmp_path / "facts.db"
    with facts_at(path) as facts:
        facts.upsert("first", [
            {"key": "diet", "value": "Vegetarian"},
            {"key": "home_city", "value": "Porto"},
        ])
    with facts_at(path, "another-user") as facts:
        assert facts.search("Which city?") == []
        facts.upsert("first", [{"key": "home_city", "value": "Lima"}])
    with facts_at(path) as facts:
        assert facts.search("Which city?", limit=1) == [
            {"key": "home_city", "value": "Porto", "session_id": "first"}
        ]


def test_new_value_replaces_old_even_after_retry(tmp_path):
    path = tmp_path / "facts.db"
    with facts_at(path) as facts:
        facts.upsert("first", [{"key": "home_city", "value": "Porto"}, {"key": "diet", "value": "Vegetarian"}])
        latest = [{"key": "home_city", "value": "Lima"}]
        facts.upsert("second", latest)
        facts.upsert("second", latest)
    with facts_at(path) as facts:
        assert facts.all() == [
            {"key": "diet", "value": "Vegetarian", "session_id": "first"},
            {"key": "home_city", "value": "Lima", "session_id": "second"},
        ]
        assert facts.search("Which city?", limit=1) == [{**latest[0], "session_id": "second"}]


def test_journal_reopens_and_selects_one_user_and_session(tmp_path):
    path = tmp_path / "journal.jsonl"
    journal = EpisodeJournal(path, "demo")
    assert journal.session("first") == []
    journal.append("first", "user", "Two\nlines")
    journal.append("first", "assistant", "OK")
    journal.append("second", "user", "A new session")
    EpisodeJournal(path, "other").append("first", "user", "Other user's session")
    rows = EpisodeJournal(path, "demo").session("first")
    assert [(r["role"], r["content"]) for r in rows] == [("user", "Two\nlines"), ("assistant", "OK")]
    assert all(r["session_id"] == "first" and r["user_id"] == "demo" for r in rows)
    assert all(datetime.fromisoformat(r["timestamp"]).utcoffset() is not None for r in rows)
