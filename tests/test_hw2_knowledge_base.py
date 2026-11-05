import json
from types import SimpleNamespace

import pytest

pytest.importorskip("pymilvus")
from hw2 import knowledge_base
from hw2.rag import rag_answer


def chunk(chunk_id, page, text):
    return {"id": chunk_id, "page": page, "text": text, "section": "January",
            "url": f"https://example.org/{page}"}


def local_search(tmp_path, monkeypatch, chunks, vector_search):
    """Run the production BM25 and fusion code; stub the encoder and Milvus transport."""
    path = tmp_path / "chunks_structure.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in chunks), encoding="utf-8")
    monkeypatch.setattr(knowledge_base, "DATA", tmp_path)
    monkeypatch.setattr(knowledge_base, "open_store", lambda uri: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(knowledge_base, "index_chunks", lambda *args: None)
    monkeypatch.setattr(knowledge_base, "encode_texts", lambda texts, role: texts)
    monkeypatch.setattr(knowledge_base, "search_many", vector_search)
    return knowledge_base.KnowledgeBase()


@pytest.mark.parametrize(("question", "expected"), [
    ("How many job cuts did Statistics Canada announce?", "850"),
    ("How many of those Statistics Canada job cuts were in IT?", "unknown"),
])
def test_total_and_division_use_the_same_retrieved_evidence(tmp_path, monkeypatch, question, expected):
    canada = chunk("ca", "Canada", "January 13 – Statistics Canada announces 850 job cuts.")
    payload = {"answer": expected}
    if expected != "unknown":
        payload.update(source_id="ca", quote=canada["text"])
    client = SimpleNamespace(complete=lambda **_: {"message": {"role": "assistant", "content": json.dumps(payload)}})
    with local_search(tmp_path, monkeypatch, [canada], lambda *args, **kwargs: [["ca"]]) as kb:
        result = rag_answer(question, knowledge_base=kb, client=client, model="scripted-test")
    assert result["retrieved_ids"] == ["ca"]
    assert result["answer"] == expected
    if expected == "unknown":
        assert result["status"] == "insufficient_evidence"
        assert result["source"] is None and result["quote"] is None
    else:
        assert result["source"]["url"] == canada["url"]
        assert result["quote"] == canada["text"]


def test_page_filters_both_rankings_before_candidate_limit(tmp_path, monkeypatch):
    # Equal BM25 scores put 100 outside rows ahead of the two in-scope rows.
    outside = [chunk(f"other-{i}", "Other", "job cuts") for i in range(100)]
    inside = [chunk(i, "Canada", "job cuts") for i in ("vector-hit", "lexical-only")]

    def vector_search(store, method, vectors, limit, page_filter):
        assert method == "structure" and limit == 100 and page_filter == "Canada"
        return [["vector-hit"]]

    with local_search(tmp_path, monkeypatch, outside + inside, vector_search) as kb:
        hits = kb.search("job cuts", page="Canada")
    assert [row["id"] for row in hits] == ["vector-hit", "lexical-only"]
    assert all(row["page"] == "Canada" for row in hits)
