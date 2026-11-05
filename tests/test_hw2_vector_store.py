import pytest

pytest.importorskip("pymilvus")
pytest.importorskip("milvus_lite")
from hw2.vector_store import index_chunks, open_store, search_many


def test_local_reopen_reindex_and_filter(tmp_path):
    uri = tmp_path / "chunks.db"
    chunks = [
        {"id": f"row{i}", "page": page, "url": f"wiki/{i}", "section": "Events",
         "text": "Example", "start": 0, "end": 7}
        for i, page in enumerate(["First", 'Document "B"'])
    ]
    vectors = [[1.0, 0.0], [0.0, 1.0]]
    store = open_store(uri)
    try:
        original = index_chunks(store, "char", chunks, vectors)
        assert search_many(store, "char", vectors, limit=1) == [["row0"], ["row1"]]
        assert search_many(store, "char", vectors[:1], limit=5, page_filter='Document "B"') == [["row1"]]
    finally:
        store.close()

    store = open_store(uri)
    try:
        assert index_chunks(store, "char", chunks, vectors) == original
        assert index_chunks(store, "char", chunks, vectors[::-1]) != original
        assert search_many(store, "char", vectors[:1], limit=1) == [["row1"]]
    finally:
        store.close()


def test_partial_index_can_resume_without_duplicate_rows(tmp_path, monkeypatch):
    from hw2 import vector_store

    monkeypatch.setattr(vector_store, "INSERT_BATCH", 1)
    chunks = [
        {"id": f"row{i}", "page": "Page", "url": "wiki", "section": "Events",
         "text": str(i), "start": i, "end": i + 1}
        for i in range(2)
    ]
    vectors = [[1., 0.], [0., 1.]]
    store = open_store(tmp_path / "partial.db")
    try:
        upsert = store.client.upsert

        def fail_second_batch(name, records):
            if records[0]["id"] == "row1":
                raise ConnectionError("simulated interruption")
            return upsert(name, records)

        with monkeypatch.context() as patch:
            patch.setattr(store.client, "upsert", fail_second_batch)
            with pytest.raises(ConnectionError):
                index_chunks(store, "char", chunks, vectors)
        assert "char" not in store.collections

        name = index_chunks(store, "char", chunks, vectors)
        count = store.client.query(name, filter="", output_fields=["count(*)"])[0]["count(*)"]
        assert count == len(chunks)
        assert search_many(store, "char", vectors, limit=1) == [["row0"], ["row1"]]
    finally:
        store.close()
