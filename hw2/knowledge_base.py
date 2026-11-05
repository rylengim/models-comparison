"""Search the local homework corpus with the measured top-three RRF setup."""

from itertools import islice

from freshqa.dataset import load_rows
from hw2.embeddings import encode_texts
from hw2.evaluate_retrieval import CANDIDATE_DEPTH, reciprocal_rank_fusion
from hw2.lexical_baseline import bm25_index, rank
from hw2.prepare_corpus import DATA
from hw2.vector_store import DEFAULT_URI, index_chunks, open_store, search_many


class KnowledgeBase:
    """Keep one Milvus connection and one lexical index for an agent session."""

    def __init__(self, uri=DEFAULT_URI):
        path = DATA / "chunks_structure.jsonl"
        if not path.is_file():
            raise FileNotFoundError("Сначала выполните python3 -m hw2.chunk_corpus из корня проекта.")
        self._chunks = load_rows(path)
        self._by_id = {row["id"]: row for row in self._chunks}
        self._postings = bm25_index(self._chunks)
        self._store = open_store(uri)
        try:
            vectors = encode_texts([row["text"] for row in self._chunks], role="passage")
            index_chunks(self._store, "structure", self._chunks, vectors)
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        if self._store is not None:
            self._store.close()
            self._store = None

    def search(self, query: str, page: str | None = None) -> list[dict[str, str]]:
        """Return up to three source passages, optionally restricted to an exact page."""
        if self._store is None:
            raise RuntimeError("Соединение с базой знаний закрыто.")
        if not query.strip():
            raise ValueError("Поисковый запрос пуст.")

        vector = encode_texts([query], role="query")
        vector_ids, = search_many(self._store, "structure", vector, limit=CANDIDATE_DEPTH, page_filter=page)
        lexical = (self._chunks[i] for i in rank(query, self._postings))
        lexical_ids = (row["id"] for row in lexical if page is None or row["page"] == page)
        ranked_ids = reciprocal_rank_fusion(vector_ids, islice(lexical_ids, CANDIDATE_DEPTH))
        fields = ("id", "text", "page", "section", "url")
        return [{key: self._by_id[chunk_id][key] for key in fields} for chunk_id in ranked_ids[:3]]
