"""Local conversation journal and per-user semantic facts for the homework agent."""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from pymilvus import DataType, MilvusClient

from hw2.embeddings import DIMENSION, encode_texts
from hw2.vector_store import DEFAULT_URI, open_store


FIELDS = ("key", "value", "session_id")
COLLECTION = "freshqa_memory_v1"


def checked_text(value, field, max_bytes):
    if not isinstance(value, str) or not value.strip() or len(value.encode()) > max_bytes:
        raise ValueError(f"{field} must be a nonempty string of at most {max_bytes} bytes.")
    return value


class EpisodeJournal:
    """Append-only JSONL. Reading a session is explicit and scoped to one user."""

    def __init__(self, path, user_id):
        self.path = Path(path)
        self.user_id = checked_text(user_id, "user_id", 256)

    def append(self, session_id, role, content):
        checked_text(session_id, "session_id", 256)
        if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
            raise ValueError("A journal entry needs a user or assistant role and nonempty text.")
        entry = {"timestamp": datetime.now(timezone.utc).isoformat(), "user_id": self.user_id,
                 "session_id": session_id, "role": role, "content": content}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def session(self, session_id):
        checked_text(session_id, "session_id", 256)
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as stream:
            return [row for row in map(json.loads, stream)
                    if row["user_id"] == self.user_id and row["session_id"] == session_id]


class FactStore:
    """Keep the latest value of each stable fact key in a separate Milvus collection."""

    def __init__(self, user_id, *, uri=DEFAULT_URI, collection=COLLECTION, encoder=encode_texts):
        self.user_id = checked_text(user_id, "user_id", 256)
        self.collection = collection
        self.encoder = encoder
        self.store = open_store(uri)
        self.client = self.store.client
        try:
            if not self.client.has_collection(collection):
                schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
                for field, size in [("id", 64), ("user_id", 256), ("key", 128),
                                    ("value", 8192), ("session_id", 256)]:
                    schema.add_field(field, DataType.VARCHAR, max_length=size, is_primary=field == "id")
                schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIMENSION)
                indexes = MilvusClient.prepare_index_params()
                indexes.add_index("vector", index_type="FLAT", metric_type="COSINE")
                self.client.create_collection(collection, schema=schema, index_params=indexes,
                                              consistency_level="Strong")
            self.client.load_collection(collection)
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def close(self):
        self.store.close()

    def upsert(self, session_id, facts):
        """Upsert the session's latest facts in one batch. Repeated calls are idempotent."""
        checked_text(session_id, "session_id", 256)
        records = []
        seen = set()
        for fact in facts:
            key = checked_text(fact["key"], "key", 128)
            if not re.fullmatch(r"[a-z][a-z0-9_]*", key) or key in seen:
                raise ValueError("Expected one fact per lowercase snake_case key.")
            seen.add(key)
            value = checked_text(fact["value"], "value", 8192)
            identity = json.dumps([self.user_id, key], ensure_ascii=False).encode()
            records.append({"id": hashlib.sha256(identity).hexdigest(), "user_id": self.user_id,
                            "key": key, "value": value, "session_id": session_id})
        if records:
            texts = [f"{r['key'].replace('_', ' ')}: {r['value']}" for r in records]
            vectors = np.asarray(self.encoder(texts, role="passage"), dtype=np.float32)
            for record, vector in zip(records, vectors.tolist(), strict=True):
                record["vector"] = vector
            self.client.upsert(self.collection, records)

    def search(self, query, limit=3):
        if not isinstance(query, str) or not query.strip() or type(limit) is not int or limit <= 0:
            raise ValueError("Fact search requires a nonempty query and a positive integer limit.")
        vectors = np.asarray(self.encoder([query], role="query"), dtype=np.float32).tolist()
        hits, = self.client.search(self.collection, vectors, anns_field="vector", limit=limit,
                                   filter=self._user_filter(), output_fields=list(FIELDS),
                                   search_params={"metric_type": "COSINE", "params": {}})
        return [{key: hit["entity"][key] for key in FIELDS} for hit in hits]

    def all(self):
        rows = self.client.query(self.collection, filter=self._user_filter(), output_fields=list(FIELDS))
        return sorted(({k: row[k] for k in FIELDS} for row in rows), key=lambda row: row["key"])

    def _user_filter(self):
        return f"user_id == {json.dumps(self.user_id, ensure_ascii=False)}"
