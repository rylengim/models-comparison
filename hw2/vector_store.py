"""Local Milvus collections for exact cosine search over the homework corpus."""
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
from pymilvus import DataType, MilvusClient


DEFAULT_URI = "http://127.0.0.1:19530"
LITE_URI = str(Path(__file__).resolve().parents[1] / "tmp" / "hw2-milvus.db")
METHODS = {"char", "structure"}
TEXT_FIELDS = {"id": 128, "page": 1024, "url": 4096, "section": 4096, "text": 8192}
INSERT_BATCH = 256


@dataclass
class ChunkStore:
    client: MilvusClient
    collections: dict[str, str] = field(default_factory=dict)

    def close(self):
        self.client.close()


def open_store(uri=DEFAULT_URI):
    uri = str(uri)
    if uri.endswith(".db") and "://" not in uri:
        path = Path(uri).resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        uri = str(path)
    else:
        url = urlsplit(uri)
        if url.scheme not in {"http", "https"} or url.hostname not in {"localhost", "127.0.0.1", "::1"}:
            raise ValueError("Use a loopback Milvus server or a local .db file.")
    return ChunkStore(MilvusClient(uri=uri, timeout=15))


def index_chunks(store, method, chunks, vectors):
    """Select a content-addressed collection, resuming an interrupted build by upsert."""
    if method not in METHODS:
        raise ValueError(f"Unknown chunking method: {method}")
    vectors = np.asarray(vectors, dtype="<f4", order="C")
    if vectors.ndim != 2 or not len(chunks) or vectors.shape[0] != len(chunks):
        raise ValueError("Expected a non-empty matrix with one vector per chunk.")
    if not np.isfinite(vectors).all() or np.any(np.linalg.norm(vectors, axis=1) == 0):
        raise ValueError("Vectors must be finite and non-zero for cosine search.")
    if len({chunk["id"] for chunk in chunks}) != len(chunks):
        raise ValueError("Chunk IDs must be unique within a collection.")
    digest = hashlib.sha256(vectors)
    for chunk in chunks:
        digest.update(json.dumps(chunk, sort_keys=True, ensure_ascii=False).encode())
        digest.update(b"\n")
    name = f"freshqa_v1_{method}_{digest.hexdigest()[:32]}"
    client = store.client
    if not client.has_collection(name):
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=False)
        for key, length in TEXT_FIELDS.items():
            schema.add_field(key, DataType.VARCHAR, max_length=length, is_primary=key == "id")
        for key in ("start", "end"):
            schema.add_field(key, DataType.INT64)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=vectors.shape[1])
        index = MilvusClient.prepare_index_params()
        index.add_index("vector", index_type="FLAT", metric_type="COSINE")
        client.create_collection(name, schema=schema, index_params=index, consistency_level="Strong")

    client.load_collection(name)
    count = client.query(name, filter="", output_fields=["count(*)"])[0]["count(*)"]
    if count != len(chunks):
        for offset in range(0, len(chunks), INSERT_BATCH):
            records = [
                {**chunk, "vector": vector.tolist()}
                for chunk, vector in zip(chunks[offset:offset + INSERT_BATCH],
                                         vectors[offset:offset + INSERT_BATCH], strict=True)
            ]
            client.upsert(name, records)
        client.flush(name)
        count = client.query(name, filter="", output_fields=["count(*)"])[0]["count(*)"]
        if count != len(chunks):
            raise RuntimeError(f"Incomplete Milvus collection {name}: {count}/{len(chunks)}")
    store.collections[method] = name
    return name


def search_many(store, method, query_vectors, limit=10, page_filter=None):
    """Return ranked chunk IDs for each query, optionally within a named article."""
    name = store.collections[method]
    expr = "" if page_filter is None else f"page == {json.dumps(page_filter, ensure_ascii=False)}"
    vectors = np.asarray(query_vectors, dtype=np.float32)
    if vectors.ndim != 2 or limit < 1:
        raise ValueError("Expected a query matrix and a positive result limit.")
    result = []
    for offset in range(0, len(vectors), 32):
        hits = store.client.search(
            name, vectors[offset:offset + 32].tolist(), anns_field="vector", limit=limit,
            filter=expr, search_params={"metric_type": "COSINE", "params": {}},
        )
        result.extend([[item["id"] for item in row] for row in hits])
    return result
