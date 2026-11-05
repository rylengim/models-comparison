"""Locally encode questions and passages. Download only pinned, public model artifacts."""

import hashlib
import sqlite3
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen

import numpy as np


MODEL_ID = "Xenova/all-MiniLM-L6-v2"
MODEL_REVISION = "751bff37182d3f1213fa05d7196b954e230abad9"
DIMENSION = 384
CACHE_ROOT = Path(__file__).resolve().parents[1] / "tmp" / "hw2-embeddings"
MAX_TOKENS = 256
BATCH_SIZE = 32
ARTIFACTS = {
    "tokenizer.json": "da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0",
    "onnx/model.onnx": "759c3cd2b7fe7e93933ad23c4c9181b7396442a2ed746ec7c1d46192c469c46e",
}


def model_file(name):
    folder = CACHE_ROOT / MODEL_REVISION
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / Path(name).name
    if path.exists():
        with path.open("rb") as source:
            if hashlib.file_digest(source, "sha256").hexdigest() == ARTIFACTS[name]:
                return path

    url = f"https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{name}"
    output = NamedTemporaryFile(dir=folder, delete=False)
    pending = Path(output.name)
    try:
        digest = hashlib.sha256()
        with output, urlopen(url, timeout=90) as response:
            while block := response.read(1024 * 1024):
                digest.update(block)
                output.write(block)
        if digest.hexdigest() != ARTIFACTS[name]:
            raise ValueError(f"Checksum mismatch for {name} at revision {MODEL_REVISION}")
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)
    return path


@lru_cache(maxsize=1)
def load_model():
    import onnxruntime as ort
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(model_file("tokenizer.json")))
    tokenizer.enable_truncation(max_length=MAX_TOKENS)
    tokenizer.enable_padding()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_file("onnx/model.onnx")), options, providers=["CPUExecutionProvider"]
    )
    return tokenizer, session


def encode_batch(texts):
    tokenizer, session = load_model()
    encodings = tokenizer.encode_batch(texts)
    ids = np.asarray([e.ids for e in encodings], dtype=np.int64)
    mask = np.asarray([e.attention_mask for e in encodings], dtype=np.int64)
    feed = {"input_ids": ids, "token_type_ids": np.zeros_like(ids), "attention_mask": mask}
    hidden, = session.run(["last_hidden_state"], feed)
    weights = mask[:, :, None].astype(np.float32)
    mean = (hidden * weights).sum(axis=1) / weights.sum(axis=1)
    return mean / np.linalg.norm(mean, axis=1, keepdims=True)


def encode_texts(texts, role):
    """Return one float32 vector per text in input order, reusing SQLite cache entries."""
    if role not in {"query", "passage"}:
        raise ValueError("role must be query or passage")
    texts = list(texts)
    if not texts:
        return np.empty((0, DIMENSION), dtype=np.float32)
    # Include every model/preprocessing choice that changes the resulting vectors.
    prefix = f"{MODEL_ID}@{MODEL_REVISION}|fp32-mean-l2|{MAX_TOKENS}|{role}\0".encode()
    keys = [hashlib.sha256(prefix + text.encode()).hexdigest() for text in texts]
    unique = dict(zip(keys, texts))
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(CACHE_ROOT / "vectors.sqlite3", timeout=60)
    try:
        db.execute("CREATE TABLE IF NOT EXISTS vectors (key TEXT PRIMARY KEY, value BLOB NOT NULL)")
        cached = {}
        missing = []
        for key, text in unique.items():
            row = db.execute("SELECT value FROM vectors WHERE key = ?", (key,)).fetchone()
            if row is None:
                missing.append((key, text))
            else:
                cached[key] = np.frombuffer(row[0], dtype="<f4", count=DIMENSION)
        # Similar-length passages in each batch reduce computation on padding tokens.
        missing.sort(key=lambda item: len(item[1]))
        for start in range(0, len(missing), BATCH_SIZE):
            batch = missing[start:start + BATCH_SIZE]
            vectors = encode_batch([text for _, text in batch])
            with db:
                for (key, _), vector in zip(batch, vectors, strict=True):
                    cached[key] = vector
                    db.execute("INSERT OR IGNORE INTO vectors VALUES (?, ?)",
                               (key, vector.astype("<f4", copy=False).tobytes()))
    finally:
        db.close()
    return np.stack([cached[key] for key in keys])
