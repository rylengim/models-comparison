import numpy as np
import pytest

from hw2 import embeddings


def test_cache_preserves_order_and_separates_role_and_revision(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(embeddings, "CACHE_ROOT", tmp_path)

    def fake_encoder(texts):
        calls.append(texts)
        return np.stack([np.full(embeddings.DIMENSION, len(text), dtype=np.float32) for text in texts])

    monkeypatch.setattr(embeddings, "encode_batch", fake_encoder)
    first = embeddings.encode_texts(["short", "a", "short"], "passage")
    assert first[:, 0].tolist() == [5, 1, 5]
    assert calls == [["a", "short"]]
    np.testing.assert_array_equal(embeddings.encode_texts(["a", "short"], "passage"), first[[1, 0]])
    assert calls == [["a", "short"]]
    embeddings.encode_texts(["a"], "query")
    monkeypatch.setattr(embeddings, "MODEL_REVISION", "another-revision")
    embeddings.encode_texts(["a"], "query")
    assert calls == [["a", "short"], ["a"], ["a"]]
    assert embeddings.encode_texts([], "passage").shape == (0, embeddings.DIMENSION)
    with pytest.raises(ValueError, match="role"):
        embeddings.encode_texts(["a"], "other")
