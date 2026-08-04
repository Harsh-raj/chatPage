import numpy as np
from app.embeddings import DeterministicHashEmbedder


def test_embedding_is_deterministic():
    embedder = DeterministicHashEmbedder(dim=32)
    v1 = embedder.embed(["hello world"])
    v2 = embedder.embed(["hello world"])
    np.testing.assert_array_equal(v1, v2)


def test_different_text_gives_different_embedding():
    embedder = DeterministicHashEmbedder(dim=32)
    v1 = embedder.embed(["hello"])
    v2 = embedder.embed(["goodbye"])
    assert not np.array_equal(v1, v2)


def test_embedding_is_unit_normalized():
    embedder = DeterministicHashEmbedder(dim=32)
    v = embedder.embed(["some text"])[0]
    assert abs(np.linalg.norm(v) - 1.0) < 1e-6
