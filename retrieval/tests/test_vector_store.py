import numpy as np
from app.vector_store import InMemoryVectorStore


def test_empty_store_returns_no_results():
    store = InMemoryVectorStore()
    results = store.search(np.zeros(4), top_k=3)
    assert results == []


def test_search_returns_most_similar_first():
    store = InMemoryVectorStore()
    store.add(
        ids=["a", "b", "c"],
        vectors=np.array([[1, 0], [0, 1], [0.9, 0.1]]),
        payloads=[{"text": "a"}, {"text": "b"}, {"text": "c"}],
    )
    results = store.search(np.array([1, 0]), top_k=2)
    assert results[0]["id"] == "a"  # exact match should rank first
    assert results[1]["id"] == "c"  # closest alternative should rank second


def test_top_k_is_capped_to_available_documents():
    store = InMemoryVectorStore()
    store.add(ids=["a"], vectors=np.array([[1, 0]]), payloads=[{"text": "a"}])
    results = store.search(np.array([1, 0]), top_k=5)
    assert len(results) == 1
