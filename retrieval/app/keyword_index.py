"""
Keyword (BM25) search index -- runs alongside dense vector search as the
other half of hybrid search. Pure Python, no GPU/heavy-model dependency,
so it's always "real" (no stub/real toggle needed, unlike the embedder).

Stores full text/metadata alongside each id (not just id+score) so that
results are shape-compatible with dense search results after fusion --
an item found ONLY via BM25 (not in the dense top-k) still carries its
full text, which the reranker requires. Without this, fusion could
produce candidates missing "text" entirely, causing a KeyError deep in
CrossEncoderReranker.rerank() whenever BM25 and dense search disagree on
which items belong in the candidate pool (increasingly likely as more
real documents are indexed).

Note: this index lives in-process and is rebuilt from scratch on service
restart. If you're using the persistent QdrantVectorStore, indexed vectors
survive restarts but this keyword index does not.
"""
from __future__ import annotations
from rank_bm25 import BM25Okapi


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


class KeywordIndex:
    def add(self, ids: list[str], texts: list[str], payloads: list[dict] | None = None) -> None:
        raise NotImplementedError

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        raise NotImplementedError

    def delete(self, ids: list[str]) -> int:
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError


class BM25KeywordIndex(KeywordIndex):
    def __init__(self):
        self._ids: list[str] = []
        self._tokenized_corpus: list[list[str]] = []
        self._payloads: list[dict] = []
        self._bm25: BM25Okapi | None = None

    def _rebuild(self) -> None:
        self._bm25 = BM25Okapi(self._tokenized_corpus) if self._tokenized_corpus else None

    def add(self, ids: list[str], texts: list[str], payloads: list[dict] | None = None) -> None:
        self._ids.extend(ids)
        self._tokenized_corpus.extend(_tokenize(t) for t in texts)
        if payloads is not None:
            self._payloads.extend(payloads)
        else:
            # Fallback: at minimum, preserve the text itself
            self._payloads.extend({"text": t} for t in texts)
        self._rebuild()

    def search(self, query: str, top_k: int = 20) -> list[dict]:
        if self._bm25 is None or not self._ids:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        top_k = min(top_k, len(self._ids))
        top_idx = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [
            {"id": self._ids[i], "score": float(scores[i]), **self._payloads[i]}
            for i in top_idx
        ]

    def delete(self, ids: list[str]) -> int:
        ids_to_remove = set(ids)
        keep_idx = [i for i, doc_id in enumerate(self._ids) if doc_id not in ids_to_remove]
        removed = len(self._ids) - len(keep_idx)

        self._ids = [self._ids[i] for i in keep_idx]
        self._tokenized_corpus = [self._tokenized_corpus[i] for i in keep_idx]
        self._payloads = [self._payloads[i] for i in keep_idx]
        self._rebuild()

        return removed

    def count(self) -> int:
        return len(self._ids)