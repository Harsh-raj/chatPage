from __future__ import annotations

import hashlib

import numpy as np


class Embedder:
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class DeterministicHashEmbedder(Embedder):
    """Turns text into a reproducible pseudo-embedding via hashing. Dev/test only."""

    def __init__(self, dim: int = 64):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = np.zeros((len(texts), self.dim), dtype=float)
        for i, text in enumerate(texts):
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            repeated = (digest * ((self.dim // len(digest)) + 1))[: self.dim]
            vectors[i] = np.frombuffer(bytes(repeated), dtype=np.uint8).astype(float)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return vectors / norms


class RealEmbedder(Embedder):
    """
    Real semantic embeddings via sentence-transformers.
    Default model is bge-small-en-v1.5: 384-dim, strong retrieval track record,
    small enough to run on CPU at reasonable speed.
    """

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(vectors, dtype=float)
