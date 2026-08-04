"""
Vector store abstraction.

InMemoryVectorStore is used for local dev and unit tests -- no external
services needed. QdrantVectorStore is the real production implementation,
same interface, swapped in via the USE_REAL_STORE environment variable.

Note: Qdrant only accepts point IDs that are unsigned integers or UUIDs --
arbitrary strings are rejected. QdrantVectorStore internally maps each
caller-supplied string ID to a deterministic UUID (so the same input ID
always maps to the same Qdrant point, making re-indexing idempotent),
while preserving the original ID in the payload so search()/list_all() can
return it unchanged to callers.
"""
from __future__ import annotations
import os
import uuid
import numpy as np


class VectorStore:
    def add(self, ids: list[str], vectors: np.ndarray, payloads: list[dict]) -> None:
        raise NotImplementedError

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        raise NotImplementedError

    def delete(self, ids: list[str]) -> int:
        """Deletes the given ids. Returns the number actually removed."""
        raise NotImplementedError

    def list_all(self) -> list[dict]:
        """Returns every stored item as {"id": ..., "metadata": ...} -- used
        to support deleting by source document rather than by exact id."""
        raise NotImplementedError

    def count(self) -> int:
        raise NotImplementedError


class InMemoryVectorStore(VectorStore):
    """Cosine-similarity search over an in-memory numpy array. Dev/test only."""

    def __init__(self):
        self._ids: list[str] = []
        self._vectors: np.ndarray | None = None
        self._payloads: list[dict] = []

    def add(self, ids: list[str], vectors: np.ndarray, payloads: list[dict]) -> None:
        vectors = np.asarray(vectors, dtype=float)
        if self._vectors is None:
            self._vectors = vectors
        else:
            self._vectors = np.vstack([self._vectors, vectors])
        self._ids.extend(ids)
        self._payloads.extend(payloads)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        if self._vectors is None or len(self._ids) == 0:
            return []

        q = np.asarray(query_vector, dtype=float).reshape(1, -1)
        norms = np.linalg.norm(self._vectors, axis=1) * np.linalg.norm(q)
        norms[norms == 0] = 1e-10
        scores = (self._vectors @ q.T).flatten() / norms

        top_k = min(top_k, len(self._ids))
        top_idx = np.argsort(-scores)[:top_k]

        return [
            {"id": self._ids[i], "score": float(scores[i]), **self._payloads[i]}
            for i in top_idx
        ]

    def delete(self, ids: list[str]) -> int:
        ids_to_remove = set(ids)
        keep_idx = [i for i, doc_id in enumerate(self._ids) if doc_id not in ids_to_remove]
        removed = len(self._ids) - len(keep_idx)

        self._ids = [self._ids[i] for i in keep_idx]
        self._payloads = [self._payloads[i] for i in keep_idx]
        self._vectors = self._vectors[keep_idx] if self._vectors is not None and keep_idx else None

        return removed

    def list_all(self) -> list[dict]:
        return [
            {"id": self._ids[i], "metadata": self._payloads[i].get("metadata", {})}
            for i in range(len(self._ids))
        ]

    def count(self) -> int:
        return len(self._ids)


def _to_qdrant_id(original_id: str) -> str:
    """Deterministically maps any string ID to a UUID that Qdrant will accept."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, original_id))


class QdrantVectorStore(VectorStore):
    """Real production vector store backed by Qdrant."""

    def __init__(
        self,
        collection_name: str = "documents",
        url: str | None = None,
        vector_size: int = 384,
    ):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams

        self.url = url or os.environ.get("QDRANT_URL", "http://localhost:6333")
        self.collection_name = collection_name
        self.client = QdrantClient(url=self.url)

        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def add(self, ids: list[str], vectors: np.ndarray, payloads: list[dict]) -> None:
        from qdrant_client.models import PointStruct

        vectors = np.asarray(vectors, dtype=float)
        points = [
            PointStruct(
                id=_to_qdrant_id(ids[i]),
                vector=vectors[i].tolist(),
                payload={**payloads[i], "_original_id": ids[i]},
            )
            for i in range(len(ids))
        ]
        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        query_vector = np.asarray(query_vector, dtype=float).flatten().tolist()
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
        )
        results = []
        for hit in response.points:
            payload = dict(hit.payload)
            original_id = payload.pop("_original_id", str(hit.id))
            results.append({"id": original_id, "score": float(hit.score), **payload})
        return results

    def delete(self, ids: list[str]) -> int:
        from qdrant_client.models import PointIdsList

        qdrant_ids = [_to_qdrant_id(i) for i in ids]
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=PointIdsList(points=qdrant_ids),
        )
        return len(ids)  # Qdrant's delete doesn't report how many actually existed

    def list_all(self) -> list[dict]:
        results = []
        next_offset = None
        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=200,
                offset=next_offset,
                with_payload=True,
            )
            for point in points:
                payload = dict(point.payload)
                original_id = payload.pop("_original_id", str(point.id))
                results.append({"id": original_id, "metadata": payload.get("metadata", {})})
            if next_offset is None:
                break
        return results

    def count(self) -> int:
        info = self.client.get_collection(self.collection_name)
        return info.points_count