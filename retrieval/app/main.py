import os
from collections import Counter
from fastapi import FastAPI, HTTPException

from app.schema import (
    IndexRequest, SearchRequest, SearchResponse, SearchResult,
    DeleteByIdsRequest, DeleteResponse, ListDocumentsResponse, DocumentSummary,
)
from app.vector_store import InMemoryVectorStore, QdrantVectorStore
from app.embeddings import DeterministicHashEmbedder, RealEmbedder
from app.keyword_index import BM25KeywordIndex
from app.fusion import reciprocal_rank_fusion
from app.reranker import NoOpReranker, CrossEncoderReranker

app = FastAPI(title="retrieval-service")

if os.environ.get("USE_REAL_EMBEDDER") == "1":
    embedder = RealEmbedder()
else:
    embedder = DeterministicHashEmbedder(dim=64)

if os.environ.get("USE_REAL_STORE") == "1":
    store = QdrantVectorStore(vector_size=embedder.dim if hasattr(embedder, "dim") else 64)
else:
    store = InMemoryVectorStore()

keyword_index = BM25KeywordIndex()

if os.environ.get("USE_RERANKER") == "1":
    reranker = CrossEncoderReranker()
else:
    reranker = NoOpReranker()

CANDIDATE_POOL_SIZE = 20
RERANK_THRESHOLD = float(os.environ.get("RERANK_THRESHOLD", "0.3"))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "indexed_documents": store.count(),
        "embedder": type(embedder).__name__,
        "store": type(store).__name__,
        "reranker": type(reranker).__name__,
    }


@app.post("/index")
def index_documents(request: IndexRequest) -> dict:
    ids = [doc.id for doc in request.documents]
    texts = [doc.text for doc in request.documents]
    payloads = [{"text": doc.text, "metadata": doc.metadata} for doc in request.documents]

    vectors = embedder.embed(texts)
    store.add(ids, vectors, payloads)
    keyword_index.add(ids, texts, payloads)

    return {"indexed": len(ids), "total_documents": store.count()}


@app.get("/documents", response_model=ListDocumentsResponse)
def list_documents() -> ListDocumentsResponse:
    """Lists every distinct source document currently indexed, with a chunk
    count each -- use this to see what's there before deleting anything."""
    items = store.list_all()
    source_counts = Counter(item["metadata"].get("source", "(unknown)") for item in items)

    documents = [
        DocumentSummary(source=source, chunk_count=count)
        for source, count in sorted(source_counts.items())
    ]
    return ListDocumentsResponse(documents=documents, total_chunks=len(items))


@app.delete("/documents/by-source", response_model=DeleteResponse)
def delete_by_source(source: str) -> DeleteResponse:
    """
    Deletes every chunk/summary belonging to the given source document
    (matched against metadata.source, e.g. the original PDF file path).
    This is the practical way to remove an entire document -- ingestion
    tags every chunk and summary with the same source value, so this
    removes all of them in one call.
    """
    items = store.list_all()
    matching_ids = [item["id"] for item in items if item["metadata"].get("source") == source]

    if not matching_ids:
        raise HTTPException(status_code=404, detail=f"No documents found with source='{source}'")

    removed_from_store = store.delete(matching_ids)
    keyword_index.delete(matching_ids)

    return DeleteResponse(deleted=removed_from_store, remaining_documents=store.count())


@app.post("/documents/delete-by-ids", response_model=DeleteResponse)
def delete_by_ids(request: DeleteByIdsRequest) -> DeleteResponse:
    """Deletes specific chunks/documents by their exact ids, for finer-grained control."""
    removed_from_store = store.delete(request.ids)
    keyword_index.delete(request.ids)

    return DeleteResponse(deleted=removed_from_store, remaining_documents=store.count())


@app.delete("/documents/all", response_model=DeleteResponse)
def delete_all_documents() -> DeleteResponse:
    """Wipes the entire index -- both the vector store and the keyword index."""
    all_ids = [item["id"] for item in store.list_all()]
    removed = store.delete(all_ids)
    keyword_index.delete(all_ids)
    return DeleteResponse(deleted=removed, remaining_documents=store.count())


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    query_vector = embedder.embed([request.query])[0]
    dense_results = store.search(query_vector, top_k=CANDIDATE_POOL_SIZE)
    keyword_results = keyword_index.search(request.query, top_k=CANDIDATE_POOL_SIZE)

    fused = reciprocal_rank_fusion([dense_results, keyword_results])

    reranked = reranker.rerank(
        request.query,
        fused,
        top_k=request.top_k,
        threshold=RERANK_THRESHOLD if isinstance(reranker, CrossEncoderReranker) else None,
    )

    results = [
        SearchResult(
            id=r["id"],
            text=r.get("text", ""),
            score=r.get("rerank_score", r.get("rrf_score", 0.0)),
            metadata=r.get("metadata", {}),
        )
        for r in reranked
    ]
    return SearchResponse(results=results)