import json
import httpx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from app.schema import QueryRequest, QueryResponse, SourceChunk
from app import client

app = FastAPI(title="gateway-service")


@app.get("/")
def root():
    return {"message": "RAG gateway is running. Visit /docs to test."}


@app.get("/health")
def health():
    """
    Aggregated readiness check. Pings retrieval and generation directly so a
    single call here tells the whole story -- the frontend can poll just this
    one endpoint instead of checking three services separately.
    """
    downstream = {}
    all_ready = True

    for name, url in [("retrieval", client.RETRIEVAL_URL), ("generation", client.GENERATION_URL)]:
        try:
            resp = httpx.get(f"{url}/health", timeout=3.0)
            resp.raise_for_status()
            downstream[name] = {"ready": True, **resp.json()}
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
            downstream[name] = {"ready": False, "error": str(e)}
            all_ready = False

    return {"status": "ok" if all_ready else "starting", "services": downstream}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    results = client.search(request.query, top_k=request.top_k)
    context_chunks = [r["text"] for r in results]

    answer = client.generate(request.query, context_chunks)

    sources = [SourceChunk(id=r["id"], text=r["text"],
                           score=r["score"]) for r in results]
    return QueryResponse(answer=answer, sources=sources)


@app.post("/query/stream")
def query_stream(request: QueryRequest) -> StreamingResponse:
    results = client.search(request.query, top_k=request.top_k)
    context_chunks = [r["text"] for r in results]

    sources = [
        {"id": r["id"], "text": r["text"], "score": r["score"]}
        for r in results
    ]

    def event_stream():
        yield json.dumps({"type": "sources", "sources": sources}) + "\n"
        for chunk in client.generate_stream(request.query, context_chunks):
            yield json.dumps({"type": "token", "text": chunk}) + "\n"
        yield json.dumps({"type": "done"}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
