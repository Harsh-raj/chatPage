import json

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from app import client
from app.auth import ApiKeyMiddleware
from app.schema import QueryRequest, QueryResponse, SourceChunk
from app.tracing import downstream_headers, incoming_trace_id, langfuse

app = FastAPI(title="gateway-service")
app.add_middleware(ApiKeyMiddleware)


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

    for name, url in [
        ("retrieval", client.RETRIEVAL_URL),
        ("generation", client.GENERATION_URL),
    ]:
        try:
            resp = httpx.get(f"{url}/health", timeout=3.0)
            resp.raise_for_status()
            downstream[name] = {"ready": True, **resp.json()}
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
            downstream[name] = {"ready": False, "error": str(e)}
            all_ready = False

    return {"status": "ok" if all_ready else "starting", "services": downstream}


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest, http_request: Request) -> QueryResponse:
    lf = langfuse()
    trace_id = incoming_trace_id(http_request.headers) or lf.create_trace_id()

    with lf.start_as_current_observation(
        name="query",
        as_type="span",
        trace_context={"trace_id": trace_id},
        input={"query": payload.query, "top_k": payload.top_k},
    ) as root_span:
        results = client.search(
            payload.query, top_k=payload.top_k, headers=downstream_headers(root_span))
        context_chunks = [r["text"] for r in results]

        answer = client.generate(
            payload.query, context_chunks, headers=downstream_headers(root_span))

        sources = [SourceChunk(id=r["id"], text=r["text"],
                               score=r["score"]) for r in results]
        root_span.update(
            output={"answer": answer, "num_sources": len(sources)})

    try:
        trace_url = lf.get_trace_url(trace_id=trace_id)
    except Exception:
        # get_trace_url() calls out to the Langfuse API to resolve a project id --
        # if Langfuse isn't configured (no keys) or is unreachable, that call
        # fails. It's a nice-to-have link, not something worth failing the
        # actual query over, so we just omit it.
        trace_url = None

    return QueryResponse(
        answer=answer,
        sources=sources,
        trace_id=trace_id,
        trace_url=trace_url,
    )


@app.post("/query/stream")
def query_stream(payload: QueryRequest, http_request: Request) -> StreamingResponse:
    lf = langfuse()
    trace_id = incoming_trace_id(http_request.headers) or lf.create_trace_id()

    # Streaming responses outlive this function's stack frame (the generator
    # below runs after we return, sometimes on a different thread/async task
    # than this handler -- start_as_current_observation's `with` form ties
    # itself to Python contextvars and breaks across that boundary). Use the
    # plain (non-"current") start_observation() instead: it returns a handle
    # we update/end manually, with no ambient-context entanglement.
    root_span = lf.start_observation(
        name="query_stream",
        as_type="span",
        trace_context={"trace_id": trace_id},
        input={"query": payload.query, "top_k": payload.top_k},
    )
    headers = downstream_headers(root_span)

    results = client.search(
        payload.query, top_k=payload.top_k, headers=headers)
    context_chunks = [r["text"] for r in results]

    sources = [{"id": r["id"], "text": r["text"], "score": r["score"]}
               for r in results]

    def event_stream():
        full_answer_parts = []
        try:
            yield (json.dumps({"type": "sources", "sources": sources, "trace_id": trace_id}) + "\n")
            for chunk in client.generate_stream(payload.query, context_chunks, headers=headers):
                full_answer_parts.append(chunk)
                yield json.dumps({"type": "token", "text": chunk}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
        finally:
            root_span.update(
                output={
                    "answer": "".join(full_answer_parts),
                    "num_sources": len(sources),
                }
            )
            root_span.end()

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
