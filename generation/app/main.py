import os

from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse

from app.auth import ApiKeyMiddleware
from app.llm_client import LLMClient, OllamaLLMClient, StubLLMClient
from app.prompt import build_rag_prompt_with_flags
from app.schema import GenerateRequest, GenerateResponse
from app.tracing import langfuse, trace_context_from_headers

app = FastAPI(title="generation-service")
app.add_middleware(ApiKeyMiddleware)

if os.environ.get("USE_REAL_LLM") == "1":
    llm_client: LLMClient = OllamaLLMClient()
else:
    llm_client = StubLLMClient()


@app.get("/health")
def health():
    return {"status": "ok", "llm_backend": type(llm_client).__name__}


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest, http_request: Request) -> GenerateResponse:
    prompt, injection_flags = build_rag_prompt_with_flags(
        request.query, request.context_chunks)

    lf = langfuse()
    trace_context = trace_context_from_headers(http_request.headers)

    with lf.start_as_current_observation(
        name="generate",
        as_type="generation",
        trace_context=trace_context,
        input=prompt,
        model=getattr(llm_client, "model", type(llm_client).__name__),
        metadata={
            "llm_backend": type(llm_client).__name__,
            "num_context_chunks": len(request.context_chunks),
            # Surfaced here specifically so a flagged chunk is visible in
            # the trace, not just in the API response -- see
            # app/guardrails.py's module docstring on why flagged chunks
            # are made observable rather than silently redacted.
            "injection_flags": injection_flags,
        },
    ) as span:
        answer = llm_client.generate(prompt)
        span.update(output=answer)
        if injection_flags:
            span.update(
                level="WARNING", status_message=f"{len(injection_flags)} context chunk(s) flagged and redacted")

    return GenerateResponse(answer=answer, prompt_char_length=len(prompt), injection_flags=injection_flags)


@app.post("/generate/stream")
def generate_stream(request: GenerateRequest, http_request: Request) -> StreamingResponse:
    """Streams the answer as plain text chunks as they're generated."""
    prompt, injection_flags = build_rag_prompt_with_flags(
        request.query, request.context_chunks)

    lf = langfuse()
    trace_context = trace_context_from_headers(http_request.headers)

    # The generator below runs after this function returns, sometimes on a
    # different thread/async task than this handler -- start_as_current_
    # observation's `with` form ties itself to Python contextvars and breaks
    # across that boundary (raises "Failed to detach context" from a token
    # created in a different Context). Use the plain (non-"current")
    # start_observation() instead: it returns a handle we update/end
    # manually, with no ambient-context entanglement.
    span = lf.start_observation(
        name="generate_stream",
        as_type="generation",
        trace_context=trace_context,
        input=prompt,
        model=getattr(llm_client, "model", type(llm_client).__name__),
        metadata={
            "llm_backend": type(llm_client).__name__,
            "num_context_chunks": len(request.context_chunks),
            "injection_flags": injection_flags,
        },
    )
    if injection_flags:
        span.update(
            level="WARNING", status_message=f"{len(injection_flags)} context chunk(s) flagged and redacted")

    def token_stream():
        parts = []
        try:
            for chunk in llm_client.generate_stream(prompt):
                parts.append(chunk)
                yield chunk
        finally:
            span.update(output="".join(parts))
            span.end()

    # The streamed body is plain text tokens, with no room for a structured
    # field the way the non-streaming /generate response has injection_flags
    # -- surfaced as a response header instead, so a caller can still see
    # whether anything was flagged without needing to check the trace.
    headers = {"X-Injection-Flags-Count": str(len(injection_flags))}
    return StreamingResponse(token_stream(), media_type="text/plain", headers=headers)
