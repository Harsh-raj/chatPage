import os

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from app.llm_client import OllamaLLMClient, StubLLMClient
from app.prompt import build_rag_prompt
from app.schema import GenerateRequest, GenerateResponse

app = FastAPI(title="generation-service")

if os.environ.get("USE_REAL_LLM") == "1":
    llm_client = OllamaLLMClient()
else:
    llm_client = StubLLMClient()


@app.get("/health")
def health():
    return {"status": "ok", "llm_backend": type(llm_client).__name__}


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    prompt = build_rag_prompt(request.query, request.context_chunks)
    answer = llm_client.generate(prompt)
    return GenerateResponse(answer=answer, prompt_char_length=len(prompt))


@app.post("/generate/stream")
def generate_stream(request: GenerateRequest) -> StreamingResponse:
    """Streams the answer as plain text chunks as they're generated."""
    prompt = build_rag_prompt(request.query, request.context_chunks)

    def token_stream():
        for chunk in llm_client.generate_stream(prompt):
            yield chunk

    return StreamingResponse(token_stream(), media_type="text/plain")