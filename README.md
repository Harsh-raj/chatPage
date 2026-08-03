# RAG microservices scaffold

Three services: `gateway` (orchestrator), `retrieval` (vector search),
`generation` (LLM). Each has its own `requirements.txt`, `Dockerfile`, and
`tests/` with fast, mocked unit tests. `tests/integration/` exercises all
three together over real HTTP.

## Local development (no Docker)
```bash
cd retrieval && pip install -r requirements.txt -r ../dev-requirements.txt && pytest tests/ -v
cd ../generation && pip install -r requirements.txt -r ../dev-requirements.txt && pytest tests/ -v
cd ../gateway && pip install -r requirements.txt -r ../dev-requirements.txt && pytest tests/ -v
```

## Full stack via Docker
```bash
docker compose up --build
```
- Gateway: http://localhost:8000/docs
- Retrieval: http://localhost:8001/docs
- Generation: http://localhost:8002/docs

## Integration tests (against the running stack)
```bash
pytest tests/integration/ -v
```

## Swapping stubs for real implementations
- `retrieval/app/vector_store.py`: swap `InMemoryVectorStore` for a Qdrant client
- `retrieval/app/embeddings.py`: swap `DeterministicHashEmbedder` for a real embedding model
- `generation/app/llm_client.py`: set `USE_REAL_LLM=1` to use `OllamaLLMClient` instead of `StubLLMClient`

## CI/CD
See `.github/workflows/ci.yml`: unit tests per service → docker-compose
integration test → build & push images to GHCR (main branch only).
