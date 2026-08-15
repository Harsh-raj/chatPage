"""
Thin HTTP clients for the retrieval and generation services.
Base URLs come from environment variables so the same code works whether
services are reached via localhost (local dev) or docker-compose service
names (containerized).
"""

import os
from collections.abc import Iterator

import httpx

from app.auth import auth_headers

RETRIEVAL_URL = os.environ.get("RETRIEVAL_URL", "http://localhost:8001")
GENERATION_URL = os.environ.get("GENERATION_URL", "http://localhost:8002")


def _merged_headers(headers: dict | None) -> dict:
    """Combines caller-supplied headers (Langfuse trace propagation, see
    app/tracing.py) with this service's own auth headers, so a caller
    doesn't need to remember to attach both on every call site."""
    return {**auth_headers(), **(headers or {})}


def search(query: str, top_k: int = 5, headers: dict | None = None) -> list[dict]:
    response = httpx.post(
        f"{RETRIEVAL_URL}/search",
        json={"query": query, "top_k": top_k},
        headers=_merged_headers(headers),
        timeout=10.0,
    )
    response.raise_for_status()
    return response.json()["results"]


def generate(query: str, context_chunks: list[str], headers: dict | None = None) -> str:
    response = httpx.post(
        f"{GENERATION_URL}/generate",
        json={"query": query, "context_chunks": context_chunks},
        headers=_merged_headers(headers),
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()["answer"]


def generate_stream(query: str, context_chunks: list[str], headers: dict | None = None) -> Iterator[str]:
    """Forwards generation's streamed text chunks onward, one at a time."""
    with httpx.stream(
        "POST",
        f"{GENERATION_URL}/generate/stream",
        json={"query": query, "context_chunks": context_chunks},
        headers=_merged_headers(headers),
        timeout=120.0,
    ) as response:
        response.raise_for_status()
        for chunk in response.iter_text():
            if chunk:
                yield chunk
