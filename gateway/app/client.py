"""
Thin HTTP clients for the retrieval and generation services.
Base URLs come from environment variables so the same code works whether
services are reached via localhost (local dev) or docker-compose service
names (containerized).
"""
import os
import json
import httpx
from typing import Iterator

RETRIEVAL_URL = os.environ.get("RETRIEVAL_URL", "http://localhost:8001")
GENERATION_URL = os.environ.get("GENERATION_URL", "http://localhost:8002")


def search(query: str, top_k: int = 5) -> list[dict]:
    response = httpx.post(
        f"{RETRIEVAL_URL}/search",
        json={"query": query, "top_k": top_k},
        timeout=10.0,
    )
    response.raise_for_status()
    return response.json()["results"]


def generate(query: str, context_chunks: list[str]) -> str:
    response = httpx.post(
        f"{GENERATION_URL}/generate",
        json={"query": query, "context_chunks": context_chunks},
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()["answer"]


def generate_stream(query: str, context_chunks: list[str]) -> Iterator[str]:
    """Forwards generation's streamed text chunks onward, one at a time."""
    with httpx.stream(
        "POST",
        f"{GENERATION_URL}/generate/stream",
        json={"query": query, "context_chunks": context_chunks},
        timeout=120.0,
    ) as response:
        response.raise_for_status()
        for chunk in response.iter_text():
            if chunk:
                yield chunk
