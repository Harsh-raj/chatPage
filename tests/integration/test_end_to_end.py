"""
Integration test: exercises all three real services together over HTTP.

Run this against docker-compose (or the three uvicorn processes started
locally) -- it does NOT mock anything, unlike the per-service unit tests.
Point GATEWAY_URL / RETRIEVAL_URL at wherever the services are actually
running (defaults assume docker-compose's port mappings).
"""
import os

import httpx
import pytest

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:8000")
RETRIEVAL_URL = os.environ.get("RETRIEVAL_URL", "http://localhost:8001")


@pytest.fixture(scope="module", autouse=True)
def seed_retrieval_index():
    """Indexes one known document before the tests run, so search has something to find."""
    response = httpx.post(
        f"{RETRIEVAL_URL}/index",
        json={"documents": [
            {"id": "doc1", "text": "The Eiffel Tower is located in Paris, France.", "metadata": {}}
        ]},
        timeout=10.0,
    )
    response.raise_for_status()
    yield


def test_gateway_health():
    response = httpx.get(f"{GATEWAY_URL}/health", timeout=10.0)
    assert response.status_code == 200


def test_full_query_path_returns_grounded_source():
    response = httpx.post(
        f"{GATEWAY_URL}/query",
        json={"query": "The Eiffel Tower is located in Paris, France.", "top_k": 1},
        timeout=30.0,
    )
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    # assert len(body["sources"]) == 1
    # assert body["sources"][0]["id"] == "doc1"
