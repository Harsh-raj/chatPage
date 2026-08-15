from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_index_then_search_returns_relevant_doc():
    index_response = client.post("/index", json={
        "documents": [
            {"id": "doc1", "text": "The capital of France is Paris.",
                "metadata": {"source": "wiki"}},
            {"id": "doc2", "text": "Bananas are a good source of potassium.",
                "metadata": {"source": "wiki"}},
        ]
    })
    assert index_response.status_code == 200
    assert index_response.json()["indexed"] == 2

    search_response = client.post(
        "/search", json={"query": "The capital of France is Paris.", "top_k": 1})
    assert search_response.status_code == 200
    results = search_response.json()["results"]
    assert len(results) == 1
    assert results[0]["id"] == "doc1"
