from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["llm_backend"] == "StubLLMClient"


def test_generate_returns_answer():
    response = client.post("/generate", json={
        "query": "What is the capital of France?",
        "context_chunks": ["Paris is the capital of France."]
    })
    assert response.status_code == 200
    body = response.json()
    assert "answer" in body
    assert body["prompt_char_length"] > 0
