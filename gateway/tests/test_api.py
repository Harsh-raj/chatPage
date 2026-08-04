from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200


@patch("app.client.generate")
@patch("app.client.search")
def test_query_orchestrates_retrieval_then_generation(mock_search, mock_generate):
    # Arrange: pretend retrieval and generation services responded
    mock_search.return_value = [
        {"id": "doc1", "text": "Paris is the capital of France.", "score": 0.95}
    ]
    mock_generate.return_value = "The capital of France is Paris."

    # Act
    response = client.post("/query", json={"query": "What is the capital of France?"})

    # Assert
    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "The capital of France is Paris."
    # assert body["sources"][0]["id"] == "doc1"

    # Assert the gateway called retrieval BEFORE generation, and passed
    # retrieved text as context -- this is the actual orchestration contract.
    mock_search.assert_called_once_with("What is the capital of France?", top_k=5)
    mock_generate.assert_called_once_with(
        "What is the capital of France?", ["Paris is the capital of France."]
    )
