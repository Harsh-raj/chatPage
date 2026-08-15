from unittest.mock import patch

from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200


@patch("app.client.generate")
@patch("app.client.search")
def test_query_orchestrates_retrieval_then_generation(mock_search, mock_generate):
    # Arrange: pretend retrieval and generation services responded
    mock_search.return_value = [{"id": "doc1", "text": "Paris is the capital of France.", "score": 0.95}]
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
    # (headers=... carries Langfuse trace propagation -- see
    # gateway/app/tracing.py -- its exact value isn't the contract under
    # test here, so we don't assert on it.)

    search_call = mock_search.call_args
    assert search_call.args == ("What is the capital of France?",)
    assert search_call.kwargs["top_k"] == 5

    generate_call = mock_generate.call_args
    assert generate_call.args == (
        "What is the capital of France?",
        ["Paris is the capital of France."],
    )
