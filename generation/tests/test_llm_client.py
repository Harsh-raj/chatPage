from app.llm_client import StubLLMClient


def test_stub_client_returns_deterministic_response():
    client = StubLLMClient()
    r1 = client.generate("some prompt")
    r2 = client.generate("some prompt")
    assert r1 == r2


def test_stub_client_response_reflects_prompt_length():
    client = StubLLMClient()
    response = client.generate("12345")
    assert "5 char prompt" in response
