from unittest.mock import patch

import app.auth as auth_module
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

FAKE_FILE = {"file": ("test.pdf", b"%PDF-1.4 fake content", "application/pdf")}


def test_health_check_never_requires_auth_even_when_key_is_set(monkeypatch):
    monkeypatch.setattr(auth_module, "API_KEY", "secret123")
    response = client.get("/health")
    assert response.status_code == 200


def test_auth_disabled_by_default_lets_requests_through(monkeypatch):
    monkeypatch.setattr(auth_module, "API_KEY", None)
    with (
        patch(
            "app.main.iter_build_documents",
            return_value=iter([{"type": "result", "documents": [], "warnings": []}]),
        ),
        patch("app.main.iter_index_documents", return_value=iter([])),
        client.stream("POST", "/ingest/stream", files=FAKE_FILE) as response,
    ):
        assert response.status_code == 200


def test_protected_endpoint_rejects_missing_key(monkeypatch):
    monkeypatch.setattr(auth_module, "API_KEY", "secret123")
    with client.stream("POST", "/ingest/stream", files=FAKE_FILE) as response:
        assert response.status_code == 401


def test_protected_endpoint_rejects_wrong_key(monkeypatch):
    monkeypatch.setattr(auth_module, "API_KEY", "secret123")
    with client.stream("POST", "/ingest/stream", files=FAKE_FILE, headers={"X-API-Key": "wrong"}) as response:
        assert response.status_code == 401


def test_protected_endpoint_accepts_correct_key(monkeypatch):
    monkeypatch.setattr(auth_module, "API_KEY", "secret123")
    with (
        patch(
            "app.main.iter_build_documents",
            return_value=iter([{"type": "result", "documents": [], "warnings": []}]),
        ),
        patch("app.main.iter_index_documents", return_value=iter([])),
        client.stream("POST", "/ingest/stream", files=FAKE_FILE, headers={"X-API-Key": "secret123"}) as response,
    ):
        assert response.status_code == 200
