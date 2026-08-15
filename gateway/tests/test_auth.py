from unittest.mock import patch

import app.auth as auth_module
from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_health_check_never_requires_auth_even_when_key_is_set(monkeypatch):
    monkeypatch.setattr(auth_module, "API_KEY", "secret123")
    response = client.get("/health")
    assert response.status_code == 200


def test_auth_disabled_by_default_lets_requests_through(monkeypatch):
    """API_KEY unset (the out-of-the-box default) means auth is off --
    no header required at all. See app/auth.py's module docstring for why
    this is a loud, logged default rather than a silent one."""
    monkeypatch.setattr(auth_module, "API_KEY", None)
    with patch("app.client.search", return_value=[]), patch("app.client.generate", return_value="ok"):
        response = client.post("/query", json={"query": "test"})
    assert response.status_code == 200


def test_protected_endpoint_rejects_missing_key(monkeypatch):
    monkeypatch.setattr(auth_module, "API_KEY", "secret123")
    response = client.post("/query", json={"query": "test"})
    assert response.status_code == 401


def test_protected_endpoint_rejects_wrong_key(monkeypatch):
    monkeypatch.setattr(auth_module, "API_KEY", "secret123")
    response = client.post("/query", json={"query": "test"}, headers={"X-API-Key": "wrong"})
    assert response.status_code == 401


def test_protected_endpoint_accepts_correct_key(monkeypatch):
    monkeypatch.setattr(auth_module, "API_KEY", "secret123")
    with patch("app.client.search", return_value=[]), patch("app.client.generate", return_value="ok"):
        response = client.post("/query", json={"query": "test"}, headers={"X-API-Key": "secret123"})
    assert response.status_code == 200
