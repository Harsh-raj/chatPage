"""
Structural/rendering tests using Streamlit's own AppTest framework, which
runs the actual main.py script and lets us inspect what it rendered.

This exists specifically because "is the upload sidebar actually visible"
turned out to be a real point of confusion once (main.py's code was
correct, the running container was stale) -- these tests would have caught
a *code-level* version of that same problem (e.g. the sidebar block
accidentally moved below a `st.stop()`), which no amount of testing
logic.py's pure functions in isolation would ever catch.

Two important environment knobs used everywhere below:
- requests.get/requests.post are patched globally (not "app.main.requests.*"
  or "app.logic.requests.*") because AppTest re-executes main.py fresh each
  run rather than reusing whatever module object a prior `import app.main`
  produced -- patching the global `requests` module's attributes is the one
  patch target guaranteed to affect whatever fresh execution AppTest does.
- FRONTEND_READY_MAX_WAIT_SECONDS / FRONTEND_READY_POLL_INTERVAL_SECONDS are
  set to 1 second so the readiness gate's real (but now tiny) time.sleep()
  calls don't make every test slow. Do NOT patch time.sleep itself here --
  Streamlit's own test harness relies on time.sleep internally, and
  patching it globally breaks AppTest's own machinery, not just main.py's.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
from streamlit.testing.v1 import AppTest

MAIN_PATH = str(Path(__file__).resolve().parent.parent / "app" / "main.py")


@pytest.fixture(autouse=True)
def fast_readiness_gate(monkeypatch):
    monkeypatch.setenv("FRONTEND_READY_MAX_WAIT_SECONDS", "1")
    monkeypatch.setenv("FRONTEND_READY_POLL_INTERVAL_SECONDS", "1")


def _healthy_response():
    mock = MagicMock()
    mock.raise_for_status.return_value = None
    mock.json.return_value = {"status": "ok", "services": {"retrieval": {"ready": True}, "generation": {"ready": True}}}
    return mock


# --- Sidebar: the upload feature itself ---


def test_sidebar_shows_upload_controls():
    with patch("requests.get", return_value=_healthy_response()):
        at = AppTest.from_file(MAIN_PATH).run(timeout=15)

    assert not at.exception
    assert any(h.value == "Add a document" for h in at.sidebar.header)
    assert len(at.sidebar.file_uploader) == 1
    assert len(at.sidebar.expander) == 1
    assert len(at.sidebar.checkbox) == 3  # summaries, tables, OCR
    assert len(at.sidebar.number_input) == 2  # chunk size, overlap
    assert len(at.sidebar.button) == 1


def test_ingest_button_disabled_with_no_file_selected():
    with patch("requests.get", return_value=_healthy_response()):
        at = AppTest.from_file(MAIN_PATH).run(timeout=15)

    assert at.sidebar.button[0].label == "Ingest document"
    assert at.sidebar.button[0].disabled is True


# --- Readiness gate ---


def test_chat_input_hidden_while_backend_not_ready():
    with patch("requests.get", side_effect=requests.exceptions.ConnectionError("refused")):
        at = AppTest.from_file(MAIN_PATH).run(timeout=15)

    assert not at.exception
    assert len(at.chat_input) == 0
    assert len(at.error) == 1
    assert "haven't started" in at.error[0].value


def test_chat_input_appears_once_backend_is_ready():
    with patch("requests.get", return_value=_healthy_response()):
        at = AppTest.from_file(MAIN_PATH).run(timeout=15)

    assert not at.exception
    assert len(at.chat_input) == 1
    assert len(at.error) == 0


def test_sidebar_still_renders_while_backend_not_ready():
    """The sidebar (upload UI) is written before the readiness gate in
    main.py -- confirm it stays visible even while the gate is blocking the
    chat UI, since a user might reasonably want to upload a document while
    waiting for the backend to finish starting."""
    with patch("requests.get", side_effect=requests.exceptions.ConnectionError("refused")):
        at = AppTest.from_file(MAIN_PATH).run(timeout=15)

    assert any(h.value == "Add a document" for h in at.sidebar.header)


# --- Chat round trip ---


def test_full_chat_round_trip_renders_answer_and_sources():
    query_response_lines = [
        json.dumps({"type": "sources", "sources": [{"id": "doc1", "text": "Paris is the capital.", "score": 0.92}]}),
        json.dumps({"type": "token", "text": "Paris "}),
        json.dumps({"type": "token", "text": "is the capital of France."}),
        json.dumps({"type": "done"}),
    ]
    mock_stream_response = MagicMock()
    mock_stream_response.__enter__.return_value = mock_stream_response
    mock_stream_response.__exit__.return_value = False
    mock_stream_response.raise_for_status.return_value = None
    mock_stream_response.iter_lines.return_value = query_response_lines

    with (
        patch("requests.get", return_value=_healthy_response()),
        patch("requests.post", return_value=mock_stream_response),
    ):
        at = AppTest.from_file(MAIN_PATH).run(timeout=15)
        at.chat_input[0].set_value("What is the capital of France?").run(timeout=15)

    assert not at.exception
    messages = [m for m in at.chat_message]
    assert len(messages) == 2  # user + assistant
    assert messages[0].name == "user"
    assert messages[1].name == "assistant"
    assistant_markdown = [m.value for m in messages[1].markdown]
    assert any("Paris is the capital of France." in v for v in assistant_markdown)
