"""
Direct unit tests for app/logic.py's non-rendering functions
(_describe_ingest_event, check_backend_ready, stream_query). These have no
st.* calls, so they're testable with plain function calls and mocked
requests -- no Streamlit test harness needed.

render_sources() and ingest_pdf_with_progress() are NOT covered here since
they're built entirely out of st.* rendering calls (st.columns, st.popover,
st.status, ...) -- their actual behavior is what renders, which is exactly
what test_app.py's AppTest-based tests check instead.
"""
import pytest
import requests
from unittest.mock import patch, MagicMock

from app.logic import _describe_ingest_event, check_backend_ready, stream_query


# --- _describe_ingest_event: every event type, plus the unknown-type case ---

@pytest.mark.parametrize("event,expected_substring", [
    ({"type": "start", "num_pages": 5}, "Found 5 page(s)"),
    ({"type": "page_extracted", "page": 2, "num_pages": 5, "documents_so_far": 7}, "Extracted page 2/5"),
    ({"type": "summarizing_page", "page": 3, "num_pages": 5}, "Summarizing page 3/5"),
    ({"type": "summarizing_document"}, "whole-document summary"),
    ({"type": "extraction_done", "counts": {"chunk": 3, "table": 1}}, "3 chunk, 1 table"),
    ({"type": "index_progress", "indexed": 10, "total": 20}, "Indexed 10/20"),
    ({"type": "done", "num_documents": 12, "filename": "report.pdf"}, "12 chunk(s) from report.pdf"),
    ({"type": "error", "message": "boom"}, "Error: boom"),
])
def test_describe_ingest_event_known_types(event, expected_substring):
    result = _describe_ingest_event(event)
    assert result is not None
    assert expected_substring in result


def test_describe_ingest_event_extraction_done_with_nothing_extracted():
    result = _describe_ingest_event({"type": "extraction_done", "counts": {}})
    assert "nothing" in result


def test_describe_ingest_event_index_progress_nothing_to_index():
    result = _describe_ingest_event({
        "type": "index_progress", "total": 0, "message": "Nothing to index.",
    })
    assert result == "Nothing to index."


def test_describe_ingest_event_unknown_type_returns_none():
    assert _describe_ingest_event({"type": "some_future_event_type"}) is None


# --- check_backend_ready ---

def test_check_backend_ready_returns_health_json_on_success():
    with patch("app.logic.requests.get") as mock_get:
        mock_get.return_value.raise_for_status.return_value = None
        mock_get.return_value.json.return_value = {"status": "ok"}
        result = check_backend_ready("http://fake-gateway:8000")

    assert result == {"status": "ok"}
    mock_get.assert_called_once_with("http://fake-gateway:8000/health", timeout=5)


def test_check_backend_ready_returns_none_when_unreachable():
    with patch("app.logic.requests.get", side_effect=requests.exceptions.ConnectionError("refused")):
        result = check_backend_ready("http://fake-gateway:8000")

    assert result is None


def test_check_backend_ready_returns_none_on_http_error():
    with patch("app.logic.requests.get") as mock_get:
        mock_get.return_value.raise_for_status.side_effect = requests.exceptions.HTTPError("503")
        result = check_backend_ready("http://fake-gateway:8000")

    assert result is None


# --- stream_query ---

def _fake_streaming_response(lines):
    """A MagicMock standing in for `with requests.post(...) as response:` --
    supports the context-manager protocol and .iter_lines()."""
    mock_response = MagicMock()
    mock_response.__enter__.return_value = mock_response
    mock_response.__exit__.return_value = False
    mock_response.raise_for_status.return_value = None
    mock_response.iter_lines.return_value = lines
    return mock_response


def test_stream_query_yields_tokens_and_collects_sources():
    import json
    lines = [
        json.dumps({"type": "sources", "sources": [{"id": "doc1", "text": "hi", "score": 0.9}]}),
        json.dumps({"type": "token", "text": "Hello "}),
        json.dumps({"type": "token", "text": "world"}),
        json.dumps({"type": "done"}),
    ]
    sources, error_holder = [], {"backend_down": False}

    with patch("app.logic.requests.post", return_value=_fake_streaming_response(lines)):
        tokens = list(stream_query("http://fake-gateway:8000", "hi", sources, error_holder))

    assert "".join(tokens) == "Hello world"
    assert sources == [{"id": "doc1", "text": "hi", "score": 0.9}]
    assert error_holder["backend_down"] is False


def test_stream_query_marks_backend_down_on_connection_error():
    sources, error_holder = [], {"backend_down": False}

    with patch("app.logic.requests.post", side_effect=requests.exceptions.ConnectionError("refused")):
        tokens = list(stream_query("http://fake-gateway:8000", "hi", sources, error_holder))

    assert error_holder["backend_down"] is True
    assert any("Could not connect" in t for t in tokens)


def test_stream_query_marks_backend_down_on_5xx_but_not_4xx():
    error = requests.exceptions.HTTPError("500")
    error.response = MagicMock(status_code=500)
    sources, error_holder = [], {"backend_down": False}
    mock_response = MagicMock()
    mock_response.__enter__.return_value = mock_response
    mock_response.__exit__.return_value = False
    mock_response.raise_for_status.side_effect = error

    with patch("app.logic.requests.post", return_value=mock_response):
        tokens = list(stream_query("http://fake-gateway:8000", "hi", sources, error_holder))

    assert error_holder["backend_down"] is True
    assert any("isn't ready yet" in t for t in tokens)


def test_stream_query_does_not_mark_backend_down_on_4xx():
    error = requests.exceptions.HTTPError("400")
    error.response = MagicMock(status_code=400)
    sources, error_holder = [], {"backend_down": False}
    mock_response = MagicMock()
    mock_response.__enter__.return_value = mock_response
    mock_response.__exit__.return_value = False
    mock_response.raise_for_status.side_effect = error

    with patch("app.logic.requests.post", return_value=mock_response):
        tokens = list(stream_query("http://fake-gateway:8000", "hi", sources, error_holder))

    # A 4xx is a real client-side error, not a "backend isn't ready" signal --
    # re-triggering the readiness gate for it would be wrong.
    assert error_holder["backend_down"] is False
    assert any("Gateway returned an error" in t for t in tokens)


def test_stream_query_handles_read_timeout_without_marking_backend_down():
    sources, error_holder = [], {"backend_down": False}

    with patch("app.logic.requests.post", side_effect=requests.exceptions.ReadTimeout("slow")):
        tokens = list(stream_query("http://fake-gateway:8000", "hi", sources, error_holder))

    assert error_holder["backend_down"] is False
    assert any("taking longer than expected" in t for t in tokens)