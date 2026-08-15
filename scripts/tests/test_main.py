import json
from unittest.mock import patch

from app.main import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ingest_stream_reports_progress_then_done():
    fake_extraction_events = [
        {"type": "start", "num_pages": 1},
        {"type": "page_extracted", "page": 1, "num_pages": 1, "documents_so_far": 1},
        {
            "type": "result",
            "documents": [
                {
                    "id": "doc::page1::chunk0",
                    "text": "hello",
                    "metadata": {"level": "chunk"},
                }
            ],
            "warnings": [],
        },
    ]
    fake_index_events = [{"type": "index_progress", "indexed": 1, "total": 1}]

    with (
        patch("app.main.iter_build_documents", return_value=iter(fake_extraction_events)),
        patch("app.main.iter_index_documents", return_value=iter(fake_index_events)),
        client.stream(
            "POST",
            "/ingest/stream",
            files={"file": ("test.pdf", b"%PDF-1.4 fake content", "application/pdf")},
        ) as response,
    ):
        assert response.status_code == 200
        events = [json.loads(line) for line in response.iter_lines() if line]

    types = [e["type"] for e in events]
    assert types == [
        "start",
        "page_extracted",
        "extraction_done",
        "index_progress",
        "done",
    ]
    assert events[-1]["num_documents"] == 1
    assert events[-1]["counts"] == {"chunk": 1}


def test_ingest_stream_reports_error_for_unreadable_pdf():
    from app.ingest_pdf import PdfIngestionError

    def raise_error(*args, **kwargs):
        raise PdfIngestionError("Could not open 'bad.pdf': not a PDF")
        yield  # pragma: no cover -- makes this a generator function, never reached

    with (
        patch("app.main.iter_build_documents", side_effect=raise_error),
        client.stream(
            "POST",
            "/ingest/stream",
            files={"file": ("bad.pdf", b"not actually a pdf", "application/pdf")},
        ) as response,
    ):
        assert response.status_code == 200
        events = [json.loads(line) for line in response.iter_lines() if line]

    assert events[-1]["type"] == "error"
    assert "Could not open" in events[-1]["message"]
