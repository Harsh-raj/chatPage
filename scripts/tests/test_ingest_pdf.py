from unittest.mock import MagicMock, patch

import pytest
import requests
from app.ingest_pdf import (
    PdfIngestionError,
    _clean_text,
    build_documents,
    extract_page_content,
    iter_build_documents,
    iter_index_documents,
    open_pdf,
    table_to_markdown,
)

# --- Text cleaning (unchanged behavior) ---


def test_clean_text_removes_null_bytes():
    assert _clean_text("hello\x00world") == "helloworld"


def test_clean_text_removes_control_characters_but_keeps_newlines():
    assert _clean_text("line one\x01\x02\nline two\x7f") == "line one\nline two"


# --- Table to markdown conversion ---


def test_table_to_markdown_produces_valid_markdown_table():
    rows = [["Name", "Value"], ["A", "1"], ["B", "2"]]
    md = table_to_markdown(rows)
    lines = md.split("\n")
    assert lines[0] == "| Name | Value |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| A | 1 |"
    assert lines[3] == "| B | 2 |"


def test_table_to_markdown_handles_none_cells():
    rows = [["Name", "Value"], ["A", None]]
    md = table_to_markdown(rows)
    assert "| A |  |" in md


def test_table_to_markdown_handles_empty_input():
    assert table_to_markdown([]) == ""


# --- Corrupt/unreadable/encrypted files ---


def test_open_pdf_raises_clear_error_for_garbage_file(tmp_path):
    fake_pdf = tmp_path / "not_a_real.pdf"
    fake_pdf.write_bytes(b"garbage bytes, not a pdf")
    with pytest.raises(PdfIngestionError, match="Could not open"):
        open_pdf(str(fake_pdf))


def test_open_pdf_decrypts_with_empty_password():
    fake_doc = MagicMock()
    fake_doc.is_encrypted = True
    fake_doc.authenticate.return_value = 1  # truthy: success

    with patch("app.ingest_pdf.fitz.open", return_value=fake_doc):
        open_pdf("/fake/encrypted.pdf")

    fake_doc.authenticate.assert_called_once_with("")


def test_open_pdf_raises_when_decrypt_fails():
    fake_doc = MagicMock()
    fake_doc.is_encrypted = True
    fake_doc.authenticate.return_value = 0  # falsy: failure

    with (
        patch("app.ingest_pdf.fitz.open", return_value=fake_doc),
        pytest.raises(PdfIngestionError, match="password-protected"),
    ):
        open_pdf("/fake/encrypted.pdf")


# --- Page content extraction: text, tables, OCR ---


def _make_fake_page(text="", tables=None, images=None):
    page = MagicMock()
    page.get_text.return_value = text
    fake_table_result = MagicMock()
    fake_table_result.tables = tables or []
    page.find_tables.return_value = fake_table_result
    page.get_images.return_value = images or []
    return page


def test_extract_page_content_captures_regular_text():
    doc = MagicMock()
    doc.__getitem__.return_value = _make_fake_page(text="Some real page text.")
    warnings = []
    content = extract_page_content(doc, 0, warnings)
    assert content["text"] == "Some real page text."
    assert warnings == []


def test_extract_page_content_extracts_tables():
    fake_table = MagicMock()
    fake_table.extract.return_value = [["A", "B"], ["1", "2"]]
    doc = MagicMock()
    doc.__getitem__.return_value = _make_fake_page(text="some text", tables=[fake_table])
    warnings = []
    content = extract_page_content(doc, 0, warnings)
    assert len(content["tables"]) == 1
    assert "| A | B |" in content["tables"][0]


def test_extract_page_content_falls_back_to_ocr_when_no_text():
    doc = MagicMock()
    fake_page = _make_fake_page(text="")  # empty -- no real text layer
    fake_pixmap = MagicMock()
    fake_pixmap.tobytes.return_value = b"fake-png-bytes"
    fake_page.get_pixmap.return_value = fake_pixmap
    doc.__getitem__.return_value = fake_page

    warnings = []
    with (
        patch("app.ingest_pdf._OCR_AVAILABLE", True),
        patch("app.ingest_pdf.Image.open", return_value=MagicMock()),
        patch(
            "app.ingest_pdf.pytesseract.image_to_string",
            return_value="Recovered OCR text",
        ),
    ):
        content = extract_page_content(doc, 0, warnings)

    assert content["text"] == "Recovered OCR text"


def test_extract_page_content_warns_when_ocr_unavailable_and_no_text():
    doc = MagicMock()
    doc.__getitem__.return_value = _make_fake_page(text="")
    warnings = []
    with patch("app.ingest_pdf._OCR_AVAILABLE", False):
        content = extract_page_content(doc, 0, warnings)
    assert content["text"] == ""
    assert any("OCR unavailable" in w for w in warnings)


def test_extract_page_content_ocrs_embedded_images():
    doc = MagicMock()
    fake_page = _make_fake_page(text="regular text here", images=[(999,)])
    doc.__getitem__.return_value = fake_page
    doc.extract_image.return_value = {"image": b"fake-image-bytes"}

    warnings = []
    with (
        patch("app.ingest_pdf._OCR_AVAILABLE", True),
        patch("app.ingest_pdf.Image.open", return_value=MagicMock()),
        patch(
            "app.ingest_pdf.pytesseract.image_to_string",
            return_value="Diagram label text here",
        ),
    ):
        content = extract_page_content(doc, 0, warnings, run_image_ocr=True)

    assert content["image_ocr_texts"] == ["Diagram label text here"]


def test_extract_page_content_discards_short_noisy_image_ocr_results():
    doc = MagicMock()
    fake_page = _make_fake_page(text="regular text", images=[(999,)])
    doc.__getitem__.return_value = fake_page
    doc.extract_image.return_value = {"image": b"fake-bytes"}

    warnings = []
    with (
        patch("app.ingest_pdf._OCR_AVAILABLE", True),
        patch("app.ingest_pdf.Image.open", return_value=MagicMock()),
        patch("app.ingest_pdf.pytesseract.image_to_string", return_value="x"),
    ):  # too short/noisy
        content = extract_page_content(doc, 0, warnings, run_image_ocr=True)

    assert content["image_ocr_texts"] == []


# --- build_documents integration ---


def test_build_documents_produces_separate_documents_per_level(tmp_path):
    fake_table = MagicMock()
    fake_table.extract.return_value = [["Method", "Effect"], ["L1", "Sparsity"]]

    page = _make_fake_page(
        text="Regularization prevents overfitting in models with many parameters and features.",
        tables=[fake_table],
    )
    fake_doc = MagicMock()
    fake_doc.is_encrypted = False
    fake_doc.__len__.return_value = 1
    fake_doc.__getitem__.return_value = page

    with patch("app.ingest_pdf.fitz.open", return_value=fake_doc):
        documents, warnings = build_documents(
            "/fake/doc.pdf",
            chunk_size=200,
            overlap=30,
            generation_url="http://fake:8002",
            with_summaries=False,
        )

    levels = {d["metadata"]["level"] for d in documents}
    assert "chunk" in levels
    assert "table" in levels


# --- iter_build_documents: streaming progress events ---


def test_iter_build_documents_yields_start_then_one_event_per_page_then_result(
    tmp_path,
):
    page1 = _make_fake_page(text="First page of real content here.")
    page2 = _make_fake_page(text="Second page of real content here.")
    fake_doc = MagicMock()
    fake_doc.is_encrypted = False
    fake_doc.__len__.return_value = 2
    fake_doc.__getitem__.side_effect = lambda i: [page1, page2][i]

    with patch("app.ingest_pdf.fitz.open", return_value=fake_doc):
        events = list(
            iter_build_documents(
                "/fake/doc.pdf",
                chunk_size=200,
                overlap=30,
                generation_url="http://fake:8002",
                with_summaries=False,
            )
        )

    assert events[0] == {"type": "start", "num_pages": 2}
    page_events = [e for e in events if e["type"] == "page_extracted"]
    assert [e["page"] for e in page_events] == [1, 2]
    assert events[-1]["type"] == "result"
    assert len(events[-1]["documents"]) > 0


def test_build_documents_result_matches_draining_iter_build_documents(tmp_path):
    """build_documents() must stay a thin wrapper -- same documents/warnings
    either way, whether you drain the generator yourself or call the
    non-streaming wrapper."""
    page = _make_fake_page(text="Some content for equivalence checking.")
    fake_doc = MagicMock()
    fake_doc.is_encrypted = False
    fake_doc.__len__.return_value = 1
    fake_doc.__getitem__.return_value = page

    with patch("app.ingest_pdf.fitz.open", return_value=fake_doc):
        wrapper_documents, wrapper_warnings = build_documents(
            "/fake/doc.pdf",
            chunk_size=200,
            overlap=30,
            generation_url="http://fake:8002",
            with_summaries=False,
        )
        generator_result = next(
            e
            for e in iter_build_documents(
                "/fake/doc.pdf",
                chunk_size=200,
                overlap=30,
                generation_url="http://fake:8002",
                with_summaries=False,
            )
            if e["type"] == "result"
        )

    assert wrapper_documents == generator_result["documents"]
    assert wrapper_warnings == generator_result["warnings"]


# --- iter_index_documents: streaming progress events ---


def test_iter_index_documents_yields_progress_per_batch():
    documents = [{"id": f"doc{i}", "text": "x", "metadata": {}} for i in range(5)]

    with patch("app.ingest_pdf.requests.post") as mock_post:
        mock_post.return_value.raise_for_status.return_value = None
        events = list(iter_index_documents(documents, "http://fake:8001", batch_size=2))

    assert [e["indexed"] for e in events] == [2, 4, 5]
    assert all(e["total"] == 5 for e in events)
    assert mock_post.call_count == 3


def test_iter_index_documents_reports_nothing_to_index_for_empty_list():
    events = list(iter_index_documents([], "http://fake:8001"))
    assert events == [
        {
            "type": "index_progress",
            "indexed": 0,
            "total": 0,
            "message": "Nothing to index.",
        }
    ]


# --- Summarization path (with_summaries=True) --
# NOTE: this is exactly the path that had zero coverage when summarize()
# itself was accidentally deleted during a refactor (build_documents() /
# iter_build_documents() split) -- nothing here caught it because every
# other test in this file passes with_summaries=False. These tests exist
# specifically to make sure that class of regression can't happen silently
# again.


def test_build_documents_with_summaries_produces_page_and_document_summaries():
    page = _make_fake_page(text="Real page content about photosynthesis.")
    fake_doc = MagicMock()
    fake_doc.is_encrypted = False
    fake_doc.__len__.return_value = 1
    fake_doc.__getitem__.return_value = page

    with (
        patch("app.ingest_pdf.fitz.open", return_value=fake_doc),
        patch("app.ingest_pdf.requests.post") as mock_post,
    ):
        mock_post.return_value.raise_for_status.return_value = None
        mock_post.return_value.json.return_value = {"answer": "A summary."}

        documents, warnings = build_documents(
            "/fake/doc.pdf",
            chunk_size=200,
            overlap=30,
            generation_url="http://fake-generation:8002",
            with_summaries=True,
        )

    levels = [d["metadata"]["level"] for d in documents]
    assert "page_summary" in levels
    assert "document_summary" in levels
    assert warnings == []
    # One call for the page summary, one for the document-level summary.
    assert mock_post.call_count == 2
    for call in mock_post.call_args_list:
        assert call.args[0] == "http://fake-generation:8002/generate"


def test_iter_build_documents_forwards_headers_to_summarize_calls():
    """The actual fix this test suite was missing: headers passed into
    iter_build_documents() must reach the summarize() calls, so ingestion's
    page/document summary spans nest under the caller's trace instead of
    each becoming a disconnected one (see app/tracing.py in the ingestion
    service)."""
    page = _make_fake_page(text="Some real content for header propagation check.")
    fake_doc = MagicMock()
    fake_doc.is_encrypted = False
    fake_doc.__len__.return_value = 1
    fake_doc.__getitem__.return_value = page

    trace_headers = {
        "X-Langfuse-Trace-Id": "abc123",
        "X-Langfuse-Parent-Span-Id": "def456",
    }

    with (
        patch("app.ingest_pdf.fitz.open", return_value=fake_doc),
        patch("app.ingest_pdf.requests.post") as mock_post,
    ):
        mock_post.return_value.raise_for_status.return_value = None
        mock_post.return_value.json.return_value = {"answer": "A summary."}

        list(
            iter_build_documents(
                "/fake/doc.pdf",
                chunk_size=200,
                overlap=30,
                generation_url="http://fake:8002",
                with_summaries=True,
                headers=trace_headers,
            )
        )

    assert mock_post.call_count >= 1
    for call in mock_post.call_args_list:
        assert call.kwargs["headers"] == trace_headers


def test_summarize_failure_is_recorded_as_a_warning_not_a_crash():
    page = _make_fake_page(text="Content that will fail to summarize.")
    fake_doc = MagicMock()
    fake_doc.is_encrypted = False
    fake_doc.__len__.return_value = 1
    fake_doc.__getitem__.return_value = page

    with (
        patch("app.ingest_pdf.fitz.open", return_value=fake_doc),
        patch(
            "app.ingest_pdf.requests.post",
            side_effect=requests.exceptions.ConnectionError("refused"),
        ),
    ):
        documents, warnings = build_documents(
            "/fake/doc.pdf",
            chunk_size=200,
            overlap=30,
            generation_url="http://fake:8002",
            with_summaries=True,
        )

    assert any("summarization failed" in w for w in warnings)
    assert not any(d["metadata"]["level"] == "page_summary" for d in documents)
