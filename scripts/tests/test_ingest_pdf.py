from unittest.mock import MagicMock, patch

import pytest
from app.ingest_pdf import (
    PdfIngestionError,
    _clean_text,
    build_documents,
    extract_page_content,
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

    with patch("app.ingest_pdf.fitz.open", return_value=fake_doc):
        with pytest.raises(PdfIngestionError, match="password-protected"):
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
    with patch("app.ingest_pdf._OCR_AVAILABLE", True), \
         patch("app.ingest_pdf.Image.open", return_value=MagicMock()), \
         patch("app.ingest_pdf.pytesseract.image_to_string", return_value="Recovered OCR text"):
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
    with patch("app.ingest_pdf._OCR_AVAILABLE", True), \
         patch("app.ingest_pdf.Image.open", return_value=MagicMock()), \
         patch("app.ingest_pdf.pytesseract.image_to_string", return_value="Diagram label text here"):
        content = extract_page_content(doc, 0, warnings, run_image_ocr=True)

    assert content["image_ocr_texts"] == ["Diagram label text here"]


def test_extract_page_content_discards_short_noisy_image_ocr_results():
    doc = MagicMock()
    fake_page = _make_fake_page(text="regular text", images=[(999,)])
    doc.__getitem__.return_value = fake_page
    doc.extract_image.return_value = {"image": b"fake-bytes"}

    warnings = []
    with patch("app.ingest_pdf._OCR_AVAILABLE", True), \
         patch("app.ingest_pdf.Image.open", return_value=MagicMock()), \
         patch("app.ingest_pdf.pytesseract.image_to_string", return_value="x"):  # too short/noisy
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
            "/fake/doc.pdf", chunk_size=200, overlap=30,
            generation_url="http://fake:8002", with_summaries=False,
        )

    levels = {d["metadata"]["level"] for d in documents}
    assert "chunk" in levels
    assert "table" in levels
