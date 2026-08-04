import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock
from ingest_pdf import build_documents, extract_pages, open_pdf, _clean_text, PdfIngestionError


# --- Text cleaning ---

def test_clean_text_removes_null_bytes():
    assert _clean_text("hello\x00world") == "helloworld"


def test_clean_text_removes_control_characters_but_keeps_newlines():
    dirty = "line one\x01\x02\nline two\x7f"
    assert _clean_text(dirty) == "line one\nline two"


def test_clean_text_collapses_excess_whitespace():
    assert _clean_text("too    many     spaces") == "too many spaces"


def test_clean_text_handles_empty_input():
    assert _clean_text("") == ""
    assert _clean_text(None) == ""


# --- Corrupt/unreadable files ---

def test_open_pdf_raises_clear_error_for_garbage_file(tmp_path):
    fake_pdf = tmp_path / "not_a_real.pdf"
    fake_pdf.write_bytes(b"this is not a pdf file at all, just garbage bytes")

    with pytest.raises(PdfIngestionError, match="Could not open"):
        open_pdf(str(fake_pdf))


def test_open_pdf_raises_clear_error_for_missing_file():
    with pytest.raises(PdfIngestionError):
        open_pdf("/tmp/this_file_does_not_exist_at_all.pdf")


# --- Per-page extraction failures ---

def test_extract_pages_skips_failing_page_but_keeps_others():
    good_page = MagicMock()
    good_page.extract_text.return_value = "This page works fine."

    bad_page = MagicMock()
    bad_page.extract_text.side_effect = Exception("corrupted content stream")

    fake_reader = MagicMock()
    fake_reader.is_encrypted = False
    fake_reader.pages = [good_page, bad_page, good_page]

    with patch("ingest_pdf.PdfReader", return_value=fake_reader):
        pages, warnings = extract_pages("/fake/path.pdf")

    assert pages == ["This page works fine.", "", "This page works fine."]
    assert len(warnings) == 1
    assert "Page 2" in warnings[0]
    assert "failed to extract" in warnings[0]


def test_extract_pages_warns_on_scanned_image_page_with_no_text():
    blank_page = MagicMock()
    blank_page.extract_text.return_value = ""

    fake_reader = MagicMock()
    fake_reader.is_encrypted = False
    fake_reader.pages = [blank_page]

    with patch("ingest_pdf.PdfReader", return_value=fake_reader):
        pages, warnings = extract_pages("/fake/path.pdf")

    assert pages == [""]
    assert any("no extractable text" in w for w in warnings)
    assert any("entirely scanned images" in w for w in warnings)


# --- Encrypted PDFs ---

def test_open_pdf_decrypts_with_empty_password():
    fake_reader = MagicMock()
    fake_reader.is_encrypted = True

    with patch("ingest_pdf.PdfReader", return_value=fake_reader):
        open_pdf("/fake/encrypted.pdf")

    fake_reader.decrypt.assert_called_once_with("")


def test_open_pdf_raises_clear_error_when_decrypt_fails():
    fake_reader = MagicMock()
    fake_reader.is_encrypted = True
    fake_reader.decrypt.side_effect = Exception("wrong password")

    with patch("ingest_pdf.PdfReader", return_value=fake_reader):
        with pytest.raises(PdfIngestionError, match="password-protected"):
            open_pdf("/fake/encrypted.pdf")


# --- build_documents integration with warnings ---

@patch("ingest_pdf.summarize")
def test_build_documents_still_produces_chunks_for_good_pages_despite_bad_page(mock_summarize):
    mock_summarize.side_effect = lambda url, instruction, texts: "fake summary"

    good_page = MagicMock()
    good_page.extract_text.return_value = "Real content here that is long enough to chunk properly for the test."

    bad_page = MagicMock()
    bad_page.extract_text.side_effect = Exception("broken stream")

    fake_reader = MagicMock()
    fake_reader.is_encrypted = False
    fake_reader.pages = [good_page, bad_page]

    with patch("ingest_pdf.PdfReader", return_value=fake_reader):
        documents, warnings = build_documents(
            "/fake/path.pdf", chunk_size=200, overlap=30,
            generation_url="http://fake:8002", with_summaries=True,
        )

    # good page's chunk should still be indexed
    chunk_docs = [d for d in documents if d["metadata"]["level"] == "chunk"]
    assert len(chunk_docs) >= 1
    assert chunk_docs[0]["metadata"]["page"] == 1

    # the failure should be reported, not silently swallowed
    assert any("Page 2" in w for w in warnings)


def test_build_documents_raises_pdf_ingestion_error_for_unreadable_file(tmp_path):
    fake_pdf = tmp_path / "garbage.pdf"
    fake_pdf.write_bytes(b"definitely not a pdf")

    with pytest.raises(PdfIngestionError):
        build_documents(
            str(fake_pdf), chunk_size=200, overlap=30,
            generation_url="http://fake:8002", with_summaries=False,
        )