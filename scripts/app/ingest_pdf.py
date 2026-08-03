"""
ingest_pdf.py -- one-time (or per-document) offline ingestion.

Indexes THREE levels of content, all through the same /index endpoint:
  1. Raw chunks -- for specific, factual questions
  2. Page-level summaries -- for section-scoped overview questions
  3. One whole-document summary -- for broad/structural questions
     ("list the chapters", "summarize this document")

Resilient to common PDF problems: password-protected files (tries an empty
password, the common case for "restricted" rather than truly secret PDFs),
individual pages that fail to extract (corrupted streams, scanned-image-only
pages with no text layer, etc.), and control characters/null bytes that can
break downstream JSON encoding. Problems are reported clearly at the end;
a single bad page never aborts the whole ingestion run.

Usage:
    python ingest_pdf.py path/to/book.pdf \
        --retrieval-url http://localhost:8001 \
        --generation-url http://localhost:8002
"""
import argparse
import os
import re
import sys
import requests
from pypdf import PdfReader
from pypdf.errors import PdfReadError

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from chunking import chunk_text


class PdfIngestionError(Exception):
    """Raised when the PDF cannot be opened or read at all -- nothing to salvage."""


def _clean_text(text: str) -> str:
    """
    Strips null bytes and other control characters that can appear in
    poorly-encoded PDF text streams and would otherwise break JSON encoding
    or produce garbage in the vector index. Keeps newlines/tabs.
    """
    if not text:
        return ""
    # Remove null bytes and other non-printable control characters except \n and \t
    text = text.replace("\x00", "")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # Collapse excessive whitespace left behind by stripped characters
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def open_pdf(pdf_path: str) -> PdfReader:
    """
    Opens a PDF, handling the two most common blockers:
      - The file isn't a valid/readable PDF at all -> raises PdfIngestionError
        with a clear message (nothing can be salvaged in this case).
      - The file is encrypted -> attempts an empty password, which succeeds
        for PDFs with "owner" restrictions but no real user password (a very
        common case). If that fails too, raises PdfIngestionError.
    """
    try:
        reader = PdfReader(pdf_path)
    except (PdfReadError, OSError) as e:
        raise PdfIngestionError(f"Could not open '{pdf_path}' as a PDF: {e}") from e

    if reader.is_encrypted:
        try:
            reader.decrypt("")
        except Exception as e:
            raise PdfIngestionError(
                f"'{pdf_path}' is password-protected and could not be opened "
                f"with an empty password. A real password is required: {e}"
            ) from e

    return reader


def extract_pages(pdf_path: str) -> tuple[list[str], list[str]]:
    """
    Returns (page_texts, warnings). page_texts has one entry per page --
    pages that fail to extract are included as an empty string, not
    dropped, so page numbering stays consistent with the actual PDF.
    Warnings describe anything that went wrong, for the caller to report.
    """
    reader = open_pdf(pdf_path)
    page_texts: list[str] = []
    warnings: list[str] = []

    for i, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text() or ""
        except Exception as e:
            warnings.append(f"Page {i}: failed to extract text ({e}). Skipped.")
            page_texts.append("")
            continue

        cleaned = _clean_text(raw_text)
        if not cleaned:
            warnings.append(
                f"Page {i}: no extractable text (likely a scanned image with no "
                f"text layer, or an empty page). Skipped."
            )
        page_texts.append(cleaned)

    if not any(page_texts):
        warnings.append(
            "No text could be extracted from ANY page. This PDF may be entirely "
            "scanned images -- consider running OCR on it first."
        )

    return page_texts, warnings


def summarize(generation_url: str, instruction: str, source_texts: list[str]) -> str:
    response = requests.post(
        f"{generation_url}/generate",
        json={"query": instruction, "context_chunks": source_texts},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["answer"]


def build_documents(
    pdf_path: str,
    chunk_size: int,
    overlap: int,
    generation_url: str,
    with_summaries: bool = True,
) -> tuple[list[dict], list[str]]:
    """Returns (documents, warnings)."""
    pages, warnings = extract_pages(pdf_path)
    documents = []
    page_summaries: list[str] = []

    for page_num, page_text in enumerate(pages, start=1):
        if not page_text:
            continue
        chunks = chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
        for chunk_idx, chunk in enumerate(chunks):
            documents.append({
                "id": f"{pdf_path}::page{page_num}::chunk{chunk_idx}",
                "text": chunk,
                "metadata": {"source": pdf_path, "page": page_num, "level": "chunk"},
            })

    if not with_summaries:
        return documents, warnings

    for page_num, page_text in enumerate(pages, start=1):
        if not page_text:
            continue
        print(f"Summarizing page {page_num}/{len(pages)}...")
        try:
            summary = summarize(
                generation_url,
                "Summarize the key topics and content of this page in 2-3 sentences. "
                "If it introduces a chapter or section title, mention it explicitly.",
                [page_text],
            )
        except requests.exceptions.RequestException as e:
            warnings.append(f"Page {page_num}: summarization failed ({e}). Skipped summary for this page.")
            continue
        page_summaries.append(summary)
        documents.append({
            "id": f"{pdf_path}::page{page_num}::summary",
            "text": summary,
            "metadata": {"source": pdf_path, "page": page_num, "level": "page_summary"},
        })

    if page_summaries:
        print("Building document-level summary...")
        try:
            doc_summary = summarize(
                generation_url,
                "Based on these page summaries, write a concise overview of what "
                "this entire document covers. Explicitly list any chapter or "
                "section titles you can identify, and briefly describe the main topics.",
                page_summaries,
            )
            documents.append({
                "id": f"{pdf_path}::document_summary",
                "text": doc_summary,
                "metadata": {"source": pdf_path, "level": "document_summary"},
            })
        except requests.exceptions.RequestException as e:
            warnings.append(f"Document-level summary failed ({e}). Skipped.")

    return documents, warnings


def index_documents(documents: list[dict], retrieval_url: str, batch_size: int = 50) -> None:
    if not documents:
        print("Nothing to index -- no usable content was extracted from this PDF.")
        return
    total = len(documents)
    for start in range(0, total, batch_size):
        batch = documents[start:start + batch_size]
        response = requests.post(
            f"{retrieval_url}/index",
            json={"documents": batch},
            timeout=120,
        )
        response.raise_for_status()
        print(f"Indexed {min(start + batch_size, total)}/{total} chunks")


def main():
    parser = argparse.ArgumentParser(description="Ingest a PDF into the retrieval service")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument("--retrieval-url", default="http://localhost:8001")
    parser.add_argument("--generation-url", default="http://localhost:8002")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--no-summaries", action="store_true", help="Skip page/document summary generation")
    args = parser.parse_args()

    print(f"Extracting text from {args.pdf_path} ...")
    try:
        documents, warnings = build_documents(
            args.pdf_path, args.chunk_size, args.overlap,
            args.generation_url, with_summaries=not args.no_summaries,
        )
    except PdfIngestionError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    if warnings:
        print(f"\n{len(warnings)} issue(s) encountered (ingestion continued anyway):")
        for w in warnings:
            print(f"  - {w}")
        print()

    print(f"Built {len(documents)} documents (chunks + summaries). Indexing into {args.retrieval_url} ...")
    index_documents(documents, args.retrieval_url)
    print("Done.")


if __name__ == "__main__":
    main()