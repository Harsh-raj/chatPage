"""
ingest_pdf.py -- one-time (or per-document) offline ingestion.

Indexes multiple levels/modalities of content, all through the same
/index endpoint:
  1. Raw text chunks -- for specific, factual questions
  2. Tables (extracted and converted to markdown) -- indexed separately,
     since raw text extraction often mangles tabular data badly
  3. OCR text recovered from pages with no real text layer (scanned pages)
  4. OCR text recovered from embedded images (diagrams/figures with text
     labels) -- reuses the same OCR path, no extra model needed
  5. Page-level summaries
  6. One whole-document summary

Uses PyMuPDF (fitz) rather than pypdf: it provides text extraction, table
detection, image extraction, and page rasterization (needed for OCR) all
in one pure-pip package with no external system binary dependency, unlike
alternatives that need Poppler or ImageMagick installed separately.

OCR itself requires the Tesseract binary to be installed on the system
running this script (a pip-only Python wrapper cannot include the actual
OCR engine). If Tesseract isn't installed, OCR-dependent steps are skipped
with a clear warning rather than crashing the whole ingestion run.

Usage:
    python ingest_pdf.py path/to/book.pdf \
        --retrieval-url http://localhost:8001 \
        --generation-url http://localhost:8002
"""
import argparse
import io
import os
import re
import sys
import requests
import fitz

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from chunking import chunk_text

try:
    import pytesseract
    from PIL import Image
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False


class PdfIngestionError(Exception):
    """Raised when the PDF cannot be opened or read at all -- nothing to salvage."""


def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\x00", "")
    text = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def open_pdf(pdf_path: str) -> fitz.Document:
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise PdfIngestionError(f"Could not open '{pdf_path}' as a PDF: {e}") from e

    if doc.is_encrypted:
        if not doc.authenticate(""):
            raise PdfIngestionError(
                f"'{pdf_path}' is password-protected and could not be opened "
                f"with an empty password. A real password is required."
            )

    return doc


def _ocr_image(image: "Image.Image", context: str, warnings: list[str]) -> str:
    """Runs OCR on a PIL image, returning cleaned text (or empty string on
    failure). Failures are recorded as warnings, never raised, so a missing
    Tesseract install or an unreadable image doesn't abort ingestion."""
    if not _OCR_AVAILABLE:
        warnings.append(f"{context}: OCR skipped (pytesseract/Pillow not installed).")
        return ""
    try:
        return _clean_text(pytesseract.image_to_string(image))
    except Exception as e:
        warnings.append(f"{context}: OCR failed ({e}). Is the Tesseract binary installed?")
        return ""


def table_to_markdown(rows: list[list[str | None]]) -> str:
    """Converts extracted table rows into a markdown table string."""
    if not rows:
        return ""
    cleaned_rows = [[(_clean_text(cell) if cell else "") for cell in row] for row in rows]
    header, *body_rows = cleaned_rows
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in body_rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def extract_page_content(doc: fitz.Document, page_num_zero_based: int, warnings: list[str], run_image_ocr: bool = True) -> dict:
    """
    Returns a dict with the different content types found on one page:
        {"text": str, "tables": list[str], "image_ocr_texts": list[str]}
    A page number (1-based, for warnings) is derived from the index.
    """
    page = doc[page_num_zero_based]
    page_num = page_num_zero_based + 1
    result = {"text": "", "tables": [], "image_ocr_texts": []}

    # 1. Regular text extraction
    try:
        raw_text = page.get_text() or ""
    except Exception as e:
        warnings.append(f"Page {page_num}: failed to extract text ({e}). Skipped.")
        raw_text = ""
    result["text"] = _clean_text(raw_text)

    # 2. If no real text layer at all, fall back to OCR on the whole page
    #    (this is what a scanned page looks like: get_text() returns nothing)
    if not result["text"]:
        try:
            pix = page.get_pixmap(dpi=200)
            img = Image.open(io.BytesIO(pix.tobytes("png"))) if _OCR_AVAILABLE else None
            if img is not None:
                ocr_text = _ocr_image(img, f"Page {page_num}", warnings)
                if ocr_text:
                    result["text"] = ocr_text
                else:
                    warnings.append(
                        f"Page {page_num}: no extractable text and OCR found nothing "
                        f"(page may be blank, or OCR unavailable)."
                    )
            else:
                warnings.append(f"Page {page_num}: no extractable text (OCR unavailable to attempt recovery).")
        except Exception as e:
            warnings.append(f"Page {page_num}: OCR fallback failed ({e}).")

    # 3. Tables
    try:
        found_tables = page.find_tables()
        for table in found_tables.tables:
            try:
                rows = table.extract()
                md = table_to_markdown(rows)
                if md:
                    result["tables"].append(md)
            except Exception as e:
                warnings.append(f"Page {page_num}: a table was found but failed to extract ({e}).")
    except Exception as e:
        warnings.append(f"Page {page_num}: table detection failed ({e}).")

    # 4. OCR on embedded images (diagrams/figures with text labels) --
    #    reuses the same OCR path, skipped if OCR isn't available.
    if run_image_ocr and _OCR_AVAILABLE:
        try:
            for img_idx, img_info in enumerate(page.get_images(full=True)):
                xref = img_info[0]
                try:
                    base_image = doc.extract_image(xref)
                    img = Image.open(io.BytesIO(base_image["image"]))
                    ocr_text = _ocr_image(img, f"Page {page_num}, image {img_idx + 1}", warnings)
                    # Only keep OCR results that look like real text, not noise
                    if ocr_text and len(ocr_text) >= 8:
                        result["image_ocr_texts"].append(ocr_text)
                except Exception as e:
                    warnings.append(f"Page {page_num}, image {img_idx + 1}: could not process ({e}).")
        except Exception as e:
            warnings.append(f"Page {page_num}: image extraction failed ({e}).")

    return result


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
    with_tables: bool = True,
    with_ocr: bool = True,
) -> tuple[list[dict], list[str]]:
    doc = open_pdf(pdf_path)
    documents = []
    warnings: list[str] = []
    page_texts: list[str] = []

    for page_idx in range(len(doc)):
        page_num = page_idx + 1
        content = extract_page_content(doc, page_idx, warnings, run_image_ocr=with_ocr)
        page_texts.append(content["text"])

        # Raw text chunks
        if content["text"]:
            for chunk_idx, chunk in enumerate(chunk_text(content["text"], chunk_size, overlap)):
                documents.append({
                    "id": f"{pdf_path}::page{page_num}::chunk{chunk_idx}",
                    "text": chunk,
                    "metadata": {"source": pdf_path, "page": page_num, "level": "chunk"},
                })

        # Tables, indexed separately as clean markdown
        if with_tables:
            for table_idx, table_md in enumerate(content["tables"]):
                documents.append({
                    "id": f"{pdf_path}::page{page_num}::table{table_idx}",
                    "text": table_md,
                    "metadata": {"source": pdf_path, "page": page_num, "level": "table"},
                })

        # OCR'd image captions
        for img_idx, ocr_text in enumerate(content["image_ocr_texts"]):
            documents.append({
                "id": f"{pdf_path}::page{page_num}::image{img_idx}::ocr",
                "text": ocr_text,
                "metadata": {"source": pdf_path, "page": page_num, "level": "image_ocr"},
            })

    if not any(page_texts):
        warnings.append(
            "No text could be extracted or recovered via OCR from ANY page. "
            "Check that this PDF has real content and that Tesseract is installed if OCR was needed."
        )

    if not with_summaries:
        return documents, warnings

    page_summaries: list[str] = []
    for page_num, page_text in enumerate(page_texts, start=1):
        if not page_text:
            continue
        print(f"Summarizing page {page_num}/{len(page_texts)}...")
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
    parser.add_argument("--no-summaries", action="store_true")
    parser.add_argument("--no-tables", action="store_true")
    parser.add_argument("--no-ocr", action="store_true", help="Skip OCR for scanned pages and embedded images")
    args = parser.parse_args()

    if not _OCR_AVAILABLE and not args.no_ocr:
        print("Note: pytesseract/Pillow not installed -- OCR will be skipped automatically.")

    print(f"Extracting content from {args.pdf_path} ...")
    try:
        documents, warnings = build_documents(
            args.pdf_path, args.chunk_size, args.overlap,
            args.generation_url,
            with_summaries=not args.no_summaries,
            with_tables=not args.no_tables,
            with_ocr=not args.no_ocr,
        )
    except PdfIngestionError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    if warnings:
        print(f"\n{len(warnings)} issue(s) encountered (ingestion continued anyway):")
        for w in warnings:
            print(f"  - {w}")
        print()

    by_level = {}
    for d in documents:
        level = d["metadata"].get("level", "unknown")
        by_level[level] = by_level.get(level, 0) + 1
    print(f"Built {len(documents)} documents: {dict(by_level)}")

    print(f"Indexing into {args.retrieval_url} ...")
    index_documents(documents, args.retrieval_url)
    print("Done.")


if __name__ == "__main__":
    main()
