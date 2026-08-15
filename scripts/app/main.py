"""
FastAPI wrapper around the ingestion pipeline in ingest_pdf.py, so PDFs can
be uploaded and indexed through the Streamlit frontend instead of only via
the CLI (`python ingest_pdf.py some.pdf`, which still works unchanged).

This service owns the heavy, ingestion-only dependencies (PyMuPDF, the
Tesseract binary) so the lightweight Streamlit container doesn't need them.
"""

from __future__ import annotations

import json
import os
import tempfile

from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import StreamingResponse

from app.auth import ApiKeyMiddleware, auth_headers
from app.ingest_pdf import PdfIngestionError, iter_build_documents, iter_index_documents
from app.tracing import downstream_headers, langfuse, trace_context_from_headers

app = FastAPI(title="ingestion-service")
app.add_middleware(ApiKeyMiddleware)

RETRIEVAL_URL = os.environ.get("RETRIEVAL_URL", "http://localhost:8001")
GENERATION_URL = os.environ.get("GENERATION_URL", "http://localhost:8002")


@app.get("/health")
def health():
    return {"status": "ok"}


def _run_ingestion(
    tmp_path: str,
    filename: str,
    chunk_size: int,
    overlap: int,
    with_summaries: bool,
    with_tables: bool,
    with_ocr: bool,
    trace_context,
):
    """
    Shared generator driving the whole ingestion run: extraction -> indexing,
    wrapped in a single Langfuse span covering the full operation (ingestion
    doesn't map cleanly onto Langfuse's retriever/generation/embedding
    as_types, so this is a plain "tool" span; the /generate calls made
    internally for page/document summaries still get their own nested
    generation spans in the generation service, same as any other caller).
    Yields NDJSON-ready dicts; the caller is responsible for serializing.
    """
    lf = langfuse()
    span = lf.start_observation(
        name="ingest_pdf",
        as_type="tool",
        trace_context=trace_context,
        input={"filename": filename, "chunk_size": chunk_size, "overlap": overlap},
    )

    documents: list[dict] = []
    warnings: list[str] = []
    headers = {**auth_headers(), **downstream_headers(span)}
    try:
        for event in iter_build_documents(
            tmp_path,
            chunk_size,
            overlap,
            GENERATION_URL,
            with_summaries=with_summaries,
            with_tables=with_tables,
            with_ocr=with_ocr,
            headers=headers,
        ):
            if event["type"] == "result":
                documents = event["documents"]
                warnings = event["warnings"]
            else:
                yield event

        by_level: dict[str, int] = {}
        for d in documents:
            level = d["metadata"].get("level", "unknown")
            by_level[level] = by_level.get(level, 0) + 1
        yield {"type": "extraction_done", "counts": by_level, "warnings": warnings}

        for event in iter_index_documents(documents, RETRIEVAL_URL, headers=headers):
            yield event

        yield {
            "type": "done",
            "filename": filename,
            "num_documents": len(documents),
            "counts": by_level,
            "warnings": warnings,
        }
        span.update(
            output={
                "num_documents": len(documents),
                "counts": by_level,
                "warnings": warnings,
            }
        )
    except PdfIngestionError as e:
        yield {"type": "error", "message": str(e)}
        span.update(level="ERROR", status_message=str(e))
    finally:
        span.end()


@app.post("/ingest/stream")
async def ingest_stream(
    http_request: Request,
    file: UploadFile = File(...),
    chunk_size: int = Form(800),
    overlap: int = Form(100),
    with_summaries: bool = Form(True),
    with_tables: bool = Form(True),
    with_ocr: bool = Form(True),
):
    """
    Streams newline-delimited JSON progress events while a PDF is extracted,
    (optionally) summarized, and indexed -- one event per page/batch, so the
    frontend can render a live log instead of blocking silently for however
    long OCR + summarization takes on a large scanned document.
    """
    trace_context = trace_context_from_headers(http_request.headers)

    suffix = os.path.splitext(file.filename or "upload.pdf")[1] or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    def event_stream():
        try:
            for event in _run_ingestion(
                tmp_path,
                file.filename or "upload.pdf",
                chunk_size,
                overlap,
                with_summaries,
                with_tables,
                with_ocr,
                trace_context,
            ):
                yield json.dumps(event) + "\n"
        finally:
            os.remove(tmp_path)

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
