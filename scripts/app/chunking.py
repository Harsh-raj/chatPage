"""
Chunking logic, kept as a standalone, testable function -- deliberately
independent of any HTTP/PDF code so it can be unit tested in isolation.
"""
from __future__ import annotations


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """
    Splits text into overlapping chunks by character count.

    chunk_size: target size of each chunk, in characters
    overlap: how many characters from the end of one chunk are repeated
             at the start of the next -- this preserves context that would
             otherwise be cut in half at a chunk boundary.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap  # step forward, but re-include the overlap region

    return chunks
