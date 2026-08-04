from app.chunking import chunk_text


def test_short_text_returns_single_chunk():
    chunks = chunk_text("Hello world", chunk_size=800, overlap=100)
    assert chunks == ["Hello world"]


def test_empty_text_returns_no_chunks():
    assert chunk_text("   ", chunk_size=800, overlap=100) == []


def test_long_text_splits_into_multiple_overlapping_chunks():
    text = "A" * 1000
    chunks = chunk_text(text, chunk_size=400, overlap=50)
    assert len(chunks) > 1
    # confirm overlap: end of chunk 1 should reappear at start of chunk 2
    assert chunks[0][-50:] == chunks[1][:50]


def test_rejects_invalid_overlap():
    import pytest
    with pytest.raises(ValueError):
        chunk_text("some text", chunk_size=100, overlap=100)
