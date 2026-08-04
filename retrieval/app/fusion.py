"""
Combines dense (vector) and sparse (BM25) search results into one ranked
list, using Reciprocal Rank Fusion (RRF) -- a simple, well-established
method that combines rankings without needing the two systems' scores to
be on comparable scales (cosine similarity and BM25 scores aren't
directly comparable, but their *ranks* can be fairly combined).
"""
from __future__ import annotations


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """
    result_lists: multiple ranked lists, each a list of {"id": ..., ...} dicts,
                  already sorted best-first.
    k: RRF constant (60 is the commonly used default from the original paper).

    Returns a single list of {"id": ..., "rrf_score": ...} sorted by fused
    score, descending. Deduplicates by id, preserving one merged entry per id.
    """
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    for results in result_lists:
        for rank, item in enumerate(results):
            item_id = item["id"]
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank + 1)
            # Keep the richest payload seen for this id (dense results carry
            # full text/metadata; BM25 results are id+score only)
            if item_id not in payloads or len(item) > len(payloads[item_id]):
                payloads[item_id] = item

    fused = [
        {**payloads[item_id], "id": item_id, "rrf_score": score}
        for item_id, score in scores.items()
    ]
    fused.sort(key=lambda x: -x["rrf_score"])
    return fused