"""
run_ragas_eval.py -- end-to-end RAGAS evaluation of the full RAG pipeline
(retrieval + generation, exercised together through the gateway's /query
endpoint).

This is different from eval/run_golden_eval.py (on the `evaluation` branch),
which checks retrieval quality only, with no LLM involved, and is cheap
enough to run in CI on every push. This script judges GENERATION quality --
faithfulness, answer relevancy, context precision, context recall -- using
an external LLM as judge (Anthropic's API). It costs real API money per run,
so it is deliberately NOT wired into the CI pipeline that runs on every
push. Run it manually, or trigger it on demand via a workflow_dispatch
GitHub Actions job (see .github/workflows/ragas-eval.yml).

Judge selection:
  - If ANTHROPIC_API_KEY is set: judges with Anthropic Claude (also requires
    RAGAS_JUDGE_MODEL, e.g. "claude-sonnet-4-6" -- check
    https://docs.claude.com for the current model list).
  - If ANTHROPIC_API_KEY is NOT set: falls back to the local Ollama model
    already in your stack (OLLAMA_JUDGE_MODEL, default "qwen2.5:3b-instruct",
    reached via OLLAMA_BASE_URL, default "http://localhost:11434"). Free and
    offline, but a materially weaker judge -- see the "Judge quality" note
    in eval/ragas/README.md before trusting these scores, and don't compare
    scores across the two judges as if they were on the same scale.

Also requires the full stack running with REAL components, not stubs --
USE_REAL_EMBEDDER=1, USE_REAL_STORE=1 (optional but recommended),
USE_REAL_LLM=1. Evaluating against stub responses would just be scoring the
stub, not your system.

Langfuse: each question is pinned to a trace_id this script mints itself
(rather than letting the gateway generate one), sent via the
X-Langfuse-Trace-Id header. Once scored, each RAGAS metric is attached to
that exact trace as a Langfuse score (ragas_faithfulness,
ragas_answer_relevancy, etc.) -- so opening a trace in Langfuse shows both
what happened (the retrieval/generation spans) and how good the eval judged
it to be, in one place. Requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY
in the environment; without them this step is skipped (scores are still
computed and saved locally). Pass --no-langfuse-scores to skip it deliberately.

Usage (Anthropic judge):
    export ANTHROPIC_API_KEY=sk-ant-...
    export RAGAS_JUDGE_MODEL=claude-sonnet-4-6
    python run_ragas_eval.py

Usage (local Ollama judge, no API key):
    python run_ragas_eval.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

import httpx

SCRIPT_DIR = Path(__file__).resolve().parent

API_KEY = os.environ.get("API_KEY")


def _auth_headers() -> dict:
    """This script talks directly to retrieval and gateway, bypassing
    normal service-to-service call paths -- it needs the same shared
    API_KEY those services require of every caller (see app/auth.py in
    each service) to not get a 401."""
    return {"X-API-Key": API_KEY} if API_KEY else {}


def load_json(path: Path) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def seed_documents(retrieval_url: str, documents: list[dict]) -> None:
    response = httpx.post(
        f"{retrieval_url}/index",
        json={"documents": documents},
        headers=_auth_headers(),
        timeout=30.0,
    )
    response.raise_for_status()
    print(f"Seeded {len(documents)} documents into retrieval ({retrieval_url}).")


def run_query(gateway_url: str, question: str, top_k: int, trace_id: str) -> dict:
    """Pins this query to a trace_id we already know (rather than letting the
    gateway mint its own), so the RAGAS scores computed afterward can be
    attached to the exact trace that produced this answer -- see
    push_scores_to_langfuse() below."""
    response = httpx.post(
        f"{gateway_url}/query",
        json={"query": question, "top_k": top_k},
        headers={"X-Langfuse-Trace-Id": trace_id, **_auth_headers()},
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()


def push_scores_to_langfuse(lf, trace_id: str, scores: dict) -> None:
    """Attaches each RAGAS metric as a score on the trace that produced the
    answer being scored, so a trace in Langfuse shows both what happened
    (the retrieval/generation spans) and how good the eval judged it to be,
    in one place -- rather than eval results living only in a local JSON
    file with no link back to the run that produced them."""
    for name, value in scores.items():
        if not isinstance(value, (int, float)):
            continue  # metric failed on this row -- nothing to attach
        lf.create_score(
            trace_id=trace_id,
            name=f"ragas_{name}",
            value=value,
            data_type="NUMERIC",
        )


def build_judge_and_embeddings() -> tuple:
    """
    LLM judge: Anthropic Claude if ANTHROPIC_API_KEY is set (preferred --
    stronger, more reliable structured-output judging).

    If it isn't set, falls back to the local Ollama model already in your
    stack (qwen2.5:3b-instruct by default) via Ollama's OpenAI-compatible
    endpoint, so you can still run this eval offline/free. See the "Judge
    quality" note in eval/ragas/README.md before trusting fallback scores --
    a 3B local model is a materially weaker judge than Claude and is more
    prone to inconsistent or malformed structured-output responses, since
    ragas relies on tool-calling/JSON-mode support to extract its scoring
    rationale, and small local models support that less reliably than
    frontier API models.

    Embeddings: always a local sentence-transformers model, reusing the same
    bge-small-en-v1.5 model the retrieval service already uses for real
    embeddings (see retrieval/app/embeddings.py). This is free either way --
    only the LLM-judged metrics differ in cost/quality between the two paths.
    """
    from ragas.embeddings import embedding_factory
    from ragas.llms import llm_factory

    embeddings = embedding_factory("huggingface", "BAAI/bge-small-en-v1.5")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        from anthropic import Anthropic

        judge_model = os.environ.get("RAGAS_JUDGE_MODEL")
        if not judge_model:
            sys.exit(
                "ANTHROPIC_API_KEY is set but RAGAS_JUDGE_MODEL is not. Set it to a "
                "current Anthropic model id, e.g. 'claude-sonnet-4-6' -- check "
                "https://docs.claude.com for the current list."
            )
        print(f"Judge: Anthropic ({judge_model})")
        client = Anthropic(api_key=api_key)
        llm = llm_factory(judge_model, provider="anthropic", client=client)
        return llm, embeddings, judge_model

    # Fallback: local Ollama via its OpenAI-compatible endpoint.
    from openai import OpenAI

    ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    judge_model = os.environ.get("OLLAMA_JUDGE_MODEL", "qwen2.5:3b-instruct")
    print(
        f"ANTHROPIC_API_KEY not set -- falling back to local judge: "
        f"Ollama ({judge_model} at {ollama_base_url}). "
        f"Scores from this judge are NOT directly comparable to Anthropic-judged runs."
    )
    client = OpenAI(base_url=f"{ollama_base_url.rstrip('/')}/v1", api_key="ollama")
    llm = llm_factory(judge_model, provider="openai", client=client)
    return llm, embeddings, judge_model


def evaluate_row(metrics: dict, user_input: str, response: str, retrieved_contexts: list[str], reference: str) -> dict:
    """Scores a single question/answer/context set against every metric.
    Metrics that don't need a reference answer just ignore the extra kwarg
    via **kwargs on their .score() method."""
    scores = {}
    for name, metric in metrics.items():
        try:
            result = metric.score(
                user_input=user_input,
                response=response,
                retrieved_contexts=retrieved_contexts,
                reference=reference,
            )
            scores[name] = result.value
        except Exception as e:  # noqa: BLE001 -- keep going even if one metric fails on one row
            scores[name] = None
            print(f"  [warn] metric '{name}' failed on this row: {e}")
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gateway-url", default=os.environ.get("GATEWAY_URL", "http://localhost:8000"))
    parser.add_argument("--retrieval-url", default=os.environ.get("RETRIEVAL_URL", "http://localhost:8001"))
    parser.add_argument("--golden-set", type=Path, default=SCRIPT_DIR / "golden_dataset.json")
    parser.add_argument("--seed-documents", type=Path, default=SCRIPT_DIR / "seed_documents.json")
    parser.add_argument(
        "--skip-seed", action="store_true", help="Skip re-indexing seed documents (use if already indexed)"
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument(
        "--no-langfuse-scores",
        action="store_true",
        help="Skip attaching RAGAS scores to Langfuse traces (still computes and saves them locally)",
    )
    args = parser.parse_args()

    golden_questions = load_json(args.golden_set)

    if not args.skip_seed:
        seed_documents(args.retrieval_url, load_json(args.seed_documents))

    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithReference,
        ContextRecall,
        Faithfulness,
    )

    llm, embeddings, judge_model = build_judge_and_embeddings()
    metrics = {
        "faithfulness": Faithfulness(llm=llm),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings),
        "context_precision": ContextPrecisionWithReference(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }

    push_to_langfuse = not args.no_langfuse_scores
    lf = None
    if push_to_langfuse:
        from langfuse import get_client

        lf = get_client()
        if not (os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")):
            print(
                "LANGFUSE_PUBLIC_KEY/LANGFUSE_SECRET_KEY not set -- scores will be computed "
                "and saved locally, but not attached to any Langfuse trace."
            )
            push_to_langfuse = False

    rows = []
    for q in golden_questions:
        print(f"Querying: {q['question']}")
        # Pin this query to a trace_id we mint ourselves, rather than letting
        # the gateway generate one, so we can attach this row's RAGAS scores
        # to the exact trace that produced it (see push_scores_to_langfuse).
        trace_id = lf.create_trace_id() if lf else uuid.uuid4().hex
        result = run_query(args.gateway_url, q["question"], args.top_k, trace_id)
        contexts = [s["text"] for s in result["sources"]]

        scores = evaluate_row(
            metrics,
            user_input=q["question"],
            response=result["answer"],
            retrieved_contexts=contexts,
            reference=q.get("ground_truth", ""),
        )

        if push_to_langfuse:
            push_scores_to_langfuse(lf, trace_id, scores)

        rows.append(
            {
                "id": q["id"],
                "question": q["question"],
                "answer": result["answer"],
                "ground_truth": q.get("ground_truth", ""),
                "retrieved_contexts": contexts,
                "scores": scores,
                "trace_id": trace_id,
                "trace_url": result.get("trace_url"),
            }
        )

    if lf:
        # This is a short-lived script, not a long-running server -- flush
        # explicitly so the scores (and any spans Langfuse's SDK batched)
        # are actually sent before the process exits, rather than sitting in
        # an in-memory queue that never gets a chance to drain.
        lf.flush()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.output_dir / f"ragas_run_{timestamp}.json"
    output = {
        "judge_model": judge_model,
        "judge_provider": "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "ollama",
        "langfuse_scores_pushed": push_to_langfuse,
        "rows": rows,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print("\n=== Per-question scores ===")
    metric_names = list(metrics.keys())
    header = "id".ljust(8) + "".join(name.ljust(20) for name in metric_names)
    print(header)
    for row in rows:
        line = row["id"].ljust(8)
        for name in metric_names:
            value = row["scores"].get(name)
            line += (f"{value:.3f}" if isinstance(value, (int, float)) else "n/a").ljust(20)
        print(line)

    print("\n=== Averages ===")
    for name in metric_names:
        values = [r["scores"][name] for r in rows if isinstance(r["scores"].get(name), (int, float))]
        avg = sum(values) / len(values) if values else float("nan")
        print(f"{name}: {avg:.3f}" if values else f"{name}: n/a (all rows failed)")

    if push_to_langfuse:
        print("\n=== Traces (scores attached) ===")
        for row in rows:
            url = row.get("trace_url") or row["trace_id"]
            print(f"{row['id']}: {url}")

    print(f"\nSaved detailed results to {out_path}")


if __name__ == "__main__":
    main()
