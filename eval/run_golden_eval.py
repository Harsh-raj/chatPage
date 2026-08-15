"""
run_golden_eval.py -- runs the golden question set against a live retrieval
service and reports pass/fail per question. This is the reusable regression
check: run it after any change to chunking, embeddings, hybrid search
weighting, or the reranker threshold, and compare against previous runs.

Checks three things per question:
  1. Does the top-ranked result come from the expected document?
     (validates multi-document disambiguation)
  2. Do expected keywords appear in the top result's text?
     (a cheap proxy for "did we find the actually relevant passage")
  3. For out-of-scope questions, does retrieval correctly return either
     nothing or only low-confidence matches, rather than confidently
     returning an irrelevant passage?

This checks RETRIEVAL quality specifically (hitting retrieval's /search
directly), not full generation quality -- it's fast, doesn't require
Ollama to be running, and isolates exactly the layer multi-document
support depends on.

Usage:
    python run_golden_eval.py --retrieval-url http://localhost:8001
"""

import argparse
import json
import os

import requests


def load_golden_set(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


def run_single_question(retrieval_url: str, question: dict, top_k: int = 3) -> dict:
    response = requests.post(
        f"{retrieval_url}/search",
        json={"query": question["question"], "top_k": top_k},
        timeout=30,
    )
    response.raise_for_status()
    results = response.json()["results"]

    result = {
        "id": question["id"],
        "question": question["question"],
        "passed": True,
        "reasons": [],
        "top_results": [{"id": r["id"], "score": round(r["score"], 4)} for r in results],
    }

    if question.get("expect_low_confidence"):
        # Out-of-scope questions should either return nothing, or only
        # weak matches -- NOT a confident, irrelevant top result.
        if results and results[0]["score"] > 0.7:
            result["passed"] = False
            result["reasons"].append(
                f"Expected low confidence for an out-of-scope question, but got "
                f"score={results[0]['score']:.3f} on '{results[0]['id']}'"
            )
        return result

    if not results:
        result["passed"] = False
        result["reasons"].append("No results returned at all")
        return result

    top = results[0]
    expected_source = question.get("expected_source_contains")
    if expected_source:
        actual_source = top.get("metadata", {}).get("source", top["id"])
        if expected_source not in actual_source:
            result["passed"] = False
            result["reasons"].append(
                f"Top result came from '{actual_source}', expected something containing '{expected_source}'"
            )

    expected_keywords = question.get("expected_keywords", [])
    combined_text = " ".join(r["text"] for r in results).lower()
    missing_keywords = [kw for kw in expected_keywords if kw.lower() not in combined_text]
    if missing_keywords:
        result["passed"] = False
        result["reasons"].append(f"Missing expected keyword(s) in retrieved text: {missing_keywords}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden-set", default=os.path.join(os.path.dirname(__file__), "golden_questions.json"))
    parser.add_argument("--retrieval-url", default="http://localhost:8001")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    questions = load_golden_set(args.golden_set)
    results = []

    for q in questions:
        try:
            result = run_single_question(args.retrieval_url, q, top_k=args.top_k)
        except requests.exceptions.RequestException as e:
            result = {
                "id": q["id"],
                "question": q["question"],
                "passed": False,
                "reasons": [f"Request failed: {e}"],
                "top_results": [],
            }
        results.append(result)

        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {result['id']}: {result['question']}")
        if not result["passed"]:
            for reason in result["reasons"]:
                print(f"         -> {reason}")
        for r in result["top_results"][:2]:
            print(f"         top: {r['id']} (score={r['score']})")

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print()
    print(f"{passed}/{total} passed ({100 * passed / total:.0f}%)")

    return 0 if passed == total else 1


if __name__ == "__main__":
    exit(main())
