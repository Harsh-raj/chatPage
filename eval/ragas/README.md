# RAGAS evaluation

Scores end-to-end answer quality -- faithfulness, answer relevancy, context
precision, context recall -- for the full RAG pipeline (retrieval +
generation, exercised together through the gateway's `/query` endpoint).

## How this differs from `eval/run_golden_eval.py` (on the `evaluation` branch)

| | `run_golden_eval.py` | `run_ragas_eval.py` (this one) |
|---|---|---|
| Layer tested | Retrieval only | Retrieval + generation, end-to-end |
| Judge | None -- keyword/source matching | External LLM (Anthropic API) |
| Cost | Free, instant | Costs API money, slower |
| Where it runs | Every push, in CI | Manual / on-demand only |

They're complementary, not redundant: the golden-question check is a fast
regression guard for retrieval changes (chunking, embeddings, hybrid search
weighting, reranker threshold); RAGAS is the deeper, slower check on whether
the generated *answer* is actually faithful to what was retrieved and
relevant to the question.

## Setup

```bash
pip install -r eval/ragas/requirements.txt
```

Bring up the full stack with **real** components -- evaluating stub output
scores the stub, not your system:

```bash
USE_REAL_EMBEDDER=1 USE_REAL_STORE=1 USE_REAL_LLM=1 docker compose up --build
```

## Judge: Anthropic (default if a key is set) or local Ollama (fallback)

The script picks a judge automatically:

```bash
# Anthropic judge -- set both of these
export ANTHROPIC_API_KEY=sk-ant-...
export RAGAS_JUDGE_MODEL=claude-sonnet-4-6   # check https://docs.claude.com for the current model list

# Local Ollama judge -- just don't set ANTHROPIC_API_KEY. Defaults to
# qwen2.5:3b-instruct at http://localhost:11434. Override if needed:
export OLLAMA_JUDGE_MODEL=qwen2.5:3b-instruct
export OLLAMA_BASE_URL=http://localhost:11434
```

### Judge quality -- read this before trusting a local-judge run

Every metric here relies on the judge model reliably producing structured
output (breaking an answer into claims, returning a score with a rationale,
etc.), which `ragas` extracts via tool-calling/JSON mode under the hood.
Claude handles this very reliably. `qwen2.5:3b-instruct` is a 3B model
running locally with no dedicated GPU on your setup -- it's usable as a free
offline fallback, but expect:
- More outright failures on a given row (the script logs `[warn]` and
  records `None` for that metric rather than crashing the whole run)
- Noisier, less consistent scores between runs on the same data
- **Scores are not comparable across judges.** A 0.85 faithfulness from
  Ollama and a 0.85 from Claude are not the same claim. Each results file
  records which judge produced it (`judge_provider`, `judge_model`) --
  don't average or trend scores across a judge switch.

Use the local judge for fast iteration while you're still shaping the
golden set or debugging the pipeline; switch to Claude for the numbers you
actually want to trust or track over time.

## Run

```bash
python eval/ragas/run_ragas_eval.py \
    --gateway-url http://localhost:8000 \
    --retrieval-url http://localhost:8001
```

This seeds `seed_documents.json` into retrieval, runs every question in
`golden_dataset.json` through the gateway, scores each answer, prints a
per-question table and averages, and writes full per-row detail (including
the actual retrieved contexts and answer text) to
`eval/ragas/results/ragas_run_<timestamp>.json`.

Already indexed the seed documents from a previous run? Add `--skip-seed`.

## Extending the golden set

Add entries to `golden_dataset.json`: `question` and `ground_truth` are
required (`ground_truth` feeds `context_recall` and `context_precision`).
Set `"expect_low_confidence": true` for deliberately out-of-scope questions
-- these are the most useful check on `faithfulness`, since they catch the
model fabricating a plausible-sounding answer from irrelevant retrieved
context rather than admitting it doesn't know.

## Why an external API judge, and why local embeddings

The judge LLM (Anthropic) needs to reason about entailment between the
answer and the retrieved context, which benefits from a stronger model than
what's running locally in `generation` (`qwen2.5:3b-instruct` via Ollama).
`answer_relevancy` only needs embedding similarity, not reasoning, so it
uses the same local `bge-small-en-v1.5` model retrieval already runs --
that keeps embedding calls free and keeps the only real per-run cost to the
LLM-judged metrics.

## Cost and CI

This is not wired into `ci.yml`. Running an LLM-as-judge eval on every push
would mean an Anthropic API key sitting in every PR's CI run and real
dollars spent on every commit. Run it manually, or use
`.github/workflows/ragas-eval.yml` (`workflow_dispatch`, manually triggered
from the Actions tab) if you want a button instead of a local shell.