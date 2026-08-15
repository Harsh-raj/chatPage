"""
Langfuse tracing helpers for the gateway.

The gateway is the root of every trace: it either accepts an externally
supplied trace_id (the RAGAS eval script deliberately pins one so it can
attach eval scores to the exact trace it just produced -- see
eval/ragas/run_ragas_eval.py) or mints a fresh one via
langfuse.create_trace_id(). It then passes its own trace_id and current
span_id downstream to retrieval and generation as plain HTTP headers, so
Langfuse renders one nested trace across all three services instead of
three disconnected ones.

get_client() reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST
from the environment and returns a process-wide singleton -- no explicit
Langfuse(...) construction needed here. If those env vars aren't set, the
client no-ops (tracing calls become harmless no-ops rather than errors),
so this is safe to leave in place even before Langfuse is configured.
"""

from __future__ import annotations

from typing import Mapping, Optional

from langfuse import Langfuse, get_client

TRACE_ID_HEADER = "X-Langfuse-Trace-Id"
PARENT_SPAN_ID_HEADER = "X-Langfuse-Parent-Span-Id"


def langfuse() -> Langfuse:
    return get_client()


def incoming_trace_id(headers: Mapping[str, str]) -> Optional[str]:
    """An external caller (e.g. the RAGAS eval script) can pin this query to
    a trace_id it already knows. Falls back to minting a new one if absent."""
    return headers.get(TRACE_ID_HEADER)


def downstream_headers(span) -> dict:
    """Headers to attach to outgoing calls to retrieval/generation so their
    spans nest under the given span, in the same trace.

    Takes the span object directly (its .trace_id / .id attributes) rather
    than reading "current" context off the client. That ambient-context
    lookup only works within a `with start_as_current_observation(...)`
    block on the same thread/task -- it breaks for the streaming endpoints,
    where the generator that makes these downstream calls runs after the
    handler function returns, sometimes on a different thread. Reading the
    ids straight off the span object works the same way in both cases.
    """
    headers = {}
    if getattr(span, "trace_id", None):
        headers[TRACE_ID_HEADER] = span.trace_id
    if getattr(span, "id", None):
        headers[PARENT_SPAN_ID_HEADER] = span.id
    return headers
