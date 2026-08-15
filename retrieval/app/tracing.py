"""
Langfuse tracing helpers for the retrieval service.

Reads the trace context propagated from the gateway (see
gateway/app/tracing.py) via the X-Langfuse-Trace-Id / X-Langfuse-Parent-Span-Id
headers, so this service's search span nests under the gateway's span in the
same trace, instead of starting a disconnected one. If those headers are
absent (e.g. someone curls this service directly), the span just becomes the
root of its own trace -- still useful, just not nested.

get_client() reads LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST
from the environment and returns a process-wide singleton. If unset, the
client no-ops rather than erroring, so this is safe even before Langfuse is
configured.
"""

from __future__ import annotations

from typing import Mapping, Optional

from langfuse import Langfuse, get_client
from langfuse.types import TraceContext

TRACE_ID_HEADER = "X-Langfuse-Trace-Id"
PARENT_SPAN_ID_HEADER = "X-Langfuse-Parent-Span-Id"


def langfuse() -> Langfuse:
    return get_client()


def trace_context_from_headers(headers: Mapping[str, str]) -> Optional[TraceContext]:
    trace_id = headers.get(TRACE_ID_HEADER)
    if not trace_id:
        return None
    context: TraceContext = {"trace_id": trace_id}
    parent_span_id = headers.get(PARENT_SPAN_ID_HEADER)
    if parent_span_id:
        context["parent_span_id"] = parent_span_id
    return context
