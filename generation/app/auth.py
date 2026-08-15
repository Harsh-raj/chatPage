"""
API-key authentication for the generation service.

A single shared secret (API_KEY env var), checked via the X-API-Key header
on every request except a small exempt list (health checks, FastAPI's own
docs). This is deliberately a shared-secret model, not per-user accounts --
appropriate for this project's actual deployment (a single developer's
local/demo stack, see docs/adr/0004-local-only-cpu-stack.md), not a
multi-tenant production system.

Every service in this stack (gateway, retrieval, generation, scripts/
ingestion) enforces this independently with its own copy of this module,
same duplication tradeoff as app/tracing.py -- see
docs/adr/0003-microservice-boundaries.md. This matters here specifically
because EVERY service's port is mapped to the host (see docker-compose.yml),
not just gateway's: someone who can reach the host network could otherwise
call retrieval's /index directly, bypassing gateway entirely, and poison
the vector store without ever touching an "authenticated" endpoint. The key
has to be enforced at every service, not just the one meant to be the
front door.

If API_KEY is not set at all, auth is disabled and a loud warning is
printed at startup -- deliberately not a silent no-op, since "quietly
unprotected" is a worse failure mode for an auth feature than for, say,
optional Langfuse tracing.
"""

import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

API_KEY = os.environ.get("API_KEY")

EXEMPT_PATHS = {"/health", "/", "/docs", "/openapi.json", "/redoc"}

if not API_KEY:
    print(
        "[auth] WARNING: API_KEY is not set -- authentication is DISABLED "
        "on this service. Do not expose it to any network you don't fully "
        "trust. Set API_KEY (and the same value on every other service in "
        "this stack, plus the frontend) to enable it."
    )


class ApiKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not API_KEY or request.url.path in EXEMPT_PATHS:
            return await call_next(request)
        if request.headers.get("X-API-Key") != API_KEY:
            return JSONResponse({"detail": "Missing or invalid X-API-Key header"}, status_code=401)
        return await call_next(request)


def auth_headers() -> dict:
    """Headers to attach to this service's own outgoing calls to other
    services in the stack, so those calls pass the receiving service's auth
    check too -- every service enforces the key independently (see the
    module docstring above), including calls between services themselves,
    not just calls arriving from outside the stack."""
    return {"X-API-Key": API_KEY} if API_KEY else {}
