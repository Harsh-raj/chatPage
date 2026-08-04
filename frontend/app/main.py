import os
import time
import json
import requests
import streamlit as st

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://localhost:8000")

st.title("Retrieval Augmented Generation")


def check_backend_ready() -> dict | None:
    """Returns the gateway's aggregated health response, or None if the
    gateway itself isn't reachable yet."""
    try:
        response = requests.get(f"{GATEWAY_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException:
        return None


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    cols = st.columns(len(sources))
    for i, (col, src) in enumerate(zip(cols, sources), start=1):
        with col:
            with st.popover(f"📄 {i}", use_container_width=True):
                st.markdown(f"**`{src['id']}`**")
                st.caption(f"score: {src['score']:.3f}")
                st.write(src["text"])


def stream_query(query_text: str, sources_holder: list, error_holder: dict):
    """
    Streams tokens from the gateway. If the backend turns out to be
    unreachable or unready mid-conversation (e.g. a container restarted
    after this session already passed the initial readiness gate),
    error_holder["backend_down"] is set to True so the caller can reset
    the readiness gate rather than leaving a stale "ready" flag that
    would keep producing failed queries indefinitely.
    """
    try:
        with requests.post(
            f"{GATEWAY_URL}/query/stream",
            json={"query": query_text, "top_k": 5},
            stream=True,
            timeout=150,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                event = json.loads(line)
                if event["type"] == "sources":
                    sources_holder.extend(event["sources"])
                elif event["type"] == "token":
                    yield event["text"]
                elif event["type"] == "done":
                    return
    except requests.exceptions.ConnectionError:
        error_holder["backend_down"] = True
        yield "Could not connect to the gateway service. Re-checking backend readiness..."
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else None
        if status and status >= 500:
            # A 5xx from the gateway usually means one of its downstream
            # services (retrieval/generation) isn't actually ready, even
            # though this session previously passed the readiness gate.
            error_holder["backend_down"] = True
            yield f"A backend service isn't ready yet (error {status}). Re-checking backend readiness..."
        else:
            yield f"Gateway returned an error: {e}"
    except requests.exceptions.ReadTimeout:
        yield "The model is taking longer than expected to respond. This can happen on the first request after Ollama has been idle. Please try again."


# --- Readiness gate: block the chat UI until every backend service is up ---
if "backend_ready" not in st.session_state:
    st.session_state.backend_ready = False

if not st.session_state.backend_ready:
    status_placeholder = st.empty()
    # Retrieval loads TWO models on startup (the embedder and, if enabled,
    # the reranker) -- on CPU-only hardware this has been observed to take
    # 3+ minutes on its own, so the wait budget needs real headroom above that.
    max_wait_seconds = 360
    poll_interval = 3
    waited = 0

    while waited < max_wait_seconds:
        health = check_backend_ready()

        if health and health.get("status") == "ok":
            st.session_state.backend_ready = True
            status_placeholder.empty()
            st.rerun()

        with status_placeholder.container():
            minutes, seconds = divmod(waited, 60)
            st.info(
                f"Starting up backend services... ({minutes}m {seconds}s elapsed)\n\n"
                f"This can take several minutes the first time, while the retrieval "
                f"service loads its embedding model and reranker model."
            )
            if health and "services" in health:
                for name, info in health["services"].items():
                    icon = "done" if info.get("ready") else "waiting"
                    st.write(
                        f"[{icon}] {name}: {'ready' if info.get('ready') else 'starting...'}")
            else:
                st.write("Waiting for gateway to respond...")

        time.sleep(poll_interval)
        waited += poll_interval

    st.error(
        f"Backend services haven't started after {max_wait_seconds // 60} minutes. "
        f"This is longer than expected -- check docker compose logs retrieval for errors "
        f"(e.g. an out-of-memory kill), or click below to keep waiting."
    )
    if st.button("Check again"):
        st.rerun()
    st.stop()

# --- Main chat UI, only reached once backend_ready is True ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            render_sources(message["sources"])

if prompt := st.chat_input("Provide your query here!"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        sources: list = []
        error_holder = {"backend_down": False}
        full_response = st.write_stream(
            stream_query(prompt, sources, error_holder))
        render_sources(sources)

    st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": sources,
    })

    # If the backend turned out to be unreachable mid-conversation, drop the
    # stale "ready" flag so the readiness gate re-appears and genuinely
    # re-checks, instead of silently repeating this failure on every
    # subsequent query.
    if error_holder["backend_down"]:
        st.session_state.backend_ready = False
        st.rerun()
