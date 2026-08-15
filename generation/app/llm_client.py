from __future__ import annotations
import os
import time
import json
import httpx
from typing import Iterator


class LLMClient:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError

    def generate_stream(self, prompt: str) -> Iterator[str]:
        """Yields the response incrementally. Default implementation falls
        back to non-streaming generate() as a single chunk, so subclasses
        aren't required to implement true streaming to satisfy the interface."""
        yield self.generate(prompt)


class StubLLMClient(LLMClient):
    """Deterministic offline stand-in. Echoes back a templated response for testing."""

    def generate(self, prompt: str) -> str:
        return f"[stub-llm response based on {len(prompt)} char prompt]"

    def generate_stream(self, prompt: str) -> Iterator[str]:
        text = self.generate(prompt)
        for word in text.split():
            yield word + " "


class OllamaLLMClient(LLMClient):
    """
    Real generation via an Ollama server.

    base_url defaults to reading OLLAMA_BASE_URL from the environment, falling
    back to the Docker-network hostname "ollama" (used when Ollama itself runs
    as a container, e.g. on Windows/Linux). On macOS, set OLLAMA_BASE_URL to
    "http://host.docker.internal:11434" instead, since Ollama runs natively
    there rather than as a container on this network.
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str = "qwen2.5:3b-instruct",
        max_retries: int = 3,
        retry_delay_seconds: float = 2.0,
    ):
        self.base_url = base_url or os.environ.get(
            "OLLAMA_BASE_URL", "http://ollama:11434")
        self.model = model
        self.max_retries = max_retries
        self.retry_delay_seconds = retry_delay_seconds

    def generate(self, prompt: str) -> str:
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = httpx.post(
                    f"{self.base_url}/api/generate",
                    json={"model": self.model,
                          "prompt": prompt, "stream": False},
                    timeout=120.0,
                )
                response.raise_for_status()
                return response.json()["response"]
            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_seconds)
        raise RuntimeError(
            f"Could not reach Ollama at {self.base_url} after {self.max_retries} attempts"
        ) from last_error

    def generate_stream(self, prompt: str) -> Iterator[str]:
        """
        Streams Ollama's response as it's generated. Ollama's streaming API
        returns newline-delimited JSON, one object per line, each containing
        a "response" fragment and a "done" flag on the final line.
        """
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                with httpx.stream(
                    "POST",
                    f"{self.base_url}/api/generate",
                    json={"model": self.model, "prompt": prompt, "stream": True},
                    timeout=120.0,
                ) as response:
                    response.raise_for_status()
                    for line in response.iter_lines():
                        if not line:
                            continue
                        chunk = json.loads(line)
                        text = chunk.get("response", "")
                        if text:
                            yield text
                        if chunk.get("done"):
                            return
                return
            except (httpx.ConnectError, httpx.ConnectTimeout) as e:
                last_error = e
                if attempt < self.max_retries:
                    time.sleep(self.retry_delay_seconds)
        raise RuntimeError(
            f"Could not reach Ollama at {self.base_url} after {self.max_retries} attempts"
        ) from last_error
