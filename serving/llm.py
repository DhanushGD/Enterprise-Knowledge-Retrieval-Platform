"""LLM access behind one small interface, so the answer pipeline never knows which model runs.

    client.stream(messages, max_tokens)  ->  yields str tokens, then one Usage

Today: Ollama (free, local, CPU-friendly with a 3B model). To use a hosted model later, add a
class with the same `stream` signature and register it in `build_llm` - nothing else changes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator, Protocol


class LLMError(RuntimeError):
    """The model call failed (bad model name, server error, malformed stream)."""


class LLMUnavailable(LLMError):
    """The model server can't be reached at all (not started, wrong URL, timeout)."""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0


class LLMClient(Protocol):
    def stream(self, messages: list[dict], max_tokens: int) -> Iterator[str | Usage]: ...


class OllamaClient:
    """Talks to a local Ollama server (https://ollama.com).

    CPU-oriented defaults: temperature 0 (repeatable answers), a small context window (prompt
    reading is the slow part on CPU, and our prompts are ~1k tokens), and keep_alive so the
    model stays in RAM between questions instead of reloading for 5-15 s each time.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2:3b",
        num_ctx: int = 2048,
        temperature: float = 0.0,
        keep_alive: str = "30m",
        connect_timeout: float = 5.0,
        read_timeout: float = 300.0,       # CPU generation is slow; this is per gap between tokens
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.keep_alive = keep_alive
        self.timeout = (connect_timeout, read_timeout)

    def _post(self, payload: dict):
        import requests   # lazy, like the other heavy/optional imports

        try:
            response = requests.post(
                f"{self.base_url}/api/chat", json=payload, stream=True, timeout=self.timeout
            )
        except requests.exceptions.ConnectionError as exc:
            raise LLMUnavailable(
                f"Cannot reach Ollama at {self.base_url}. Is it running? (start the Ollama app "
                f"or run `ollama serve`)"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise LLMUnavailable(f"Ollama at {self.base_url} timed out") from exc

        if response.status_code == 404:
            raise LLMError(f"Model '{self.model}' is not installed. Run: ollama pull {self.model}")
        if response.status_code != 200:
            raise LLMError(f"Ollama returned HTTP {response.status_code}: {response.text[:300]}")
        return response

    def stream(self, messages: list[dict], max_tokens: int = 256) -> Iterator[str | Usage]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": max_tokens,
                "seed": 0,
            },
        }
        response = self._post(payload)
        finished = False
        try:
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except ValueError as exc:
                    raise LLMError(f"Malformed response from Ollama: {line[:120]!r}") from exc
                if "error" in event:
                    raise LLMError(f"Ollama error: {event['error']}")
                token = (event.get("message") or {}).get("content", "")
                if token:
                    yield token
                if event.get("done"):
                    finished = True
                    yield Usage(event.get("prompt_eval_count", 0), event.get("eval_count", 0))
                    break
        except Exception as exc:
            if isinstance(exc, LLMError):
                raise
            raise LLMUnavailable(f"Connection to Ollama dropped: {type(exc).__name__}: {exc}") from exc
        finally:
            response.close()
        if not finished:
            raise LLMError("Ollama stream ended before the answer was complete")

    def warm_up(self) -> None:
        """Load the model into RAM now (so the first real question isn't 10 s slower)."""
        for _ in self.stream([{"role": "user", "content": "ok"}], max_tokens=1):
            pass


def build_llm(num_ctx: int = 2048) -> LLMClient:
    from core.settings import settings

    if settings.llm_provider == "ollama":
        return OllamaClient(settings.ollama_base_url, settings.llm_model, num_ctx=num_ctx)
    raise ValueError(
        f"Unknown LLM_PROVIDER '{settings.llm_provider}'. Supported: ollama. "
        f"(Add a client with the same .stream() signature in serving/llm.py to support more.)"
    )
