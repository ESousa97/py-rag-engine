"""HTTP client for an LM Studio / OpenAI-compatible server.

Exposes one thin class — `LMStudioClient` — used by:
  - `py_rag_engine.embeddings.lm_studio_embedder`
  - `py_rag_engine.generation.lm_studio_chat`
  - `py_rag_engine.evaluation.*`
  - the demo and evaluation CLIs

All retry, timeout, and connection-handling logic lives here. Tests can swap
the underlying request function via the `_http_json` constructor argument.
"""
from __future__ import annotations

import http.client
import json
import time
from collections.abc import Callable, Iterable
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from py_rag_engine.config import LMStudioConfig

HttpJsonFn = Callable[..., Any]


class LMStudioClient:
    """Minimal LM Studio (OpenAI-compatible) HTTP client with retry/backoff.

    Retries on `OSError` (covers `WinError 10054` socket resets) and
    `json.JSONDecodeError`. Uses `Connection: close` to avoid keep-alive
    bugs that surface when chat completions exceed timeouts.
    """

    def __init__(
        self,
        config: LMStudioConfig | None = None,
        *,
        http_json: HttpJsonFn | None = None,
    ) -> None:
        self.config = config or LMStudioConfig.from_env()
        self._http_json = http_json or self._default_http_json

    # ── Endpoint methods ─────────────────────────────────────────────────────

    def models(self) -> list[dict[str, Any]]:
        """Return the list of models known to the server."""
        body = self._http_json(f"{self.config.base_url}/v1/models", None, timeout=10)
        return list(body.get("data", []))

    def embed(self, texts: Iterable[str], *, model: str | None = None) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per input string."""
        model_id = model or self.config.embed_model
        texts_list = list(texts)
        out: list[list[float]] = []
        for start in range(0, len(texts_list), self.config.embed_batch_size):
            batch = texts_list[start : start + self.config.embed_batch_size]
            body = self._http_json(
                f"{self.config.base_url}/v1/embeddings",
                {"model": model_id, "input": batch},
            )
            out.extend(list(map(float, item["embedding"])) for item in body["data"])
        return out

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        model: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 400,
        timeout: int | None = None,
    ) -> str:
        """Run chat completions and return the assistant message content."""
        model_id = model or self.config.chat_model
        if not model_id:
            raise ValueError("chat_model not configured; set LM_STUDIO_CHAT_MODEL or pass model=")
        body = self._http_json(
            f"{self.config.base_url}/v1/chat/completions",
            {
                "model": model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=timeout or self.config.request_timeout,
        )
        return body["choices"][0]["message"]["content"].strip()

    def warm_up_chat(self, model: str | None = None) -> bool:
        """Tiny chat call so LM Studio loads the model into VRAM up front.

        Returns True when the server answered. The first request after a fresh
        load can take 20-60 seconds on a 7B GGUF — warming up here avoids
        per-question timeouts later.
        """
        try:
            self.chat(
                [{"role": "user", "content": "Reply with the single word: OK"}],
                model=model,
                temperature=0.0,
                max_tokens=5,
                timeout=120,
            )
            return True
        except Exception:
            return False

    # ── HTTP plumbing ────────────────────────────────────────────────────────

    def _default_http_json(
        self,
        url: str,
        payload: dict | None = None,
        *,
        timeout: int | None = None,
    ) -> Any:
        timeout = timeout or self.config.request_timeout
        retries = self.config.retries
        backoff = self.config.backoff
        last_exc: Exception | None = None
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError(
                f"Unsupported URL scheme {parsed.scheme!r} in {url!r}; only http/https allowed"
            )
        # Plain http goes through http.client: it avoids the `urllib.request`
        # + OpenSSL Applink conflict that aborts the process on Windows +
        # Python 3.14 when psycopg's libpq is also loaded.
        use_httpclient = parsed.scheme == "http"
        for attempt in range(1, retries + 1):
            try:
                if use_httpclient:
                    conn = http.client.HTTPConnection(
                        parsed.hostname, parsed.port or 80, timeout=timeout,
                    )
                    try:
                        if payload is None:
                            conn.request("GET", parsed.path or "/")
                        else:
                            data = json.dumps(payload).encode("utf-8")
                            conn.request(
                                "POST", parsed.path or "/", body=data,
                                headers={
                                    "Content-Type": "application/json",
                                    "Connection": "close",
                                },
                            )
                        resp = conn.getresponse()
                        raw = resp.read().decode("utf-8")
                        if resp.status >= 500:
                            raise OSError(f"HTTP {resp.status} from {url}: {raw[:200]}")
                        if resp.status >= 400:
                            raise ValueError(f"HTTP {resp.status} from {url}: {raw[:200]}")
                        return json.loads(raw)
                    finally:
                        conn.close()
                if payload is None:
                    req = Request(url, method="GET")
                else:
                    data = json.dumps(payload).encode("utf-8")
                    req = Request(
                        url,
                        data=data,
                        headers={"Content-Type": "application/json", "Connection": "close"},
                        method="POST",
                    )
                try:
                    # Scheme restricted to http/https above.  # nosec B310
                    with urlopen(req, timeout=timeout) as resp:
                        return json.loads(resp.read().decode("utf-8"))
                except HTTPError as exc:
                    if exc.code < 500:  # client error — retrying will not help
                        detail = exc.read().decode("utf-8", "replace")[:200]
                        raise ValueError(f"HTTP {exc.code} from {url}: {detail}") from exc
                    raise
            except (OSError, json.JSONDecodeError) as exc:
                last_exc = exc
                if attempt < retries:
                    wait = backoff ** attempt
                    print(
                        f"        [retry {attempt}/{retries-1}] "
                        f"{type(exc).__name__}: {exc} — sleeping {wait:.1f}s"
                    )
                    time.sleep(wait)
                    continue
                raise
        if last_exc is not None:                # pragma: no cover
            raise last_exc


# ── Chat-model auto-detection ────────────────────────────────────────────────


def detect_chat_model(
    client: LMStudioClient,
    *,
    preferred_patterns: tuple[str, ...] = (
        "qwen2.5-7b",
        "qwen-2-5-7b",
        "llama-3.1-8b",
        "llama-3.2-8b",
        "mistral-7b",
        "gemma-2-9b",
        "phi-3.5-mini",
        "phi-3-mini",
    ),
) -> str | None:
    """Pick the best chat model loaded in LM Studio.

    Preference order:
      1. Substring match against `preferred_patterns` (7-8B sweet spot).
      2. Non-embedding model with 6-9B parameters.
      3. Smallest model whose name advertises >= 4B parameters.
      4. First non-embedding model.
      5. First model of any kind.
    """
    import re

    try:
        models = client.models()
    except Exception:
        return None

    all_ids = [m.get("id", "") for m in models if m.get("id")]
    if not all_ids:
        return None

    embedding_hints = {"bge", "embed", "e5-", "ada", "minilm", "clip", "nomic"}
    chat_ids = [mid for mid in all_ids
                if not any(h in mid.lower() for h in embedding_hints)]
    if not chat_ids:
        return all_ids[0]

    # 1. Preferred substring match — shortest match wins (usually the vanilla variant)
    for pattern in preferred_patterns:
        matches = [mid for mid in chat_ids if pattern in mid.lower()]
        if matches:
            matches.sort(key=len)
            return matches[0]

    def size_b(mid: str) -> int:
        match = re.search(r"(\d+)\s*b", mid.lower())
        return int(match.group(1)) if match else 0

    # 2. 6-9B band, smallest first
    in_band = [m for m in chat_ids if 6 <= size_b(m) <= 9]
    if in_band:
        in_band.sort(key=size_b)
        return in_band[0]

    # 3. Smallest >= 4B
    with_size = [m for m in chat_ids if size_b(m) >= 4]
    if with_size:
        with_size.sort(key=size_b)
        return with_size[0]

    # 4-5. First non-embedding or first overall
    return chat_ids[0]
