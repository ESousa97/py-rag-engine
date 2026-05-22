from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Sequence
from typing import Literal, Protocol, TypeAlias, cast

EmbeddingProvider: TypeAlias = Literal["openai", "sentence-transformers"]
SleepFn: TypeAlias = Callable[[float], Awaitable[None]]


class _OpenAIEmbeddingItem(Protocol):
    embedding: Sequence[float]


class _OpenAIEmbeddingResponse(Protocol):
    data: Sequence[_OpenAIEmbeddingItem]


class _OpenAIEmbeddingsClient(Protocol):
    async def create(self, *, model: str, input: list[str]) -> _OpenAIEmbeddingResponse:
        """Create embeddings using the official OpenAI async client."""


class _OpenAIClient(Protocol):
    embeddings: _OpenAIEmbeddingsClient


class VectorClient:
    """Async embedding client backed by OpenAI or a local SentenceTransformer."""

    def __init__(
        self,
        *,
        provider: EmbeddingProvider = "sentence-transformers",
        model: str | None = None,
        batch_size: int = 32,
        max_retries: int = 5,
        initial_backoff: float = 1.0,
        max_backoff: float = 60.0,
        jitter: float = 0.1,
        client: _OpenAIClient | None = None,
        device: str | None = None,
        sleep: SleepFn | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if max_retries < 1:
            raise ValueError("max_retries must be at least 1.")
        if initial_backoff <= 0.0:
            raise ValueError("initial_backoff must be positive.")
        if max_backoff < initial_backoff:
            raise ValueError("max_backoff must be greater than or equal to initial_backoff.")
        if jitter < 0.0:
            raise ValueError("jitter must be non-negative.")

        self.provider = provider
        self.model = model or _default_model(provider)
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.jitter = jitter
        self.device = device
        self._openai_client = client
        self._sentence_transformer_model: object | None = None
        self._sleep = sleep or asyncio.sleep

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per text, retrying OpenAI rate limits."""
        if not texts:
            return []

        embeddings: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            if self.provider == "openai":
                embeddings.extend(await self._get_openai_embeddings(batch))
            else:
                embeddings.extend(await self._get_sentence_transformer_embeddings(batch))
        return embeddings

    async def _get_openai_embeddings(self, texts: list[str]) -> list[list[float]]:
        client = self._get_openai_client()
        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                response = await client.embeddings.create(model=self.model, input=texts)
                return [_as_float_list(item.embedding) for item in response.data]
            except Exception as exc:
                if not _is_rate_limit_error(exc):
                    raise
                last_error = exc
                if attempt == self.max_retries - 1:
                    raise
                await self._sleep(self._backoff_seconds(attempt))

        if last_error is not None:
            raise last_error
        raise RuntimeError("OpenAI embedding request failed without an exception.")

    async def _get_sentence_transformer_embeddings(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self._encode_sentence_transformer, texts)

    def _encode_sentence_transformer(self, texts: list[str]) -> list[list[float]]:
        model = self._get_sentence_transformer_model()
        encode = getattr(model, "encode")
        raw_vectors = encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            batch_size=self.batch_size,
        )
        return [_as_float_list(vector) for vector in raw_vectors]

    def _get_openai_client(self) -> _OpenAIClient:
        if self._openai_client is not None:
            return self._openai_client

        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "openai is required for VectorClient(provider='openai'); "
                "install it with: pip install 'py-rag-engine[embeddings]'"
            ) from exc

        self._openai_client = cast(_OpenAIClient, AsyncOpenAI())
        return self._openai_client

    def _get_sentence_transformer_model(self) -> object:
        if self._sentence_transformer_model is not None:
            return self._sentence_transformer_model

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is required for "
                "VectorClient(provider='sentence-transformers'); install it with: "
                "pip install 'py-rag-engine[embeddings]'"
            ) from exc

        if self.device is None:
            self._sentence_transformer_model = SentenceTransformer(self.model)
        else:
            self._sentence_transformer_model = SentenceTransformer(self.model, device=self.device)
        return self._sentence_transformer_model

    def _backoff_seconds(self, attempt: int) -> float:
        base = min(self.max_backoff, self.initial_backoff * (2**attempt))
        if self.jitter == 0.0:
            return base
        return base + random.uniform(0.0, self.jitter * base)


def _default_model(provider: EmbeddingProvider) -> str:
    if provider == "openai":
        return "text-embedding-3-small"
    return "all-MiniLM-L6-v2"


def _as_float_list(vector: Sequence[float]) -> list[float]:
    return [float(value) for value in vector]


def _is_rate_limit_error(exc: Exception) -> bool:
    if exc.__class__.__name__ == "RateLimitError":
        return True

    status_code = getattr(exc, "status_code", None)
    if status_code == 429:
        return True

    response = getattr(exc, "response", None)
    response_status = getattr(response, "status_code", None)
    return response_status == 429
