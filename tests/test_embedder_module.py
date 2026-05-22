from __future__ import annotations

import asyncio
from collections.abc import Sequence

from py_rag_engine.embedder import VectorClient


class _RateLimitError(Exception):
    status_code = 429


class _FakeEmbeddingItem:
    def __init__(self, embedding: Sequence[float]) -> None:
        self.embedding = embedding


class _FakeEmbeddingResponse:
    def __init__(self, data: Sequence[_FakeEmbeddingItem]) -> None:
        self.data = data


class _FakeEmbeddingsClient:
    def __init__(self) -> None:
        self.calls = 0

    async def create(self, *, model: str, input: list[str]) -> _FakeEmbeddingResponse:
        self.calls += 1
        if self.calls == 1:
            raise _RateLimitError("rate limited")
        return _FakeEmbeddingResponse(
            [_FakeEmbeddingItem([float(len(text)), 1.0]) for text in input]
        )


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.embeddings = _FakeEmbeddingsClient()


def test_vector_client_retries_openai_rate_limits_with_backoff() -> None:
    slept: list[float] = []

    async def sleep(seconds: float) -> None:
        slept.append(seconds)

    fake_client = _FakeOpenAIClient()
    vector_client = VectorClient(
        provider="openai",
        model="test-embedding-model",
        client=fake_client,
        sleep=sleep,
        max_retries=2,
        initial_backoff=0.5,
        jitter=0.0,
    )

    embeddings = asyncio.run(vector_client.get_embeddings(["a", "abcd"]))

    assert embeddings == [[1.0, 1.0], [4.0, 1.0]]
    assert fake_client.embeddings.calls == 2
    assert slept == [0.5]


def test_vector_client_returns_empty_list_without_provider_call() -> None:
    fake_client = _FakeOpenAIClient()
    vector_client = VectorClient(provider="openai", client=fake_client)

    embeddings = asyncio.run(vector_client.get_embeddings([]))

    assert embeddings == []
    assert fake_client.embeddings.calls == 0
