"""Async parallel retrieval for sub-queries.

The router hands this module a `SubQueryPlan`. We:
  1. Embed every sub-query in parallel (single batched call when possible —
     LM Studio handles batching server-side and is cheaper than N round-trips).
  2. Run the configured retrieval function for each sub-query concurrently
     using `asyncio.gather`. Sync embed/retrieval functions are off-loaded
     to a thread pool via `asyncio.to_thread` so they don't block the loop.
  3. Pass the N ranked lists to `multi_query_rrf` for fusion.

The retrieval function signature is intentionally narrow:

    retrieve_fn(query_text: str, query_embedding: list[float]) -> Sequence[_RankedItem]

so callers can wire in plain dense search, hybrid (dense + FTS), or hybrid +
rerank — the router does not care which.
"""
from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from py_rag_engine.core.router.fusion import (
    DEFAULT_MULTI_QUERY_RRF_K,
    DEFAULT_MULTI_QUERY_TOP_K,
    multi_query_rrf,
)
from py_rag_engine.core.router.schemas import FusedRetrievalResult, SubQueryPlan

# Sync or async embed function: takes a batch of texts, returns one vector per text.
EmbedBatchFn = Callable[[list[str]], list[list[float]] | Awaitable[list[list[float]]]]

# Sync or async retrieval function for a single sub-query.
RetrieveFn = Callable[[str, list[float]], Sequence[Any] | Awaitable[Sequence[Any]]]


async def _maybe_await(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


async def _run_callable(func: Callable[..., Any], *args: Any) -> Any:
    """Run `func(*args)` — directly if it's a coroutine, in a thread otherwise.

    This lets us accept the existing *sync* embed/retrieve functions from
    `py_rag_engine.embeddings` and `py_rag_engine.retrieval` unchanged.
    """
    if inspect.iscoroutinefunction(func):
        return await func(*args)
    result = await asyncio.to_thread(func, *args)
    return await _maybe_await(result)


async def embed_sub_queries(plan: SubQueryPlan, embed: EmbedBatchFn) -> list[list[float]]:
    """Embed all sub-queries. One batched call when `embed` is sync."""
    texts = plan.texts
    if not texts:
        return []
    result = await _run_callable(embed, texts)
    if len(result) != len(texts):
        raise ValueError(
            f"embedder returned {len(result)} vectors for {len(texts)} sub-queries"
        )
    return list(result)


async def retrieve_in_parallel(
    plan: SubQueryPlan,
    embeddings: Sequence[Sequence[float]],
    retrieve: RetrieveFn,
) -> list[Sequence[Any]]:
    """Fan out `retrieve(text, embedding)` calls with `asyncio.gather`.

    Exceptions are NOT swallowed individually — if one sub-query fails the
    whole call raises. Treating partial failure as success would silently
    degrade fusion quality (a missing list shifts every score).
    """
    if len(embeddings) != len(plan.sub_queries):
        raise ValueError(
            f"got {len(embeddings)} embeddings for {len(plan.sub_queries)} sub-queries"
        )

    tasks = [
        _run_callable(retrieve, sq.text, list(emb))
        for sq, emb in zip(plan.sub_queries, embeddings, strict=True)
    ]
    return list(await asyncio.gather(*tasks))


async def run_parallel_retrieval(
    plan: SubQueryPlan,
    *,
    embed: EmbedBatchFn,
    retrieve: RetrieveFn,
    rrf_k: int = DEFAULT_MULTI_QUERY_RRF_K,
    top_k: int = DEFAULT_MULTI_QUERY_TOP_K,
) -> list[FusedRetrievalResult]:
    """End-to-end: embed sub-queries → retrieve in parallel → fuse with RRF."""
    embeddings = await embed_sub_queries(plan, embed)
    ranked_lists = await retrieve_in_parallel(plan, embeddings, retrieve)
    return multi_query_rrf(ranked_lists, rrf_k=rrf_k, top_k=top_k)


__all__ = [
    "EmbedBatchFn",
    "RetrieveFn",
    "embed_sub_queries",
    "retrieve_in_parallel",
    "run_parallel_retrieval",
]
