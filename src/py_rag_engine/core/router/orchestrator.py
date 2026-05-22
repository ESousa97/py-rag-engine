"""Top-level QueryRouter — the single entry point for the ingestion layer.

Wires together:
  * an `IntentClassifier` (LLM or heuristic) → `RoutingDecision`,
  * optionally a `SubQueryGenerator` → `SubQueryPlan` (1..3 sub-queries),
  * async parallel retrieval + multi-query RRF fusion.

Typical wiring (LM Studio + Postgres + hybrid retrieval):

    from py_rag_engine.clients import LMStudioClient
    from py_rag_engine.core.router import LLMIntentClassifier, LLMSubQueryGenerator, QueryRouter
    from py_rag_engine.retrieval import retrieve_hybrid

    client   = LMStudioClient()
    embed    = lambda texts: client.embed(texts)
    retrieve = lambda q, vec: retrieve_hybrid(vec, q, store, top_k=20)

    router = QueryRouter(
        classifier=LLMIntentClassifier(chat=client.chat),
        decomposer=LLMSubQueryGenerator(chat=client.chat),
        embed=embed,
        retrieve=retrieve,
    )
    result = await router.route("...")            # async
    result = router.route_sync("...")             # blocking convenience
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Protocol

from py_rag_engine.core.router.fusion import (
    DEFAULT_MULTI_QUERY_RRF_K,
    DEFAULT_MULTI_QUERY_TOP_K,
    multi_query_rrf,
)
from py_rag_engine.core.router.parallel import (
    EmbedBatchFn,
    RetrieveFn,
    embed_sub_queries,
    retrieve_in_parallel,
)
from py_rag_engine.core.router.schemas import (
    RoutedQueryResult,
    RoutingDecision,
    SubQuery,
    SubQueryPlan,
)


class IntentClassifier(Protocol):
    """Anything with a `classify(question) -> RoutingDecision`."""

    def classify(self, question: str) -> RoutingDecision: ...


class SubQueryGenerator(Protocol):
    """Anything with a `generate(question) -> SubQueryPlan`."""

    def generate(self, question: str) -> SubQueryPlan: ...


def _plan_from_single_query(question: str) -> SubQueryPlan:
    """Wrap a single question in a one-shot plan so fusion code stays uniform."""
    return SubQueryPlan(
        original_query=question,
        sub_queries=[SubQuery(text=question, reasoning="atomic question, no decomposition")],
    )


@dataclass(slots=True)
class QueryRouter:
    """Async-first query-ingestion orchestrator.

    The router is *transport-agnostic*: it only depends on `embed` and
    `retrieve` callables. Use sync functions (they'll be off-loaded to a
    thread) or native coroutines — either works.
    """

    classifier: IntentClassifier
    decomposer: SubQueryGenerator
    embed: EmbedBatchFn
    retrieve: RetrieveFn
    rrf_k: int = DEFAULT_MULTI_QUERY_RRF_K
    top_k: int = DEFAULT_MULTI_QUERY_TOP_K

    async def route(self, question: str) -> RoutedQueryResult:
        """Classify → optionally decompose → embed → retrieve in parallel → fuse."""
        cleaned = (question or "").strip()
        if not cleaned:
            raise ValueError("question must not be empty")

        decision = self.classifier.classify(cleaned)

        if decision.needs_decomposition:
            plan = self.decomposer.generate(cleaned)
        else:
            plan = _plan_from_single_query(cleaned)

        embeddings = await embed_sub_queries(plan, self.embed)
        ranked_lists = await retrieve_in_parallel(plan, embeddings, self.retrieve)
        fused = multi_query_rrf(ranked_lists, rrf_k=self.rrf_k, top_k=self.top_k)

        return RoutedQueryResult(
            decision=decision,
            plan=plan if decision.needs_decomposition else None,
            results=fused,
        )

    def route_sync(self, question: str) -> RoutedQueryResult:
        """Blocking convenience wrapper for callers outside an event loop."""
        return asyncio.run(self.route(question))


__all__ = [
    "IntentClassifier",
    "QueryRouter",
    "SubQueryGenerator",
]
