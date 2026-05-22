"""Tests for `py_rag_engine.core.router`.

We test the pure pieces (schemas, RRF math, heuristic fallbacks) directly,
and exercise the LLM-backed pieces with a stubbed `ChatFn` so no network is
involved.
"""
from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import ValidationError

from py_rag_engine.core.router import (
    MAX_SUB_QUERIES,
    FusedRetrievalResult,
    LLMIntentClassifier,
    LLMSubQueryGenerator,
    QueryIntent,
    QueryRouter,
    RoutingDecision,
    SubQuery,
    SubQueryPlan,
    call_structured,
    embed_sub_queries,
    heuristic_classify,
    heuristic_decompose,
    multi_query_rrf,
    retrieve_in_parallel,
    run_parallel_retrieval,
)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------

@dataclass
class FakeChunk:
    """Structural stand-in for retriever result types."""

    id: int
    text: str = ""
    metadata: dict[str, Any] | None = None
    content_hash: str | None = None
    embedding_model: str = "test-embed"

    def __post_init__(self) -> None:
        if not self.text:
            self.text = f"chunk-{self.id}"
        if self.metadata is None:
            self.metadata = {"source": "test"}


def scripted_chat(responses: list[str]):
    """Return a `ChatFn` that pops scripted replies in order."""
    queue = list(responses)
    calls: list[dict[str, Any]] = []

    def _chat(messages, *, temperature: float = 0.0, max_tokens: int = 600) -> str:
        calls.append(
            {"messages": list(messages), "temperature": temperature, "max_tokens": max_tokens}
        )
        if not queue:
            raise AssertionError("scripted_chat exhausted; no more replies queued")
        return queue.pop(0)

    _chat.calls = calls  # type: ignore[attr-defined]
    return _chat


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class TestSchemas:
    def test_query_intent_needs_decomposition_flag(self):
        assert QueryIntent.COMPLEX_MULTI_HOP.needs_decomposition is True
        assert QueryIntent.COMPARATIVE.needs_decomposition is True
        assert QueryIntent.SIMPLE_FACTUAL.needs_decomposition is False
        assert QueryIntent.CONVERSATIONAL.needs_decomposition is False
        assert QueryIntent.OUT_OF_SCOPE.needs_decomposition is False

    def test_routing_decision_backfills_needs_decomposition_for_complex_intents(self):
        decision = RoutingDecision(
            intent=QueryIntent.COMPLEX_MULTI_HOP,
            complexity_score=0.8,
            reasoning="multi-part",
        )
        # backfilled by model_validator
        assert decision.needs_decomposition is True

    def test_routing_decision_validates_complexity_range(self):
        with pytest.raises(ValidationError):
            RoutingDecision(intent=QueryIntent.SIMPLE_FACTUAL, complexity_score=1.5)
        with pytest.raises(ValidationError):
            RoutingDecision(intent=QueryIntent.SIMPLE_FACTUAL, complexity_score=-0.1)

    def test_sub_query_strips_and_rejects_blank(self):
        sq = SubQuery(text="  what is X?  ")
        assert sq.text == "what is X?"
        with pytest.raises(ValidationError):
            SubQuery(text="   ")

    def test_sub_query_plan_caps_at_max_and_dedupes(self):
        plan = SubQueryPlan(
            original_query="big complex question",
            sub_queries=[
                SubQuery(text="What is A?"),
                SubQuery(text="what is a?"),  # case-only dup
                SubQuery(text="What is B?"),
                SubQuery(text="What is C?"),
                SubQuery(text="What is D?"),  # beyond cap, dropped
            ],
        )
        assert len(plan.sub_queries) == MAX_SUB_QUERIES
        assert plan.texts == ["What is A?", "What is B?", "What is C?"]

    def test_sub_query_plan_requires_at_least_one_after_dedup(self):
        # All identical → after dedup we keep just one, which still satisfies min.
        plan = SubQueryPlan(
            original_query="q",
            sub_queries=[SubQuery(text="same"), SubQuery(text="same")],
        )
        assert len(plan.sub_queries) == 1


# ---------------------------------------------------------------------------
# Heuristic classifier
# ---------------------------------------------------------------------------

class TestHeuristicClassify:
    def test_empty_input_is_conversational(self):
        assert heuristic_classify("").intent is QueryIntent.CONVERSATIONAL

    def test_greeting_is_conversational(self):
        assert heuristic_classify("hello there").intent is QueryIntent.CONVERSATIONAL

    def test_simple_question_is_simple_factual(self):
        decision = heuristic_classify("What is pgvector?")
        assert decision.intent is QueryIntent.SIMPLE_FACTUAL
        assert decision.needs_decomposition is False

    def test_comparative_question_triggers_decomposition(self):
        decision = heuristic_classify("Compare pgvector vs FAISS for ANN search.")
        assert decision.intent is QueryIntent.COMPARATIVE
        assert decision.needs_decomposition is True

    def test_multi_clause_triggers_complex(self):
        decision = heuristic_classify(
            "What is RRF and how does it differ from cosine similarity, "
            "and when should I use it?"
        )
        assert decision.intent is QueryIntent.COMPLEX_MULTI_HOP
        assert decision.needs_decomposition is True

    def test_multiple_question_marks_triggers_complex(self):
        decision = heuristic_classify("What is X? How does Y relate?")
        assert decision.intent is QueryIntent.COMPLEX_MULTI_HOP


# ---------------------------------------------------------------------------
# LLM classifier (with stubbed ChatFn)
# ---------------------------------------------------------------------------

class TestLLMIntentClassifier:
    def test_parses_valid_json_response(self):
        reply = json.dumps(
            {
                "intent": "simple_factual",
                "complexity_score": 0.2,
                "reasoning": "single fact",
                "needs_decomposition": False,
            }
        )
        clf = LLMIntentClassifier(chat=scripted_chat([reply]))
        decision = clf.classify("What is pgvector?")
        assert decision.intent is QueryIntent.SIMPLE_FACTUAL
        assert decision.needs_decomposition is False

    def test_strips_markdown_fence(self):
        reply = (
            "Sure!\n```json\n"
            '{"intent": "comparative", "complexity_score": 0.7, '
            '"reasoning": "x", "needs_decomposition": true}\n'
            "```\nDone."
        )
        clf = LLMIntentClassifier(chat=scripted_chat([reply]))
        decision = clf.classify("pgvector vs faiss")
        assert decision.intent is QueryIntent.COMPARATIVE

    def test_falls_back_to_heuristic_on_total_garbage(self):
        # Bad JSON → repair attempt also bad → fallback to heuristic.
        clf = LLMIntentClassifier(
            chat=scripted_chat(["not json at all", "still not json"]),
            fallback_on_error=True,
        )
        decision = clf.classify("Compare A vs B.")
        # heuristic should classify as comparative
        assert decision.intent is QueryIntent.COMPARATIVE

    def test_raises_when_fallback_disabled(self):
        clf = LLMIntentClassifier(
            chat=scripted_chat(["nope", "still nope"]),
            fallback_on_error=False,
        )
        with pytest.raises(ValueError):
            clf.classify("anything")


# ---------------------------------------------------------------------------
# Decomposer
# ---------------------------------------------------------------------------

class TestHeuristicDecompose:
    def test_returns_at_least_one(self):
        plan = heuristic_decompose("trivial question")
        assert len(plan.sub_queries) >= 1
        assert plan.original_query == "trivial question"

    def test_splits_on_and(self):
        plan = heuristic_decompose("What is A and what is B and what is C?")
        assert 1 <= len(plan.sub_queries) <= MAX_SUB_QUERIES

    def test_respects_max_cap(self):
        plan = heuristic_decompose(
            "What is A? Also what is B; and what is C and what is D and what is E?"
        )
        assert len(plan.sub_queries) <= MAX_SUB_QUERIES


class TestLLMSubQueryGenerator:
    def test_parses_structured_plan(self):
        reply = json.dumps(
            {
                "original_query": "How does X compare to Y under condition Z?",
                "sub_queries": [
                    {"text": "What is X?", "reasoning": "facet 1"},
                    {"text": "What is Y?", "reasoning": "facet 2"},
                    {"text": "What is condition Z?", "reasoning": "facet 3"},
                ],
            }
        )
        gen = LLMSubQueryGenerator(chat=scripted_chat([reply]))
        plan = gen.generate("How does X compare to Y under condition Z?")
        assert plan.texts == ["What is X?", "What is Y?", "What is condition Z?"]

    def test_caps_at_constructor_limit_even_when_llm_returns_more(self):
        reply = json.dumps(
            {
                "original_query": "q",
                "sub_queries": [
                    {"text": "A?"},
                    {"text": "B?"},
                    {"text": "C?"},
                ],
            }
        )
        gen = LLMSubQueryGenerator(chat=scripted_chat([reply]), max_sub_queries=2)
        plan = gen.generate("q")
        assert len(plan.sub_queries) == 2

    def test_constructor_rejects_over_global_cap(self):
        with pytest.raises(ValueError):
            LLMSubQueryGenerator(chat=scripted_chat([]), max_sub_queries=MAX_SUB_QUERIES + 1)

    def test_falls_back_to_heuristic_on_bad_json(self):
        gen = LLMSubQueryGenerator(
            chat=scripted_chat(["???", "still bad"]),
            fallback_on_error=True,
        )
        plan = gen.generate("What is A and what is B?")
        assert plan.original_query == "What is A and what is B?"
        assert len(plan.sub_queries) >= 1


# ---------------------------------------------------------------------------
# multi_query_rrf
# ---------------------------------------------------------------------------

class TestMultiQueryRRF:
    def test_single_list_preserves_order(self):
        ranked = [FakeChunk(i) for i in [1, 2, 3]]
        out = multi_query_rrf([ranked], top_k=3)
        assert [r.id for r in out] == [1, 2, 3]
        assert all(r.contributing_query_indices == (0,) for r in out)

    def test_overlap_boosts_shared_documents(self):
        # doc-5 is rank-3 in list 1 and rank-1 in list 2 → should beat doc-1.
        list_a = [FakeChunk(1), FakeChunk(2), FakeChunk(5)]
        list_b = [FakeChunk(5), FakeChunk(4)]
        out = multi_query_rrf([list_a, list_b], rrf_k=1, top_k=4)
        # doc-5: 1/(1+3) + 1/(1+1) = 0.25 + 0.5 = 0.75
        # doc-1: 1/(1+1) = 0.5
        assert out[0].id == 5
        assert out[0].contributing_query_indices == (0, 1)

    def test_three_lists_fuse(self):
        # Mimics 3 sub-queries each returning slightly different orderings.
        lists = [
            [FakeChunk(1), FakeChunk(2), FakeChunk(3)],
            [FakeChunk(2), FakeChunk(3), FakeChunk(4)],
            [FakeChunk(3), FakeChunk(1), FakeChunk(5)],
        ]
        out = multi_query_rrf(lists, rrf_k=10, top_k=5)
        # doc-3 appears in all three lists → should rank first.
        assert out[0].id == 3
        assert sorted(out[0].contributing_query_indices) == [0, 1, 2]

    def test_empty_input_returns_empty(self):
        assert multi_query_rrf([], top_k=5) == []
        assert multi_query_rrf([[], [], []], top_k=5) == []

    def test_validates_rrf_k(self):
        with pytest.raises(ValueError, match="rrf_k must be greater than zero"):
            multi_query_rrf([[FakeChunk(1)]], rrf_k=0)

    def test_validates_top_k(self):
        with pytest.raises(ValueError, match="top_k must be greater than zero"):
            multi_query_rrf([[FakeChunk(1)]], top_k=0)

    def test_result_is_fused_retrieval_result(self):
        out = multi_query_rrf([[FakeChunk(1)]], top_k=1)
        assert isinstance(out[0], FusedRetrievalResult)
        assert out[0].rrf_score > 0


# ---------------------------------------------------------------------------
# Parallel retrieval
# ---------------------------------------------------------------------------

class TestParallelRetrieval:
    def test_embed_sub_queries_sync_function(self):
        plan = SubQueryPlan(
            original_query="q",
            sub_queries=[SubQuery(text="A?"), SubQuery(text="B?")],
        )

        def embed(texts: list[str]) -> list[list[float]]:
            return [[float(len(t))] for t in texts]

        result = asyncio.run(embed_sub_queries(plan, embed))
        assert result == [[2.0], [2.0]]

    def test_embed_sub_queries_async_function(self):
        plan = SubQueryPlan(original_query="q", sub_queries=[SubQuery(text="A?")])

        async def embed(texts: list[str]) -> list[list[float]]:
            await asyncio.sleep(0)
            return [[1.0, 2.0, 3.0] for _ in texts]

        result = asyncio.run(embed_sub_queries(plan, embed))
        assert result == [[1.0, 2.0, 3.0]]

    def test_embed_dimension_mismatch_raises(self):
        plan = SubQueryPlan(
            original_query="q",
            sub_queries=[SubQuery(text="A?"), SubQuery(text="B?")],
        )

        def embed(texts: list[str]) -> list[list[float]]:
            return [[0.0]]  # only one vector for two queries

        with pytest.raises(ValueError, match="returned 1 vectors for 2"):
            asyncio.run(embed_sub_queries(plan, embed))

    def test_retrieve_in_parallel_runs_each_sub_query(self):
        plan = SubQueryPlan(
            original_query="q",
            sub_queries=[SubQuery(text="A?"), SubQuery(text="B?"), SubQuery(text="C?")],
        )
        embeddings = [[0.1], [0.2], [0.3]]
        seen: list[tuple[str, list[float]]] = []

        def retrieve(text: str, emb: list[float]):
            seen.append((text, emb))
            return [FakeChunk(hash(text) % 100)]

        result = asyncio.run(retrieve_in_parallel(plan, embeddings, retrieve))
        assert len(result) == 3
        assert {t for t, _ in seen} == {"A?", "B?", "C?"}

    def test_retrieve_in_parallel_propagates_errors(self):
        plan = SubQueryPlan(original_query="q", sub_queries=[SubQuery(text="A?")])

        def retrieve(text: str, emb: list[float]):
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            asyncio.run(retrieve_in_parallel(plan, [[0.0]], retrieve))

    def test_run_parallel_retrieval_end_to_end(self):
        plan = SubQueryPlan(
            original_query="q",
            sub_queries=[SubQuery(text="A?"), SubQuery(text="B?")],
        )

        def embed(texts: list[str]) -> list[list[float]]:
            return [[float(i)] for i, _ in enumerate(texts)]

        def retrieve(text: str, emb: list[float]) -> Sequence[FakeChunk]:
            # Both sub-queries surface doc-7 as their top hit → fusion should
            # rank it first with contributions from indices 0 and 1.
            return [FakeChunk(7), FakeChunk(8)]

        out = asyncio.run(run_parallel_retrieval(plan, embed=embed, retrieve=retrieve))
        assert out[0].id == 7
        assert sorted(out[0].contributing_query_indices) == [0, 1]


# ---------------------------------------------------------------------------
# call_structured
# ---------------------------------------------------------------------------

class TestCallStructured:
    def test_round_trip_with_valid_json(self):
        reply = json.dumps(
            {"intent": "simple_factual", "complexity_score": 0.1, "reasoning": "ok"}
        )
        chat = scripted_chat([reply])
        out = call_structured(
            chat,
            system="sys",
            user="usr",
            response_model=RoutingDecision,
        )
        assert isinstance(out, RoutingDecision)

    def test_repair_attempt_recovers(self):
        good = json.dumps(
            {"intent": "simple_factual", "complexity_score": 0.0, "reasoning": ""}
        )
        chat = scripted_chat(["garbage", good])
        out = call_structured(
            chat,
            system="sys",
            user="usr",
            response_model=RoutingDecision,
            max_repair_attempts=1,
        )
        assert out.intent is QueryIntent.SIMPLE_FACTUAL
        # The repair turn appended the failure context — 4 messages on attempt 2.
        last_call_messages = chat.calls[-1]["messages"]  # type: ignore[attr-defined]
        assert any("could not be parsed" in m["content"] for m in last_call_messages)


# ---------------------------------------------------------------------------
# QueryRouter
# ---------------------------------------------------------------------------

class _StubClassifier:
    def __init__(self, decision: RoutingDecision) -> None:
        self._decision = decision
        self.called_with: list[str] = []

    def classify(self, question: str) -> RoutingDecision:
        self.called_with.append(question)
        return self._decision


class _StubDecomposer:
    def __init__(self, plan: SubQueryPlan) -> None:
        self._plan = plan
        self.called_with: list[str] = []

    def generate(self, question: str) -> SubQueryPlan:
        self.called_with.append(question)
        return self._plan


class TestQueryRouter:
    def _build_router(self, classifier, decomposer, *, dense_hits):
        def embed(texts):
            return [[float(i)] for i, _ in enumerate(texts)]

        def retrieve(text, emb):
            return dense_hits

        return QueryRouter(
            classifier=classifier,
            decomposer=decomposer,
            embed=embed,
            retrieve=retrieve,
            top_k=3,
        )

    def test_simple_intent_skips_decomposition(self):
        decision = RoutingDecision(
            intent=QueryIntent.SIMPLE_FACTUAL, complexity_score=0.1
        )
        decomposer = _StubDecomposer(
            SubQueryPlan(original_query="x", sub_queries=[SubQuery(text="x?")])
        )
        router = self._build_router(
            classifier=_StubClassifier(decision),
            decomposer=decomposer,
            dense_hits=[FakeChunk(1), FakeChunk(2)],
        )
        result = router.route_sync("What is pgvector?")

        assert decomposer.called_with == []  # decomposer never invoked
        assert result.plan is None
        assert result.was_decomposed is False
        assert [r.id for r in result.results] == [1, 2]

    def test_complex_intent_runs_decomposition_and_fusion(self):
        decision = RoutingDecision(
            intent=QueryIntent.COMPLEX_MULTI_HOP, complexity_score=0.9
        )
        plan = SubQueryPlan(
            original_query="big question",
            sub_queries=[
                SubQuery(text="part A?"),
                SubQuery(text="part B?"),
                SubQuery(text="part C?"),
            ],
        )
        decomposer = _StubDecomposer(plan)
        router = self._build_router(
            classifier=_StubClassifier(decision),
            decomposer=decomposer,
            dense_hits=[FakeChunk(42), FakeChunk(7)],
        )
        result = router.route_sync("Some big multi-part question that needs splitting.")

        assert decomposer.called_with == [
            "Some big multi-part question that needs splitting."
        ]
        assert result.was_decomposed is True
        assert result.plan is not None
        assert len(result.plan.sub_queries) == 3
        # All 3 sub-queries returned the same hits, so doc-42 wins with full overlap.
        assert result.results[0].id == 42
        assert sorted(result.results[0].contributing_query_indices) == [0, 1, 2]

    def test_rejects_blank_input(self):
        router = self._build_router(
            classifier=_StubClassifier(
                RoutingDecision(intent=QueryIntent.SIMPLE_FACTUAL, complexity_score=0.0)
            ),
            decomposer=_StubDecomposer(
                SubQueryPlan(original_query="x", sub_queries=[SubQuery(text="x?")])
            ),
            dense_hits=[],
        )
        with pytest.raises(ValueError, match="question must not be empty"):
            router.route_sync("   ")

    def test_async_route_works_inside_running_loop(self):
        decision = RoutingDecision(
            intent=QueryIntent.SIMPLE_FACTUAL, complexity_score=0.0
        )
        router = self._build_router(
            classifier=_StubClassifier(decision),
            decomposer=_StubDecomposer(
                SubQueryPlan(original_query="x", sub_queries=[SubQuery(text="x?")])
            ),
            dense_hits=[FakeChunk(99)],
        )

        async def driver():
            return await router.route("What is pgvector?")

        result = asyncio.run(driver())
        assert [r.id for r in result.results] == [99]
