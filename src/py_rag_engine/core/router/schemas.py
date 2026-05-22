"""Pydantic models for the query-ingestion router.

These schemas double as:

* the request/response contract for downstream consumers,
* the JSON-Schema document we hand to an LLM when asking for *structured*
  output (works with OpenAI's `response_format={"type": "json_schema"}` and
  with Anthropic's tool-use / `output_format` patterns alike).

Keeping every field annotated and validated here means the router never has
to babysit raw dictionaries returned by the model.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MAX_SUB_QUERIES = 3


class QueryIntent(str, Enum):
    """High-level intent classes the router can recognise."""

    SIMPLE_FACTUAL = "simple_factual"
    COMPLEX_MULTI_HOP = "complex_multi_hop"
    COMPARATIVE = "comparative"
    CONVERSATIONAL = "conversational"
    OUT_OF_SCOPE = "out_of_scope"

    @property
    def needs_decomposition(self) -> bool:
        """Intents whose questions benefit from sub-query generation."""
        return self in {QueryIntent.COMPLEX_MULTI_HOP, QueryIntent.COMPARATIVE}


class RoutingDecision(BaseModel):
    """Classifier output — the LLM is asked to fill this exact shape."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    intent: QueryIntent = Field(
        ...,
        description=(
            "Best matching intent class. Use 'complex_multi_hop' for questions "
            "that require chaining several facts, 'comparative' for A-vs-B "
            "questions, 'simple_factual' for single-fact lookups, "
            "'conversational' for chit-chat, 'out_of_scope' otherwise."
        ),
    )
    complexity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="0.0 = trivial single lookup, 1.0 = highly decomposable.",
    )
    reasoning: str = Field(
        default="",
        max_length=500,
        description="Short justification (<=2 sentences).",
    )
    needs_decomposition: bool = Field(
        default=False,
        description=(
            "True when the question should be split into sub-queries. "
            "Defaults follow `intent.needs_decomposition` if the model omits it."
        ),
    )

    @model_validator(mode="after")
    def _sync_decomposition_flag(self) -> "RoutingDecision":
        # If the LLM emitted a complex intent but forgot to set the flag,
        # backfill so downstream code can trust a single field.
        if self.intent.needs_decomposition and not self.needs_decomposition:
            object.__setattr__(self, "needs_decomposition", True)
        return self


class SubQuery(BaseModel):
    """One independent sub-question produced by the decomposer."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    text: str = Field(..., min_length=1, max_length=400)
    reasoning: str = Field(default="", max_length=300)

    @field_validator("text")
    @classmethod
    def _strip_and_validate(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("sub-query text must not be blank")
        return cleaned


class SubQueryPlan(BaseModel):
    """Bundle of sub-queries (1..MAX_SUB_QUERIES) plus the original prompt."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    original_query: str = Field(..., min_length=1)
    # No `max_length` here — we cap+dedup in the validator below so we can
    # absorb an over-eager LLM gracefully instead of raising on it.
    sub_queries: list[SubQuery] = Field(..., min_length=1)

    @field_validator("sub_queries", mode="before")
    @classmethod
    def _enforce_cap(cls, value: Any) -> Any:
        if not isinstance(value, list):
            return value
        # Hard cap even if the model returns more, dedup by lowercased text.
        seen: set[str] = set()
        unique: list[Any] = []
        for sq in value:
            text = sq.text if isinstance(sq, SubQuery) else (sq or {}).get("text", "")
            key = (text or "").strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(sq)
            if len(unique) == MAX_SUB_QUERIES:
                break
        return unique

    @property
    def texts(self) -> list[str]:
        return [sq.text for sq in self.sub_queries]


class FusedRetrievalResult(BaseModel):
    """One row in the final fused result list.

    Wraps a single underlying chunk plus the multi-query RRF score and the
    list of sub-queries (by index) that contributed to it. This makes the
    fusion auditable downstream.
    """

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: int
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    content_hash: str | None = None
    embedding_model: str = ""
    rrf_score: float
    contributing_query_indices: tuple[int, ...] = Field(default_factory=tuple)


class RoutedQueryResult(BaseModel):
    """Top-level router payload returned to callers."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    decision: RoutingDecision
    plan: SubQueryPlan | None
    results: list[FusedRetrievalResult]

    @property
    def was_decomposed(self) -> bool:
        return self.plan is not None and len(self.plan.sub_queries) > 1
