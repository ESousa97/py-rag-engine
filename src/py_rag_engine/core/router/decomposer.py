"""Sub-query generation for complex questions.

Given a single user question flagged as complex by the classifier, this
module produces a `SubQueryPlan` containing up to `MAX_SUB_QUERIES` (3)
independent sub-questions. Each sub-question is meant to be answerable from
the knowledge base on its own — the parallel retriever fans out one search
per sub-query and the multi-query RRF fuser merges the results.

The decomposer never emits more than 3 sub-queries: that cap matters because
(a) parallel retrieval cost scales linearly, and (b) the LLM tends to keep
quality high with a tight budget rather than padding with weak rephrasings.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from py_rag_engine.core.router.llm import ChatFn, call_structured
from py_rag_engine.core.router.schemas import (
    MAX_SUB_QUERIES,
    SubQuery,
    SubQueryPlan,
)

_DECOMPOSER_SYSTEM_PROMPT = (
    "You decompose complex user questions into a small set of independent "
    "sub-questions that a retrieval engine can answer in parallel. You always "
    "reply with valid JSON matching the schema."
)

_DECOMPOSER_USER_TEMPLATE = (
    "Decompose the question below into AT MOST {max_sub} self-contained "
    "sub-questions. Rules:\n"
    "  * Each sub-question must be answerable independently (no pronouns "
    "referring to other sub-questions).\n"
    "  * Cover distinct facets — do NOT paraphrase the same question.\n"
    "  * If the question is already atomic, return exactly one sub-question "
    "that mirrors it.\n"
    "  * Keep each sub-question under 30 words.\n\n"
    "Original question: {question}\n\n"
    "Set 'original_query' to the verbatim original question."
)


def _split_on_clauses(text: str) -> list[str]:
    """Naive clause splitter for the heuristic decomposer."""
    # Split on `?`, ` and `, ` also `, semicolons, ` as well as `.
    parts = re.split(r"(?:\?|;|\band\b|\balso\b|\bas well as\b)", text, flags=re.IGNORECASE)
    cleaned = []
    for p in parts:
        chunk = p.strip(" ,.")
        if len(chunk) >= 3:
            cleaned.append(chunk)
    return cleaned


def heuristic_decompose(question: str, *, max_sub_queries: int = MAX_SUB_QUERIES) -> SubQueryPlan:
    """Fallback decomposer using punctuation/conjunction splitting.

    Always returns at least one sub-query (mirroring the original) so callers
    can treat the result uniformly.
    """
    if max_sub_queries < 1:
        raise ValueError("max_sub_queries must be >= 1")
    cap = min(max_sub_queries, MAX_SUB_QUERIES)

    pieces = _split_on_clauses(question)
    # Drop near-duplicate pieces, keep first occurrences.
    seen: set[str] = set()
    deduped: list[str] = []
    for piece in pieces:
        key = piece.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(piece)

    if not deduped:
        deduped = [question.strip() or question]

    # If splitting produced just one chunk that is essentially the whole
    # question, return it as-is rather than fabricating fake variations.
    sub_texts = deduped[:cap]
    sub_queries = [
        SubQuery(text=t if t.endswith("?") else f"{t}?", reasoning="heuristic clause split")
        for t in sub_texts
    ]
    return SubQueryPlan(original_query=question.strip() or question, sub_queries=sub_queries)


@dataclass(slots=True)
class LLMSubQueryGenerator:
    """LLM-backed decomposer with structured Pydantic output."""

    chat: ChatFn
    max_sub_queries: int = MAX_SUB_QUERIES
    temperature: float = 0.2
    max_tokens: int = 500
    system_prompt: str = _DECOMPOSER_SYSTEM_PROMPT
    fallback_on_error: bool = True

    def __post_init__(self) -> None:
        if self.max_sub_queries < 1:
            raise ValueError("max_sub_queries must be >= 1")
        if self.max_sub_queries > MAX_SUB_QUERIES:
            # Enforced statically too, but be loud if a caller tries to override.
            raise ValueError(f"max_sub_queries cannot exceed {MAX_SUB_QUERIES}")

    def generate(self, question: str) -> SubQueryPlan:
        try:
            plan = call_structured(
                self.chat,
                system=self.system_prompt,
                user=_DECOMPOSER_USER_TEMPLATE.format(
                    max_sub=self.max_sub_queries,
                    question=question,
                ),
                response_model=SubQueryPlan,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception:
            if not self.fallback_on_error:
                raise
            return heuristic_decompose(question, max_sub_queries=self.max_sub_queries)

        # Schema already caps at MAX_SUB_QUERIES via the validator; enforce the
        # tighter per-instance cap if the caller asked for fewer.
        if len(plan.sub_queries) > self.max_sub_queries:
            plan = SubQueryPlan(
                original_query=plan.original_query,
                sub_queries=list(plan.sub_queries[: self.max_sub_queries]),
            )
        return plan


__all__ = ["LLMSubQueryGenerator", "heuristic_decompose"]
