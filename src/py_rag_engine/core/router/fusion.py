"""Multi-query Reciprocal Rank Fusion.

The existing `py_rag_engine.retrieval.hybrid.reciprocal_rank_fusion` fuses
exactly TWO ranked lists (dense + FTS). The router needs the N-list variant:
one ranked list per sub-query, fused into a single ordering with provenance
back to the sub-queries that contributed.

The math is the canonical RRF formula:

    score(d) = sum over each list L containing d of  1 / (k + rank_L(d))

where rank starts at 1. Documents appearing in multiple lists naturally win.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from py_rag_engine.core.router.schemas import FusedRetrievalResult

DEFAULT_MULTI_QUERY_RRF_K = 60
DEFAULT_MULTI_QUERY_TOP_K = 5


@runtime_checkable
class _RankedItem(Protocol):
    """Structural subset every retriever result type happens to expose.

    `HybridSearchResult`, `SimilaritySearchResult`, and `RerankedResult` all
    satisfy this — see `py_rag_engine.retrieval` and `py_rag_engine.storage`.
    """

    id: int
    text: str
    metadata: dict[str, Any]
    content_hash: str | None
    embedding_model: str


def multi_query_rrf(
    ranked_lists: Sequence[Sequence[_RankedItem]],
    *,
    rrf_k: int = DEFAULT_MULTI_QUERY_RRF_K,
    top_k: int = DEFAULT_MULTI_QUERY_TOP_K,
) -> list[FusedRetrievalResult]:
    """Fuse `len(ranked_lists)` ranked candidate lists with RRF.

    Args:
        ranked_lists: One ranked list per sub-query (order = rank).
        rrf_k: Standard RRF dampener. 60 is the literature default.
        top_k: Maximum number of fused results to return.

    Returns:
        Up to `top_k` `FusedRetrievalResult` items sorted by descending RRF
        score. Each carries the indices of the sub-queries that surfaced it.

    Raises:
        ValueError: when `rrf_k` or `top_k` are not positive.
    """
    if rrf_k < 1:
        raise ValueError("rrf_k must be greater than zero")
    if top_k < 1:
        raise ValueError("top_k must be greater than zero")

    scores: dict[int, float] = {}
    contributors: dict[int, list[int]] = {}
    canonical: dict[int, _RankedItem] = {}

    for query_idx, results in enumerate(ranked_lists):
        for rank, item in enumerate(results, start=1):
            doc_id = item.id
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (rrf_k + rank)
            contributors.setdefault(doc_id, []).append(query_idx)
            # Keep the first seen object as the canonical text/metadata source;
            # subsequent lists for the same doc are guaranteed equivalent text.
            canonical.setdefault(doc_id, item)

    sorted_ids = sorted(scores, key=lambda d: scores[d], reverse=True)

    out: list[FusedRetrievalResult] = []
    for doc_id in sorted_ids[:top_k]:
        item = canonical[doc_id]
        out.append(
            FusedRetrievalResult(
                id=doc_id,
                text=item.text,
                metadata=dict(item.metadata),
                content_hash=item.content_hash,
                embedding_model=item.embedding_model,
                rrf_score=scores[doc_id],
                contributing_query_indices=tuple(contributors[doc_id]),
            )
        )
    return out


__all__ = [
    "DEFAULT_MULTI_QUERY_RRF_K",
    "DEFAULT_MULTI_QUERY_TOP_K",
    "multi_query_rrf",
]
