from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from py_rag_engine.storage.postgres import (
    FTSSearchResult,
    PostgresEmbeddingStore,
    SimilaritySearchResult,
)

DEFAULT_RRF_K = 60
DEFAULT_DENSE_K = 20
DEFAULT_FTS_K = 20
DEFAULT_HYBRID_TOP_K = 5


@dataclass(frozen=True, slots=True)
class HybridSearchResult:
    """A retrieval result produced by RRF-fused dense + full-text search."""

    id: int
    text: str
    metadata: dict[str, Any]
    content_hash: str | None
    embedding_model: str
    rrf_score: float
    cosine_similarity: float
    fts_score: float


def reciprocal_rank_fusion(
    dense_results: Sequence[SimilaritySearchResult],
    fts_results: Sequence[FTSSearchResult],
    *,
    rrf_k: int = DEFAULT_RRF_K,
    top_k: int = DEFAULT_HYBRID_TOP_K,
) -> list[HybridSearchResult]:
    """Merge dense and FTS ranked lists using Reciprocal Rank Fusion.

    Each document scores 1/(rrf_k + rank) per list it appears in. Documents
    present in both lists receive contributions from both, boosting recall
    overlap above results that appear in only one list.
    """
    if rrf_k < 1:
        raise ValueError("rrf_k must be greater than zero")
    if top_k < 1:
        raise ValueError("top_k must be greater than zero")

    scores: dict[int, float] = {}
    dense_by_id: dict[int, SimilaritySearchResult] = {}
    fts_by_id: dict[int, FTSSearchResult] = {}

    for rank, result in enumerate(dense_results, start=1):
        scores[result.id] = scores.get(result.id, 0.0) + 1.0 / (rrf_k + rank)
        dense_by_id[result.id] = result

    for rank, result in enumerate(fts_results, start=1):
        scores[result.id] = scores.get(result.id, 0.0) + 1.0 / (rrf_k + rank)
        fts_by_id[result.id] = result

    sorted_ids = sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)

    out: list[HybridSearchResult] = []
    for doc_id in sorted_ids[:top_k]:
        source = dense_by_id.get(doc_id) or fts_by_id[doc_id]
        out.append(
            HybridSearchResult(
                id=doc_id,
                text=source.text,
                metadata=source.metadata,
                content_hash=source.content_hash,
                embedding_model=source.embedding_model,
                rrf_score=scores[doc_id],
                cosine_similarity=dense_by_id[doc_id].cosine_similarity if doc_id in dense_by_id else 0.0,
                fts_score=fts_by_id[doc_id].fts_score if doc_id in fts_by_id else 0.0,
            )
        )
    return out


def retrieve_hybrid(
    query_embedding: Sequence[float],
    query_text: str,
    store: PostgresEmbeddingStore,
    *,
    dense_k: int = DEFAULT_DENSE_K,
    fts_k: int = DEFAULT_FTS_K,
    top_k: int = DEFAULT_HYBRID_TOP_K,
    rrf_k: int = DEFAULT_RRF_K,
    metadata_filter: Mapping[str, Any] | None = None,
    ef_search: int | None = None,
) -> list[HybridSearchResult]:
    """Two-track retrieval: dense ANN + full-text search, fused with RRF."""
    if dense_k < 1:
        raise ValueError("dense_k must be greater than zero")
    if fts_k < 1:
        raise ValueError("fts_k must be greater than zero")
    if top_k < 1:
        raise ValueError("top_k must be greater than zero")

    dense_results = store.similarity_search(
        query_embedding,
        top_k=dense_k,
        metadata_filter=metadata_filter,
        ef_search=ef_search,
    )
    fts_results = store.fts_search(
        query_text,
        top_k=fts_k,
        metadata_filter=metadata_filter,
    )
    return reciprocal_rank_fusion(dense_results, fts_results, rrf_k=rrf_k, top_k=top_k)
