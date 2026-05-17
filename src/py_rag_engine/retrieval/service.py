from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from py_rag_engine.domain import DocumentChunk
from py_rag_engine.retrieval.rerank import (
    CrossEncoderReranker,
    RerankedResult,
)
from py_rag_engine.storage import PostgresEmbeddingStore
from py_rag_engine.vector_math import cosine_similarity

DEFAULT_CANDIDATE_K = 20
DEFAULT_TOP_K = 5


def rank_chunks_by_similarity(
    query_embedding: Sequence[float],
    chunk_embeddings: Sequence[Sequence[float]],
    chunks: Sequence[DocumentChunk],
    *,
    top_k: int = 5,
) -> list[tuple[DocumentChunk, float]]:
    """Rank chunks by cosine similarity against a query embedding."""
    if len(chunk_embeddings) != len(chunks):
        raise ValueError("chunk_embeddings and chunks must have the same length")

    scored = [
        (chunk, cosine_similarity(query_embedding, emb))
        for chunk, emb in zip(chunks, chunk_embeddings, strict=True)
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]


def retrieve_with_rerank(
    query: str,
    query_embedding: Sequence[float],
    store: PostgresEmbeddingStore,
    reranker: CrossEncoderReranker,
    *,
    candidate_k: int = DEFAULT_CANDIDATE_K,
    top_k: int = DEFAULT_TOP_K,
    metadata_filter: Mapping[str, Any] | None = None,
    ef_search: int | None = None,
) -> list[RerankedResult]:
    """Dense pgvector retrieval (top candidate_k) followed by cross-encoder re-ranking.

    The pipeline first pulls ``candidate_k`` chunks ordered by pgvector cosine
    distance, then asks the cross-encoder to score each (query, chunk) pair and
    returns the ``top_k`` highest-scoring chunks.
    """
    if candidate_k < 1:
        raise ValueError("candidate_k must be greater than zero")
    if top_k < 1:
        raise ValueError("top_k must be greater than zero")
    if top_k > candidate_k:
        raise ValueError("top_k must not exceed candidate_k")

    candidates = store.similarity_search(
        query_embedding,
        top_k=candidate_k,
        metadata_filter=metadata_filter,
        ef_search=ef_search,
    )
    return reranker.rerank(query, candidates, top_k=top_k)
