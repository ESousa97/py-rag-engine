from __future__ import annotations

from collections.abc import Sequence

from py_rag_engine.domain import DocumentChunk
from py_rag_engine.retrieval.semantic import cosine_similarity


def rank_chunks_by_similarity(
    query_embedding: Sequence[float],
    chunk_embeddings: Sequence[Sequence[float]],
    chunks: Sequence[DocumentChunk],
    *,
    top_k: int = 5,
) -> list[tuple[DocumentChunk, float]]:
    if len(chunk_embeddings) != len(chunks):
        raise ValueError("chunk_embeddings and chunks must have the same length")

    scored = [
        (chunk, cosine_similarity(query_embedding, emb))
        for chunk, emb in zip(chunks, chunk_embeddings, strict=True)
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:top_k]
