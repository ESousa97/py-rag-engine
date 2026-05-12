from __future__ import annotations

import pytest

from py_rag_engine.domain import ChunkMetadata, DocumentChunk
from py_rag_engine.retrieval.semantic import cosine_similarity
from py_rag_engine.retrieval.service import rank_chunks_by_similarity


def test_cosine_similarity_basic() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_rank_chunks_by_similarity() -> None:
    chunks = [
        DocumentChunk("a", ChunkMetadata(source="x", page=None), "h1"),
        DocumentChunk("b", ChunkMetadata(source="x", page=None), "h2"),
    ]
    embeddings = [[1.0, 0.0], [0.0, 1.0]]

    ranked = rank_chunks_by_similarity([1.0, 0.0], embeddings, chunks, top_k=1)
    assert len(ranked) == 1
    assert ranked[0][0].text == "a"


def test_rank_chunks_size_mismatch() -> None:
    chunks = [DocumentChunk("a", ChunkMetadata(source="x", page=None), "h1")]
    with pytest.raises(ValueError):
        rank_chunks_by_similarity([1.0], [], chunks)
