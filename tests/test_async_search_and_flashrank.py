from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from py_rag_engine.reranker import FlashrankReranker, RerankedDocument, rerank_documents
from py_rag_engine.search import (
    AsyncHybridSearcher,
    FullTextSearchDocument,
    HybridSearchDocument,
    VectorSearchDocument,
    _vector_literal,
    reciprocal_rank_fusion,
)


def _vector_doc(doc_id: int, *, rank: int, text: str | None = None) -> VectorSearchDocument:
    return VectorSearchDocument(
        id=doc_id,
        text=text or f"doc-{doc_id}",
        metadata={"source": "test"},
        content_hash=f"hash-{doc_id}",
        embedding_model="bge-m3",
        vector_rank=rank,
        cosine_distance=rank / 100.0,
        cosine_similarity=1.0 - rank / 100.0,
    )


def _fts_doc(doc_id: int, *, rank: int, text: str | None = None) -> FullTextSearchDocument:
    return FullTextSearchDocument(
        id=doc_id,
        text=text or f"doc-{doc_id}",
        metadata={"source": "test"},
        content_hash=f"hash-{doc_id}",
        embedding_model="bge-m3",
        fts_rank=rank,
        fts_score=1.0 / rank,
    )


def test_reciprocal_rank_fusion_returns_top_20_by_default() -> None:
    vector = [_vector_doc(doc_id, rank=doc_id) for doc_id in range(1, 31)]
    fts = [_fts_doc(30 + doc_id, rank=doc_id) for doc_id in range(1, 31)]

    result = reciprocal_rank_fusion(vector, fts)

    assert len(result) == 20
    assert result[0].id == 1
    assert result[0].confidence_score == pytest.approx(1.0 / 61.0)


def test_reciprocal_rank_fusion_boosts_documents_in_both_lists() -> None:
    vector = [_vector_doc(1, rank=1), _vector_doc(2, rank=2)]
    fts = [_fts_doc(2, rank=1)]

    result = reciprocal_rank_fusion(vector, fts, rrf_k=1, top_k=2)

    assert [item.id for item in result] == [2, 1]
    assert result[0].vector_rank == 2
    assert result[0].fts_rank == 1


def test_vector_literal_rejects_invalid_embeddings() -> None:
    assert _vector_literal([0.1, -2.0]) == "[0.10000000000000001,-2]"
    with pytest.raises(ValueError, match="must not be empty"):
        _vector_literal([])
    with pytest.raises(ValueError, match="finite numbers"):
        _vector_literal([float("nan")])


class _FakeAsyncHybridSearcher(AsyncHybridSearcher):
    def __init__(
        self,
        vector_results: list[VectorSearchDocument],
        fts_results: list[FullTextSearchDocument],
    ) -> None:
        super().__init__(object())  # type: ignore[arg-type]
        self.vector_results = vector_results
        self.fts_results = fts_results

    async def _vector_search(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        metadata_filter: Mapping[str, Any] | None,
        ef_search: int | None,
    ) -> list[VectorSearchDocument]:
        await asyncio.sleep(0.05)
        return self.vector_results[:top_k]

    async def _full_text_search(
        self,
        query_text: str,
        *,
        top_k: int,
        metadata_filter: Mapping[str, Any] | None,
    ) -> list[FullTextSearchDocument]:
        await asyncio.sleep(0.05)
        return self.fts_results[:top_k]


def test_async_hybrid_search_runs_vector_and_fts_in_parallel() -> None:
    searcher = _FakeAsyncHybridSearcher(
        [_vector_doc(1, rank=1), _vector_doc(2, rank=2)],
        [_fts_doc(2, rank=1)],
    )

    async def run() -> tuple[list[HybridSearchDocument], float]:
        start = time.perf_counter()
        result = await searcher.search("query", [0.1, 0.2], top_k=2, rrf_k=1)
        return result, time.perf_counter() - start

    result, elapsed = asyncio.run(run())

    assert [item.id for item in result] == [2, 1]
    assert elapsed < 0.09


class _FakeRanker:
    def rerank(self, request: Any) -> list[dict[str, Any]]:
        assert request.query == "original query"
        scored = []
        for passage in request.passages:
            score = 0.9 if passage["id"] == 3 else 0.1 * passage["id"]
            scored.append({**passage, "score": score})
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored


def test_flashrank_reranker_returns_top_5_with_updated_scores() -> None:
    docs = [
        HybridSearchDocument(
            id=doc_id,
            text=f"doc-{doc_id}",
            metadata={"source": "test"},
            content_hash=f"hash-{doc_id}",
            embedding_model="bge-m3",
            rrf_score=1.0 / doc_id,
            confidence_score=1.0 / doc_id,
            vector_rank=doc_id,
            fts_rank=None,
            cosine_similarity=0.9,
            fts_score=None,
        )
        for doc_id in range(1, 21)
    ]
    reranker = FlashrankReranker(ranker=_FakeRanker())

    result = rerank_documents("original query", docs, reranker=reranker)

    assert len(result) == 5
    assert [item.id for item in result] == [20, 19, 18, 17, 16]
    assert all(isinstance(item, RerankedDocument) for item in result)
    assert result[0].confidence_score == pytest.approx(2.0)
    assert result[0].rerank_score == result[0].confidence_score


def test_flashrank_reranker_validates_top_k() -> None:
    reranker = FlashrankReranker(ranker=_FakeRanker())
    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        reranker.rerank("q", [{"id": 1, "text": "x"}], top_k=0)
