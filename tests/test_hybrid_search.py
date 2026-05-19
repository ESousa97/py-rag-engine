from __future__ import annotations

from collections.abc import Sequence

import pytest

from py_rag_engine.retrieval import (
    CrossEncoderReranker,
    HybridSearchResult,
    RerankedResult,
    reciprocal_rank_fusion,
    retrieve_hybrid,
    retrieve_hybrid_with_rerank,
)
from py_rag_engine.storage import FTSSearchResult, SimilaritySearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dense(doc_id: int, cosine: float, text: str = "") -> SimilaritySearchResult:
    return SimilaritySearchResult(
        id=doc_id,
        text=text or f"doc-{doc_id}",
        metadata={"source": "test"},
        content_hash=f"hash-{doc_id}",
        embedding_model="openai-3-small",
        cosine_similarity=cosine,
    )


def _fts(doc_id: int, score: float, text: str = "") -> FTSSearchResult:
    return FTSSearchResult(
        id=doc_id,
        text=text or f"doc-{doc_id}",
        metadata={"source": "test"},
        content_hash=f"hash-{doc_id}",
        embedding_model="openai-3-small",
        fts_score=score,
    )


# ---------------------------------------------------------------------------
# reciprocal_rank_fusion – pure function tests
# ---------------------------------------------------------------------------

def test_rrf_dense_only() -> None:
    dense = [_dense(1, 0.9), _dense(2, 0.8), _dense(3, 0.7)]
    result = reciprocal_rank_fusion(dense, [], top_k=3)
    assert [r.id for r in result] == [1, 2, 3]
    assert all(r.fts_score == 0.0 for r in result)


def test_rrf_fts_only() -> None:
    fts = [_fts(10, 0.5), _fts(20, 0.3)]
    result = reciprocal_rank_fusion([], fts, top_k=2)
    assert [r.id for r in result] == [10, 20]
    assert all(r.cosine_similarity == 0.0 for r in result)


def test_rrf_overlap_boosts_shared_documents() -> None:
    # doc-2 is rank-3 in dense but rank-1 in FTS → should beat doc-1 (rank-1 dense only)
    dense = [_dense(1, 0.99), _dense(2, 0.50), _dense(3, 0.40)]
    fts = [_fts(2, 0.9), _fts(4, 0.7)]

    result = reciprocal_rank_fusion(dense, fts, rrf_k=1, top_k=4)

    # doc-2: 1/(1+2) + 1/(1+1) = 0.333 + 0.5 = 0.833
    # doc-1: 1/(1+1) = 0.5
    # doc-4: 1/(1+2) = 0.333
    # doc-3: 1/(1+3) = 0.25
    ids = [r.id for r in result]
    assert ids[0] == 2, "overlap candidate should win"
    assert ids[1] == 1


def test_rrf_returns_at_most_top_k() -> None:
    dense = [_dense(i, 1.0 / i) for i in range(1, 11)]
    fts = [_fts(i, 1.0 / i) for i in range(1, 11)]
    result = reciprocal_rank_fusion(dense, fts, top_k=3)
    assert len(result) == 3


def test_rrf_empty_both_lists() -> None:
    assert reciprocal_rank_fusion([], [], top_k=5) == []


def test_rrf_result_carries_both_scores() -> None:
    dense = [_dense(1, 0.8)]
    fts = [_fts(1, 0.6)]
    result = reciprocal_rank_fusion(dense, fts, top_k=1)
    assert result[0].cosine_similarity == pytest.approx(0.8)
    assert result[0].fts_score == pytest.approx(0.6)
    assert isinstance(result[0], HybridSearchResult)


def test_rrf_validates_rrf_k() -> None:
    with pytest.raises(ValueError, match="rrf_k must be greater than zero"):
        reciprocal_rank_fusion([], [], rrf_k=0, top_k=1)


def test_rrf_validates_top_k() -> None:
    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        reciprocal_rank_fusion([], [], top_k=0)


def test_rrf_score_formula() -> None:
    # Verify exact RRF score: 1/(60+1) for rank-1 in a single list
    dense = [_dense(99, 0.9)]
    result = reciprocal_rank_fusion(dense, [], rrf_k=60, top_k=1)
    assert result[0].rrf_score == pytest.approx(1.0 / 61.0)


# ---------------------------------------------------------------------------
# retrieve_hybrid – orchestration with a fake store
# ---------------------------------------------------------------------------

class _FakeHybridStore:
    """Records calls to similarity_search and fts_search."""

    def __init__(
        self,
        dense_results: list[SimilaritySearchResult],
        fts_results: list[FTSSearchResult],
    ) -> None:
        self._dense = dense_results
        self._fts = fts_results
        self.dense_call: dict | None = None
        self.fts_call: dict | None = None

    def similarity_search(
        self,
        query_embedding,
        *,
        top_k,
        metadata_filter=None,
        ef_search=None,
    ) -> list[SimilaritySearchResult]:
        self.dense_call = {
            "top_k": top_k,
            "metadata_filter": metadata_filter,
            "ef_search": ef_search,
        }
        return list(self._dense[:top_k])

    def fts_search(
        self,
        query_text,
        *,
        top_k,
        metadata_filter=None,
    ) -> list[FTSSearchResult]:
        self.fts_call = {
            "query_text": query_text,
            "top_k": top_k,
            "metadata_filter": metadata_filter,
        }
        return list(self._fts[:top_k])


def test_retrieve_hybrid_calls_both_searches() -> None:
    dense = [_dense(i, 1.0 - i * 0.1) for i in range(1, 6)]
    fts = [_fts(10 + i, 1.0 - i * 0.1) for i in range(1, 6)]
    store = _FakeHybridStore(dense, fts)

    result = retrieve_hybrid(
        [0.1] * 4,
        "search query",
        store,  # type: ignore[arg-type]
        dense_k=5,
        fts_k=3,
        top_k=4,
        metadata_filter={"source": "x"},
        ef_search=80,
    )

    assert store.dense_call == {"top_k": 5, "metadata_filter": {"source": "x"}, "ef_search": 80}
    assert store.fts_call == {"query_text": "search query", "top_k": 3, "metadata_filter": {"source": "x"}}
    assert len(result) == 4
    assert all(isinstance(r, HybridSearchResult) for r in result)


def test_retrieve_hybrid_validates_dense_k() -> None:
    store = _FakeHybridStore([], [])
    with pytest.raises(ValueError, match="dense_k must be greater than zero"):
        retrieve_hybrid([0.0], "q", store, dense_k=0)  # type: ignore[arg-type]


def test_retrieve_hybrid_validates_fts_k() -> None:
    store = _FakeHybridStore([], [])
    with pytest.raises(ValueError, match="fts_k must be greater than zero"):
        retrieve_hybrid([0.0], "q", store, fts_k=0)  # type: ignore[arg-type]


def test_retrieve_hybrid_validates_top_k() -> None:
    store = _FakeHybridStore([], [])
    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        retrieve_hybrid([0.0], "q", store, top_k=0)  # type: ignore[arg-type]


def test_retrieve_hybrid_overlap_wins_over_dense_only() -> None:
    # doc-5 is last in dense but first in FTS → should win after RRF
    dense = [_dense(1, 0.95), _dense(2, 0.90), _dense(3, 0.85), _dense(4, 0.80), _dense(5, 0.75)]
    fts = [_fts(5, 0.99)]  # doc-5 is top FTS hit
    store = _FakeHybridStore(dense, fts)

    result = retrieve_hybrid(
        [0.0] * 2, "error code XYZ", store, dense_k=5, fts_k=5, top_k=5, rrf_k=1  # type: ignore[arg-type]
    )
    ids = [r.id for r in result]
    # doc-5 gets 1/(1+5) + 1/(1+1) = 0.167 + 0.5 = 0.667
    # doc-1 gets 1/(1+1) = 0.5
    assert ids[0] == 5


# ---------------------------------------------------------------------------
# retrieve_hybrid_with_rerank – full pipeline
# ---------------------------------------------------------------------------

def test_retrieve_hybrid_with_rerank_runs_pipeline() -> None:
    dense = [_dense(i, 1.0 - i * 0.05, text=f"chunk-{i}") for i in range(1, 11)]
    fts = [_fts(5, 0.8, text="chunk-5"), _fts(8, 0.6, text="chunk-8")]
    store = _FakeHybridStore(dense, fts)

    # Cross-encoder boosts doc-5 (which also has FTS hit) to the top
    def predict(pairs: Sequence[tuple[str, str]]) -> list[float]:
        scores: list[float] = []
        for _, text in pairs:
            idx = int(text.removeprefix("chunk-"))
            scores.append(10.0 if idx == 5 else -float(idx))
        return scores

    reranker = CrossEncoderReranker(predict=predict)
    result = retrieve_hybrid_with_rerank(
        "error code XYZ",
        [0.1] * 4,
        store,  # type: ignore[arg-type]
        reranker,
        dense_k=10,
        fts_k=5,
        top_k=3,
    )

    assert len(result) == 3
    assert result[0].id == 5
    assert all(isinstance(r, RerankedResult) for r in result)


def test_retrieve_hybrid_with_rerank_validates_top_k() -> None:
    store = _FakeHybridStore([], [])
    reranker = CrossEncoderReranker(predict=lambda pairs: [])
    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        retrieve_hybrid_with_rerank(
            "q", [0.0], store, reranker, top_k=0  # type: ignore[arg-type]
        )
