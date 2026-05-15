from __future__ import annotations

from collections.abc import Sequence

import pytest

from py_rag_engine.domain import ChunkMetadata, DocumentChunk
from py_rag_engine.retrieval import (
    CrossEncoderReranker,
    RerankedResult,
    cosine_similarity,
    rank_chunks_by_similarity,
    rerank_candidates,
    retrieve_with_rerank,
)
from py_rag_engine.storage import SimilaritySearchResult


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


def _make_candidate(
    candidate_id: int,
    text: str,
    cosine: float,
    *,
    embedding_model: str = "openai-3-small",
) -> SimilaritySearchResult:
    return SimilaritySearchResult(
        id=candidate_id,
        text=text,
        metadata={"source": "test", "chunk_index": candidate_id},
        content_hash=f"hash-{candidate_id}",
        embedding_model=embedding_model,
        cosine_similarity=cosine,
    )


def test_rerank_candidates_orders_by_cross_encoder_score() -> None:
    candidates = [
        _make_candidate(1, "irrelevant text about cooking", cosine=0.95),
        _make_candidate(2, "exact answer to query", cosine=0.40),
        _make_candidate(3, "tangentially related", cosine=0.80),
    ]
    # Dense order is 1 > 3 > 2; cross-encoder flips it so 2 wins.
    cross_scores = {1: -2.5, 2: 9.1, 3: 1.3}

    def predict(pairs: Sequence[tuple[str, str]]) -> list[float]:
        scores = []
        for query, text in pairs:
            assert query == "what is the answer"
            candidate = next(c for c in candidates if c.text == text)
            scores.append(cross_scores[candidate.id])
        return scores

    reranked = rerank_candidates("what is the answer", candidates, predict, top_k=2)

    assert [item.id for item in reranked] == [2, 3]
    assert reranked[0].rerank_score == pytest.approx(9.1)
    assert reranked[0].cosine_similarity == pytest.approx(0.40)
    assert isinstance(reranked[0], RerankedResult)


def test_rerank_candidates_empty_returns_empty() -> None:
    def predict(pairs: Sequence[tuple[str, str]]) -> list[float]:
        raise AssertionError("predict should not be called for empty candidates")

    assert rerank_candidates("q", [], predict, top_k=5) == []


def test_rerank_candidates_validates_top_k() -> None:
    candidate = _make_candidate(1, "x", cosine=0.5)
    with pytest.raises(ValueError, match="top_k must be greater than zero"):
        rerank_candidates("q", [candidate], lambda pairs: [0.0], top_k=0)


def test_rerank_candidates_rejects_mismatched_predict_output() -> None:
    candidates = [
        _make_candidate(1, "a", cosine=0.9),
        _make_candidate(2, "b", cosine=0.8),
    ]
    with pytest.raises(ValueError, match="predict returned 1 scores for 2 candidates"):
        rerank_candidates("q", candidates, lambda pairs: [0.0], top_k=2)


def test_cross_encoder_reranker_uses_injected_predict() -> None:
    candidates = [
        _make_candidate(1, "a", cosine=0.9),
        _make_candidate(2, "b", cosine=0.5),
    ]
    captured: list[tuple[str, str]] = []

    def predict(pairs: Sequence[tuple[str, str]]) -> list[float]:
        captured.extend(pairs)
        return [0.1, 0.9]

    reranker = CrossEncoderReranker(predict=predict)
    reranked = reranker.rerank("q", candidates, top_k=1)

    assert captured == [("q", "a"), ("q", "b")]
    assert len(reranked) == 1
    assert reranked[0].id == 2


def test_cross_encoder_reranker_validates_batch_size() -> None:
    with pytest.raises(ValueError, match="batch_size must be greater than zero"):
        CrossEncoderReranker(batch_size=0)


class _FakeStore:
    """In-memory stand-in for PostgresEmbeddingStore used by the pipeline test."""

    def __init__(self, results: list[SimilaritySearchResult]) -> None:
        self._results = results
        self.last_call: dict[str, object] | None = None

    def similarity_search(
        self,
        query_embedding,
        *,
        top_k,
        metadata_filter=None,
        ef_search=None,
    ) -> list[SimilaritySearchResult]:
        self.last_call = {
            "query_embedding": list(query_embedding),
            "top_k": top_k,
            "metadata_filter": metadata_filter,
            "ef_search": ef_search,
        }
        return list(self._results[:top_k])


def test_retrieve_with_rerank_runs_dense_then_cross_encoder() -> None:
    dense_results = [
        _make_candidate(idx, f"doc-{idx}", cosine=1.0 - idx * 0.01)
        for idx in range(20)
    ]
    store = _FakeStore(dense_results)

    # Cross-encoder picks doc-7 and doc-14 as best, flipping the dense ranking.
    def predict(pairs):
        scores: list[float] = []
        for _, text in pairs:
            idx = int(text.removeprefix("doc-"))
            if idx == 7:
                scores.append(10.0)
            elif idx == 14:
                scores.append(8.0)
            else:
                scores.append(-idx * 0.1)
        return scores

    reranker = CrossEncoderReranker(predict=predict)

    top = retrieve_with_rerank(
        "user query",
        [0.1] * 4,
        store,  # type: ignore[arg-type]
        reranker,
        candidate_k=20,
        top_k=5,
        metadata_filter={"source": "test"},
        ef_search=80,
    )

    assert store.last_call == {
        "query_embedding": [0.1] * 4,
        "top_k": 20,
        "metadata_filter": {"source": "test"},
        "ef_search": 80,
    }
    assert len(top) == 5
    assert top[0].id == 7
    assert top[1].id == 14
    # Remaining slots are filled by the least-negative scores (lowest dense indexes).
    assert [item.id for item in top[2:]] == [0, 1, 2]


def test_retrieve_with_rerank_validates_top_k_against_candidate_k() -> None:
    store = _FakeStore([])
    reranker = CrossEncoderReranker(predict=lambda pairs: [])
    with pytest.raises(ValueError, match="top_k must not exceed candidate_k"):
        retrieve_with_rerank(
            "q",
            [0.0],
            store,  # type: ignore[arg-type]
            reranker,
            candidate_k=3,
            top_k=5,
        )


def test_retrieve_with_rerank_validates_candidate_k() -> None:
    store = _FakeStore([])
    reranker = CrossEncoderReranker(predict=lambda pairs: [])
    with pytest.raises(ValueError, match="candidate_k must be greater than zero"):
        retrieve_with_rerank(
            "q",
            [0.0],
            store,  # type: ignore[arg-type]
            reranker,
            candidate_k=0,
            top_k=1,
        )
