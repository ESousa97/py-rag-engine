from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

from py_rag_engine.storage import SimilaritySearchResult

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CrossEncoderPredictFn: TypeAlias = Callable[[Sequence[tuple[str, str]]], Sequence[float]]


@dataclass(frozen=True, slots=True)
class RerankedResult:
    """A retrieval result enriched with a cross-encoder relevance score."""

    id: int
    text: str
    metadata: dict[str, Any]
    content_hash: str | None
    embedding_model: str
    cosine_similarity: float
    rerank_score: float


def rerank_candidates(
    query: str,
    candidates: Sequence[SimilaritySearchResult],
    predict: CrossEncoderPredictFn,
    *,
    top_k: int = 5,
) -> list[RerankedResult]:
    """Re-rank dense-search candidates with a cross-encoder and keep the top_k."""
    if top_k < 1:
        raise ValueError("top_k must be greater than zero")
    if not candidates:
        return []

    pairs: list[tuple[str, str]] = [(query, candidate.text) for candidate in candidates]
    scores = predict(pairs)
    if len(scores) != len(candidates):
        raise ValueError(
            f"predict returned {len(scores)} scores for {len(candidates)} candidates"
        )

    scored = [
        RerankedResult(
            id=candidate.id,
            text=candidate.text,
            metadata=candidate.metadata,
            content_hash=candidate.content_hash,
            embedding_model=candidate.embedding_model,
            cosine_similarity=candidate.cosine_similarity,
            rerank_score=float(score),
        )
        for candidate, score in zip(candidates, scores, strict=True)
    ]
    scored.sort(key=lambda item: item.rerank_score, reverse=True)
    return scored[:top_k]


class CrossEncoderReranker:
    """Cross-encoder re-ranker backed by sentence-transformers.

    The underlying model is loaded lazily so importing this class stays cheap
    and tests can inject a stub predict function without pulling heavy
    dependencies.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_RERANKER_MODEL,
        *,
        predict: CrossEncoderPredictFn | None = None,
        batch_size: int = 32,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be greater than zero")
        self.model_name = model_name
        self.batch_size = batch_size
        self._predict = predict

    def _resolve_predict(self) -> CrossEncoderPredictFn:
        if self._predict is not None:
            return self._predict
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker; "
                "install with: pip install 'py-rag-engine[embeddings]'"
            ) from exc

        model = CrossEncoder(self.model_name)
        batch_size = self.batch_size

        def _predict(pairs: Sequence[tuple[str, str]]) -> list[float]:
            scores = model.predict(list(pairs), batch_size=batch_size, show_progress_bar=False)
            return [float(score) for score in scores]

        self._predict = _predict
        return self._predict

    def rerank(
        self,
        query: str,
        candidates: Sequence[SimilaritySearchResult],
        *,
        top_k: int = 5,
    ) -> list[RerankedResult]:
        """Score candidates against the query and return the top_k re-ranked."""
        predict = self._resolve_predict()
        return rerank_candidates(query, candidates, predict, top_k=top_k)
