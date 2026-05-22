from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol, TypeAlias

from py_rag_engine.chunking.semantic import split_paragraphs
from py_rag_engine.domain import ChunkMetadata, DocumentChunk
from py_rag_engine.embeddings.hashing import content_sha256
from py_rag_engine.vector_math import cosine_similarity

AsyncEmbeddingBatchFn: TypeAlias = Callable[[list[str]], Awaitable[list[list[float]]]]
DEFAULT_DISTANCE_THRESHOLD = 0.55


class EmbeddingClient(Protocol):
    """Async embedding client contract used by `SemanticChunker`."""

    async def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text."""


class SemanticChunker:
    """Create idempotent chunks by grouping adjacent semantic paragraphs."""

    def __init__(
        self,
        embedder: EmbeddingClient | None = None,
        *,
        embed: AsyncEmbeddingBatchFn | None = None,
        distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
        max_paragraphs_per_chunk: int = 48,
    ) -> None:
        _validate_embedding_provider(embedder=embedder, embed=embed)
        if not 0.0 <= distance_threshold <= 2.0:
            raise ValueError("distance_threshold must be between 0.0 and 2.0.")
        if max_paragraphs_per_chunk < 1:
            raise ValueError("max_paragraphs_per_chunk must be at least 1.")

        self._embedder = embedder
        self._embed = embed
        self.distance_threshold = distance_threshold
        self.max_paragraphs_per_chunk = max_paragraphs_per_chunk

    @classmethod
    async def from_sample(
        cls,
        sample_texts: str | Sequence[str],
        embedder: EmbeddingClient | None = None,
        *,
        embed: AsyncEmbeddingBatchFn | None = None,
        percentile: float = 0.85,
        margin: float = 0.05,
        fallback_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
        max_paragraphs_per_chunk: int = 48,
    ) -> "SemanticChunker":
        """Build a chunker with a distance threshold calibrated from sample text."""
        threshold = await calibrate_distance_threshold(
            sample_texts,
            embedder=embedder,
            embed=embed,
            percentile=percentile,
            margin=margin,
            fallback_threshold=fallback_threshold,
        )
        return cls(
            embedder=embedder,
            embed=embed,
            distance_threshold=threshold,
            max_paragraphs_per_chunk=max_paragraphs_per_chunk,
        )

    async def chunk(
        self,
        text: str,
        *,
        page: int | None = None,
        source: str = "",
        chunk_index_start: int = 0,
    ) -> list[DocumentChunk]:
        """Split text into semantic chunks with page metadata and content hash."""
        if chunk_index_start < 0:
            raise ValueError("chunk_index_start must be non-negative.")

        paragraphs = split_paragraphs(text)
        if not paragraphs:
            return []

        if len(paragraphs) == 1:
            return [
                self._make_chunk(
                    paragraphs[0],
                    source=source,
                    page=page,
                    chunk_index=chunk_index_start,
                )
            ]

        embeddings = await self._get_embeddings(paragraphs)
        self._validate_embeddings(paragraphs, embeddings)

        ranges = self._semantic_ranges(paragraphs, embeddings)
        chunks: list[DocumentChunk] = []
        for start, end in ranges:
            chunk_text = "\n\n".join(paragraphs[start:end])
            chunks.append(
                self._make_chunk(
                    chunk_text,
                    source=source,
                    page=page,
                    chunk_index=chunk_index_start + len(chunks),
                )
            )
        return chunks

    async def chunk_text(
        self,
        text: str,
        *,
        page: int | None = None,
        source: str = "",
        chunk_index_start: int = 0,
    ) -> list[DocumentChunk]:
        """Alias for callers that prefer an explicit method name."""
        return await self.chunk(
            text,
            page=page,
            source=source,
            chunk_index_start=chunk_index_start,
        )

    async def _get_embeddings(self, texts: list[str]) -> list[list[float]]:
        return await _get_embeddings(texts, embedder=self._embedder, embed=self._embed)

    def _semantic_ranges(
        self,
        paragraphs: Sequence[str],
        embeddings: Sequence[Sequence[float]],
    ) -> list[tuple[int, int]]:
        ranges: list[tuple[int, int]] = []
        start = 0

        for idx in range(len(paragraphs) - 1):
            current_len = idx + 1 - start
            distance = _cosine_distance(embeddings[idx], embeddings[idx + 1])
            context_shift = distance >= self.distance_threshold
            reached_limit = current_len >= self.max_paragraphs_per_chunk

            if context_shift or reached_limit:
                ranges.append((start, idx + 1))
                start = idx + 1

        ranges.append((start, len(paragraphs)))
        return [item for item in ranges if item[0] < item[1]]

    @staticmethod
    def _validate_embeddings(
        paragraphs: Sequence[str],
        embeddings: Sequence[Sequence[float]],
    ) -> None:
        if len(embeddings) != len(paragraphs):
            raise ValueError(
                "Embedding provider returned "
                f"{len(embeddings)} vectors for {len(paragraphs)} paragraphs."
            )
        if any(len(vector) == 0 for vector in embeddings):
            raise ValueError("Embedding provider returned an empty vector.")

    @staticmethod
    def _make_chunk(
        text: str,
        *,
        source: str,
        page: int | None,
        chunk_index: int,
    ) -> DocumentChunk:
        return DocumentChunk(
            text=text,
            metadata=ChunkMetadata(source=source, page=page, chunk_index=chunk_index),
            content_hash=content_sha256(text),
        )


def _cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Return cosine distance in the [0, 2] range for normalized and raw vectors."""
    return 1.0 - cosine_similarity(a, b)


async def calibrate_distance_threshold(
    sample_texts: str | Sequence[str],
    embedder: EmbeddingClient | None = None,
    *,
    embed: AsyncEmbeddingBatchFn | None = None,
    percentile: float = 0.85,
    margin: float = 0.05,
    fallback_threshold: float = DEFAULT_DISTANCE_THRESHOLD,
) -> float:
    """Estimate a semantic distance threshold from adjacent sample paragraphs.

    The returned threshold is `percentile(adjacent_distances) + margin`, clamped
    to cosine-distance bounds. Use a representative sample for the target
    embedding model and corpus; for bge-m3, same-topic paragraph distances near
    0.45 typically need a threshold above that value.
    """
    _validate_embedding_provider(embedder=embedder, embed=embed)
    if not 0.0 <= percentile <= 1.0:
        raise ValueError("percentile must be between 0.0 and 1.0.")
    if margin < 0.0:
        raise ValueError("margin must be non-negative.")
    if not 0.0 <= fallback_threshold <= 2.0:
        raise ValueError("fallback_threshold must be between 0.0 and 2.0.")

    distances: list[float] = []
    for paragraphs in _paragraph_samples(sample_texts):
        if len(paragraphs) < 2:
            continue
        embeddings = await _get_embeddings(paragraphs, embedder=embedder, embed=embed)
        SemanticChunker._validate_embeddings(paragraphs, embeddings)
        distances.extend(
            _cosine_distance(embeddings[idx], embeddings[idx + 1])
            for idx in range(len(paragraphs) - 1)
        )

    if not distances:
        return fallback_threshold

    return _clamp_distance(_percentile(distances, percentile) + margin)


def _paragraph_samples(sample_texts: str | Sequence[str]) -> list[list[str]]:
    texts = [sample_texts] if isinstance(sample_texts, str) else list(sample_texts)
    return [paragraphs for text in texts if (paragraphs := split_paragraphs(text))]


async def _get_embeddings(
    texts: list[str],
    *,
    embedder: EmbeddingClient | None,
    embed: AsyncEmbeddingBatchFn | None,
) -> list[list[float]]:
    if embed is not None:
        return await embed(texts)
    if embedder is None:
        raise RuntimeError("No embedding provider configured.")
    return await embedder.get_embeddings(texts)


def _validate_embedding_provider(
    *,
    embedder: EmbeddingClient | None,
    embed: AsyncEmbeddingBatchFn | None,
) -> None:
    if embedder is None and embed is None:
        raise ValueError("SemanticChunker requires an embedder or embed callable.")
    if embedder is not None and embed is not None:
        raise ValueError("Pass either embedder or embed, not both.")


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    rank = percentile * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _clamp_distance(value: float) -> float:
    return max(0.0, min(2.0, value))
