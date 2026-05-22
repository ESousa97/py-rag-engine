from __future__ import annotations

import asyncio
import math

from py_rag_engine.chunker import DEFAULT_DISTANCE_THRESHOLD, SemanticChunker
from py_rag_engine.chunker import calibrate_distance_threshold
from py_rag_engine.embeddings.hashing import content_sha256


def test_semantic_chunker_groups_by_adjacent_cosine_distance() -> None:
    async def embed(texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            if "beta" in text:
                vectors.append([0.0, 1.0])
            else:
                vectors.append([1.0, 0.0])
        return vectors

    chunker = SemanticChunker(embed=embed, distance_threshold=0.9)

    chunks = asyncio.run(
        chunker.chunk(
            "alpha topic.\n\nalpha details.\n\nbeta topic.",
            page=7,
            source="doc.md",
        )
    )

    assert len(chunks) == 2
    assert chunks[0].text == "alpha topic.\n\nalpha details."
    assert chunks[1].text == "beta topic."
    assert chunks[0].metadata.page == 7
    assert chunks[0].metadata.source == "doc.md"
    assert chunks[0].content_hash == content_sha256(chunks[0].text)


def test_semantic_chunker_rejects_embedding_count_mismatch() -> None:
    async def embed(_: list[str]) -> list[list[float]]:
        return [[1.0, 0.0]]

    chunker = SemanticChunker(embed=embed)

    try:
        asyncio.run(chunker.chunk("first.\n\nsecond."))
    except ValueError as exc:
        assert "returned 1 vectors for 2 paragraphs" in str(exc)
    else:
        raise AssertionError("Expected embedding count mismatch to raise ValueError.")


def test_calibrate_distance_threshold_from_sample() -> None:
    same_topic = [0.55, math.sqrt(1.0 - 0.55**2)]

    async def embed(texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            if "middle" in text:
                vectors.append(same_topic)
            elif "other" in text:
                vectors.append([-1.0, 0.0])
            else:
                vectors.append([1.0, 0.0])
        return vectors

    threshold = asyncio.run(
        calibrate_distance_threshold(
            "topic start.\n\nmiddle same topic.\n\ntopic continuation.\n\nother context.",
            embed=embed,
            percentile=0.5,
            margin=0.05,
        )
    )

    assert 0.49 < threshold < 0.51


def test_semantic_chunker_from_sample_uses_calibrated_threshold() -> None:
    async def embed(_: list[str]) -> list[list[float]]:
        return [[1.0, 0.0], [0.5, 0.866]]

    chunker = asyncio.run(
        SemanticChunker.from_sample(
            "first.\n\nsecond.",
            embed=embed,
            percentile=0.0,
            margin=0.1,
        )
    )

    assert 0.59 < chunker.distance_threshold < 0.61


def test_calibrate_distance_threshold_falls_back_without_adjacent_pairs() -> None:
    async def embed(_: list[str]) -> list[list[float]]:
        raise AssertionError("Embedding should not be called for a single paragraph sample.")

    threshold = asyncio.run(calibrate_distance_threshold("single paragraph", embed=embed))

    assert threshold == DEFAULT_DISTANCE_THRESHOLD
