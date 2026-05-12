from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from internal.ingestion.hashing import content_sha256, normalize_for_hash
from internal.ingestion.loaders import load_markdown
from internal.ingestion.pipeline import ingest_file
from internal.ingestion.semantic import semantic_paragraph_chunking
from internal.ingestion.splitters import dynamic_chunk_overlap, split_text_recursive


def test_dynamic_overlap_clamped() -> None:
    assert dynamic_chunk_overlap(500, ratio=0.12, min_overlap=32, max_overlap=512) == 60
    assert dynamic_chunk_overlap(100, ratio=0.12, min_overlap=32, max_overlap=512) == 32
    assert dynamic_chunk_overlap(10000, ratio=0.12, min_overlap=32, max_overlap=512) == 512


def test_content_hash_stable() -> None:
    a = content_sha256("  hello\n\nworld  ")
    b = content_sha256("hello world")
    assert a == b


def test_normalize_for_hash() -> None:
    assert normalize_for_hash("a\n\nb") == "a b"


def test_split_text_recursive() -> None:
    text = "para1\n\n" + ("word " * 400)
    parts = split_text_recursive(text, chunk_size=200, chunk_overlap=None)
    assert len(parts) >= 2
    assert all(len(p) <= 250 for p in parts)


def test_semantic_paragraph_chunking_topic_split() -> None:
    """Orthogonal pseudo-embeddings: low sim forces a boundary."""

    def embed(batch: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in batch:
            if "alpha" in t:
                out.append([1.0, 0.0])
            elif "beta" in t:
                out.append([0.0, 1.0])
            else:
                out.append([1.0, 0.0])
        return out

    text = "alpha topic here.\n\nbeta completely different topic.\n\nalpha again."
    chunks = semantic_paragraph_chunking(text, embed, similarity_threshold=0.9)
    assert len(chunks) >= 2


def test_ingest_markdown(tmp_path: Path) -> None:
    p = tmp_path / "doc.md"
    p.write_text("# Title\n\n" + ("line " * 300), encoding="utf-8")
    chunks = ingest_file(p, chunk_size=120, deduplicate_by_hash=False)
    assert chunks
    assert chunks[0].metadata.source == str(p.resolve())
    assert chunks[0].metadata.page is None
    assert len(chunks[0].content_hash) == 64


def test_ingest_markdown_semantic_dummy_embed(tmp_path: Path) -> None:
    p = tmp_path / "s.md"
    p.write_text("a\n\nb\n\nc", encoding="utf-8")

    def embed(batch: list[str]) -> list[list[float]]:
        return [list(map(float, np.random.default_rng(abs(hash(s)) % 2**32).normal(size=4))) for s in batch]

    chunks = ingest_file(
        p,
        chunk_size=50,
        use_semantic_chunking=True,
        embed=embed,
        deduplicate_by_hash=False,
    )
    assert chunks
