from __future__ import annotations

from py_rag_engine.chunking.recursive import dynamic_chunk_overlap, split_text_recursive
from py_rag_engine.chunking.semantic import semantic_paragraph_chunking


def test_dynamic_overlap_clamped() -> None:
    assert dynamic_chunk_overlap(500, ratio=0.12, min_overlap=32, max_overlap=512) == 60
    assert dynamic_chunk_overlap(100, ratio=0.12, min_overlap=32, max_overlap=512) == 32
    assert dynamic_chunk_overlap(10000, ratio=0.12, min_overlap=32, max_overlap=512) == 512


def test_split_text_recursive() -> None:
    text = "para1\n\n" + ("word " * 400)
    parts = split_text_recursive(text, chunk_size=200, chunk_overlap=None)
    assert len(parts) >= 2
    assert all(len(p) <= 250 for p in parts)


def test_semantic_paragraph_chunking_topic_split() -> None:
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
