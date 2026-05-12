from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import TypeAlias

import numpy as np

EmbeddingBatchFn: TypeAlias = Callable[[list[str]], Sequence[Sequence[float]]]
"""Maps a batch of strings to embedding vectors (any sequence of floats per row)."""


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def split_paragraphs(text: str) -> list[str]:
    """Split on blank lines; single newlines inside a paragraph are preserved."""
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]


def semantic_paragraph_chunking(
    text: str,
    embed: EmbeddingBatchFn,
    *,
    similarity_threshold: float = 0.55,
    max_paragraphs_per_chunk: int = 48,
    embed_batch_size: int = 32,
) -> list[str]:
    """
    Group paragraphs into segments where **low** consecutive embedding similarity
    indicates a topic shift (semantic boundary).

    For each pair of adjacent paragraphs, cosine similarity of their embeddings is
    computed. If similarity falls below ``similarity_threshold``, a new chunk starts
    at the following paragraph.

    Long homogeneous runs are further split every ``max_paragraphs_per_chunk``
    paragraphs so downstream recursive splitting stays bounded.
    """
    paragraphs = split_paragraphs(text)
    if not paragraphs:
        return []
    if len(paragraphs) == 1:
        return [paragraphs[0]]

    embeddings: list[np.ndarray] = []
    for start in range(0, len(paragraphs), embed_batch_size):
        batch = paragraphs[start : start + embed_batch_size]
        raw = embed(batch)
        for row in raw:
            vec = np.asarray(row, dtype=np.float64)
            embeddings.append(vec)

    topic_starts: list[int] = [0]
    for i in range(len(embeddings) - 1):
        sim = _cosine_similarity(embeddings[i], embeddings[i + 1])
        if sim < similarity_threshold:
            topic_starts.append(i + 1)
    topic_starts.append(len(paragraphs))

    ranges: list[tuple[int, int]] = []
    for lo, hi in zip(topic_starts[:-1], topic_starts[1:], strict=True):
        cursor = lo
        while cursor < hi:
            nxt = min(cursor + max_paragraphs_per_chunk, hi)
            ranges.append((cursor, nxt))
            cursor = nxt

    return ["\n\n".join(paragraphs[s:e]) for s, e in ranges if s < e]

