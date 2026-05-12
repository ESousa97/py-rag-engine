from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import TypeAlias

import numpy as np

from py_rag_engine.vector_math import cosine_similarity

EmbeddingBatchFn: TypeAlias = Callable[[list[str]], Sequence[Sequence[float]]]


def split_paragraphs(text: str) -> list[str]:
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
            embeddings.append(np.asarray(row, dtype=np.float64))

    topic_starts: list[int] = [0]
    for i in range(len(embeddings) - 1):
        if cosine_similarity(embeddings[i], embeddings[i + 1]) < similarity_threshold:
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
