from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter


def dynamic_chunk_overlap(
    chunk_size: int,
    *,
    ratio: float = 0.12,
    min_overlap: int = 32,
    max_overlap: int = 512,
) -> int:
    """
    Overlap scales with ``chunk_size`` (ratio), clamped to ``[min_overlap, max_overlap]``.

    Longer chunks carry more context in the overlap window without dominating small chunks.
    """
    overlap = int(round(chunk_size * ratio))
    return max(min_overlap, min(max_overlap, overlap))


def make_recursive_splitter(
    chunk_size: int,
    chunk_overlap: int | None = None,
    *,
    overlap_ratio: float = 0.12,
    min_overlap: int = 32,
    max_overlap: int = 512,
    separators: list[str] | None = None,
) -> RecursiveCharacterTextSplitter:
    """
    Build a ``RecursiveCharacterTextSplitter`` with optional dynamic overlap.

    If ``chunk_overlap`` is ``None``, overlap is computed via :func:`dynamic_chunk_overlap`.
    """
    overlap = (
        chunk_overlap
        if chunk_overlap is not None
        else dynamic_chunk_overlap(
            chunk_size, ratio=overlap_ratio, min_overlap=min_overlap, max_overlap=max_overlap
        )
    )
    seps = separators if separators is not None else ["\n\n", "\n", ". ", " ", ""]
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=seps,
        length_function=len,
        is_separator_regex=False,
    )


def split_text_recursive(
    text: str,
    chunk_size: int,
    chunk_overlap: int | None = None,
    *,
    overlap_ratio: float = 0.12,
    min_overlap: int = 32,
    max_overlap: int = 512,
    separators: list[str] | None = None,
) -> list[str]:
    splitter = make_recursive_splitter(
        chunk_size,
        chunk_overlap,
        overlap_ratio=overlap_ratio,
        min_overlap=min_overlap,
        max_overlap=max_overlap,
        separators=separators,
    )
    parts = splitter.split_text(text)
    return [p for p in parts if p.strip()]
