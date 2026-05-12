from __future__ import annotations

from langchain_text_splitters import RecursiveCharacterTextSplitter


def dynamic_chunk_overlap(
    chunk_size: int,
    *,
    ratio: float = 0.12,
    min_overlap: int = 32,
    max_overlap: int = 512,
) -> int:
    """Compute overlap from chunk size while keeping it within practical bounds."""
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
    """Build the recursive splitter used by the ingestion pipeline."""
    overlap = (
        chunk_overlap
        if chunk_overlap is not None
        else dynamic_chunk_overlap(
            chunk_size, ratio=overlap_ratio, min_overlap=min_overlap, max_overlap=max_overlap
        )
    )
    # Prefer natural document boundaries before falling back to smaller separators.
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
    """Split text into non-empty recursive chunks."""
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
