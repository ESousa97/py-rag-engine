from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from py_rag_engine.chunking import EmbeddingBatchFn, semantic_paragraph_chunking, split_text_recursive
from py_rag_engine.domain import ChunkMetadata, DocumentChunk
from py_rag_engine.embeddings.hashing import content_sha256
from py_rag_engine.ingestion.loaders import LoadedPage, load_markdown, load_pdf

_SUFFIX_LOADERS: dict[str, Callable[[Path], list[LoadedPage]]] = {
    ".pdf": load_pdf,
    ".md": load_markdown,
    ".markdown": load_markdown,
}


def _load_pages(path: Path) -> list[LoadedPage]:
    """Load a document using the registered loader for its file suffix."""
    suffix = path.suffix.lower()
    loader = _SUFFIX_LOADERS.get(suffix)
    if loader is None:
        raise ValueError(f"Unsupported document type: {suffix!r} (path={path})")
    return loader(path)


def _segments_for_page(
    text: str,
    *,
    use_semantic: bool,
    embed: EmbeddingBatchFn | None,
    semantic_threshold: float,
    max_paragraphs_per_chunk: int,
) -> list[str]:
    """Return semantic page segments when enabled, otherwise the original page text."""
    if use_semantic and embed is not None:
        return semantic_paragraph_chunking(
            text,
            embed,
            similarity_threshold=semantic_threshold,
            max_paragraphs_per_chunk=max_paragraphs_per_chunk,
        )
    return [text] if text.strip() else []


def ingest_path(
    path: str | Path,
    *,
    chunk_size: int = 1200,
    chunk_overlap: int | None = None,
    overlap_ratio: float = 0.12,
    min_overlap: int = 32,
    max_overlap: int = 512,
    use_semantic_chunking: bool = False,
    embed: EmbeddingBatchFn | None = None,
    semantic_similarity_threshold: float = 0.55,
    semantic_max_paragraphs_per_chunk: int = 48,
    deduplicate_by_hash: bool = True,
) -> list[DocumentChunk]:
    """Resolve a user-supplied path and ingest the target document."""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(p)
    return ingest_file(
        p,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        overlap_ratio=overlap_ratio,
        min_overlap=min_overlap,
        max_overlap=max_overlap,
        use_semantic_chunking=use_semantic_chunking,
        embed=embed,
        semantic_similarity_threshold=semantic_similarity_threshold,
        semantic_max_paragraphs_per_chunk=semantic_max_paragraphs_per_chunk,
        deduplicate_by_hash=deduplicate_by_hash,
    )


def ingest_file(
    path: str | Path,
    *,
    chunk_size: int = 1200,
    chunk_overlap: int | None = None,
    overlap_ratio: float = 0.12,
    min_overlap: int = 32,
    max_overlap: int = 512,
    use_semantic_chunking: bool = False,
    embed: EmbeddingBatchFn | None = None,
    semantic_similarity_threshold: float = 0.55,
    semantic_max_paragraphs_per_chunk: int = 48,
    deduplicate_by_hash: bool = True,
) -> list[DocumentChunk]:
    """Ingest a PDF or Markdown file into deduplicated document chunks."""
    if use_semantic_chunking and embed is None:
        raise ValueError("use_semantic_chunking=True requires an embed function.")

    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(p)

    source = str(p)
    pages = _load_pages(p)
    out: list[DocumentChunk] = []
    seen_hashes: set[str] = set()
    chunk_index = 0

    for page in pages:
        # Semantic segmentation happens before recursive splitting so topic
        # boundaries are preserved when the final chunks are created.
        segments = _segments_for_page(
            page.text,
            use_semantic=use_semantic_chunking,
            embed=embed,
            semantic_threshold=semantic_similarity_threshold,
            max_paragraphs_per_chunk=semantic_max_paragraphs_per_chunk,
        )
        for segment in segments:
            pieces = split_text_recursive(
                segment,
                chunk_size,
                chunk_overlap,
                overlap_ratio=overlap_ratio,
                min_overlap=min_overlap,
                max_overlap=max_overlap,
            )
            for piece in pieces:
                h = content_sha256(piece)
                # Hash-based deduplication prevents repeated extracted text from
                # being embedded and indexed more than once.
                if deduplicate_by_hash and h in seen_hashes:
                    continue
                if deduplicate_by_hash:
                    seen_hashes.add(h)
                meta = ChunkMetadata(source=source, page=page.page, chunk_index=chunk_index)
                out.append(DocumentChunk(text=piece, metadata=meta, content_hash=h))
                chunk_index += 1

    return out


def deduplicate_chunks(chunks: Sequence[DocumentChunk]) -> list[DocumentChunk]:
    """Remove duplicated chunks while preserving first-seen order."""
    seen: set[str] = set()
    result: list[DocumentChunk] = []
    for c in chunks:
        if c.content_hash in seen:
            continue
        seen.add(c.content_hash)
        result.append(c)
    return result


def chunks_to_dicts(chunks: Sequence[DocumentChunk]) -> list[dict[str, object]]:
    """Convert domain chunks into JSON-ready dictionaries."""
    return [chunk.to_dict() for chunk in chunks]


