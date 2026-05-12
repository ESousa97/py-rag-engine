from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from internal.ingestion.hashing import content_sha256
from internal.ingestion.loaders import LoadedPage, load_markdown, load_pdf
from internal.ingestion.models import ChunkMetadata, DocumentChunk
from internal.ingestion.semantic import EmbeddingBatchFn, semantic_paragraph_chunking
from internal.ingestion.splitters import split_text_recursive

_SUFFIX_LOADERS: dict[str, Callable[[Path], list[LoadedPage]]] = {
    ".pdf": load_pdf,
    ".md": load_markdown,
    ".markdown": load_markdown,
}


def _load_pages(path: Path) -> list[LoadedPage]:
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
    """
    Load a PDF or Markdown file and return :class:`DocumentChunk` records.

    Pipeline: load pages → optional semantic paragraph grouping → recursive character
    splitting with dynamic overlap → SHA-256 per chunk.
    """
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
    path: Path,
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
    if use_semantic_chunking and embed is None:
        raise ValueError("use_semantic_chunking=True requires an embed function.")

    source = str(path.resolve())
    pages = _load_pages(path)
    out: list[DocumentChunk] = []
    seen_hashes: set[str] = set()
    chunk_index = 0

    for page in pages:
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
                if deduplicate_by_hash and h in seen_hashes:
                    continue
                if deduplicate_by_hash:
                    seen_hashes.add(h)
                meta = ChunkMetadata(
                    source=source,
                    page=page.page,
                    chunk_index=chunk_index,
                )
                out.append(DocumentChunk(text=piece, metadata=meta, content_hash=h))
                chunk_index += 1

    return out


def deduplicate_chunks(chunks: Sequence[DocumentChunk]) -> list[DocumentChunk]:
    """Keep first occurrence per ``content_hash`` (order preserved)."""
    seen: set[str] = set()
    result: list[DocumentChunk] = []
    for c in chunks:
        if c.content_hash in seen:
            continue
        seen.add(c.content_hash)
        result.append(c)
    return result


def make_sentence_transformer_embed(model_name: str) -> EmbeddingBatchFn:
    """
    Optional helper when ``sentence-transformers`` is installed::

        pip install 'py-rag-engine[embeddings]'
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:  # pragma: no cover - import guarded
        raise ImportError(
            "sentence-transformers is required for make_sentence_transformer_embed; "
            "install with: pip install 'py-rag-engine[embeddings]'"
        ) from e

    model = SentenceTransformer(model_name)

    def _embed(batch: list[str]) -> list[list[float]]:
        vectors = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
        return [list(map(float, row)) for row in vectors]

    return _embed
