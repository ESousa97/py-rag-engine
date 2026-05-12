"""Document ingestion: PDF/Markdown loading, semantic and recursive chunking."""

from internal.ingestion.models import ChunkMetadata, DocumentChunk
from internal.ingestion.pipeline import (
    deduplicate_chunks,
    ingest_file,
    ingest_path,
    make_sentence_transformer_embed,
)
from internal.ingestion.semantic import semantic_paragraph_chunking
from internal.ingestion.splitters import dynamic_chunk_overlap, make_recursive_splitter

__all__ = [
    "ChunkMetadata",
    "DocumentChunk",
    "deduplicate_chunks",
    "dynamic_chunk_overlap",
    "ingest_file",
    "ingest_path",
    "make_recursive_splitter",
    "make_sentence_transformer_embed",
    "semantic_paragraph_chunking",
]
