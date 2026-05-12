from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ChunkMetadata:
    """Per-chunk provenance for retrieval and auditing."""

    source: str
    """Filesystem path or URI of the original document."""
    page: int | None
    """1-based page index for PDFs; ``None`` for Markdown / non-paginated sources."""
    chunk_index: int = 0
    """Order of the chunk within the ingestion run."""
    extra: dict[str, Any] = field(default_factory=dict)
    """Optional additional fields (e.g. heading, mime type)."""


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """A single text segment ready for embedding / vector store."""

    text: str
    metadata: ChunkMetadata
    content_hash: str
    """SHA-256 (hex) of normalized chunk text; use for deduplication."""
