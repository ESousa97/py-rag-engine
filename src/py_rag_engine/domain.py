from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ChunkMetadata:
    """Per-chunk provenance for retrieval and auditing."""

    source: str
    page: int | None
    chunk_index: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """A single text segment ready for embedding / vector store."""

    text: str
    metadata: ChunkMetadata
    content_hash: str
