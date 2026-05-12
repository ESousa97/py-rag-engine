from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ChunkMetadata:
    """Per-chunk provenance used for retrieval, citations, and auditing."""

    source: str
    page: int | None
    chunk_index: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation of the metadata."""
        return {
            "source": self.source,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """A single text segment ready for embedding and vector-store indexing."""

    text: str
    metadata: ChunkMetadata
    content_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready representation of the chunk."""
        return {
            "text": self.text,
            "metadata": self.metadata.to_dict(),
            "content_hash": self.content_hash,
        }
