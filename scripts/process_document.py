from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from py_rag_engine.ingestion import chunks_to_dicts, ingest_file


def _portable_source(source: str, base_dir: Path) -> str:
    """Return a repository-relative source path when the file is under the project."""
    path = Path(source)
    try:
        return str(path.resolve().relative_to(base_dir.resolve()))
    except ValueError:
        return source


def _metadata_for_use(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a lightweight metadata report from full chunk output."""
    base_dir = Path.cwd()
    return {
        "chunk_count": len(chunks),
        "unique_hash_count": len({chunk["content_hash"] for chunk in chunks}),
        "sources": sorted(
            {_portable_source(chunk["metadata"]["source"], base_dir) for chunk in chunks}
        ),
        "pages": sorted(
            {
                chunk["metadata"]["page"]
                for chunk in chunks
                if chunk["metadata"]["page"] is not None
            }
        ),
        "chunks": [
            {
                "content_hash": chunk["content_hash"],
                "source": _portable_source(chunk["metadata"]["source"], base_dir),
                "page": chunk["metadata"]["page"],
                "chunk_index": chunk["metadata"]["chunk_index"],
                "text_chars": len(chunk["text"]),
            }
            for chunk in chunks
        ],
    }


def main() -> None:
    """Process one document and write full chunks plus shareable metadata."""
    parser = argparse.ArgumentParser(description="Process a PDF or Markdown document into RAG chunks.")
    parser.add_argument("input", help="Path to a .pdf, .md, or .markdown file.")
    parser.add_argument(
        "--output",
        default="results/chunks.json",
        help="Path for full chunk output with text, metadata, and content_hash.",
    )
    parser.add_argument(
        "--metadata-output",
        default="data/processed_metadata.json",
        help="Path for metadata-only output suitable for sharing/versioning.",
    )
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument("--overlap-ratio", type=float, default=0.12)
    parser.add_argument("--no-deduplicate", action="store_true")
    args = parser.parse_args()

    chunks = chunks_to_dicts(
        ingest_file(
            args.input,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            overlap_ratio=args.overlap_ratio,
            deduplicate_by_hash=not args.no_deduplicate,
        )
    )

    output = Path(args.output)
    metadata_output = Path(args.metadata_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata_output.write_text(
        json.dumps(_metadata_for_use(chunks), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"chunks={len(chunks)}")
    print(f"output={output}")
    print(f"metadata_output={metadata_output}")


if __name__ == "__main__":
    main()
