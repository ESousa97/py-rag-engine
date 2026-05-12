from __future__ import annotations

from pathlib import Path

import numpy as np

from py_rag_engine.ingestion.pipeline import chunks_to_dicts, ingest_file


def test_ingest_markdown(tmp_path: Path) -> None:
    p = tmp_path / "doc.md"
    p.write_text("# Title\n\n" + ("line " * 300), encoding="utf-8")
    chunks = ingest_file(p, chunk_size=120, deduplicate_by_hash=False)
    assert chunks
    assert chunks[0].metadata.source == str(p.resolve())
    assert chunks[0].metadata.page is None
    assert len(chunks[0].content_hash) == 64


def test_ingest_file_accepts_string_path(tmp_path: Path) -> None:
    p = tmp_path / "doc.md"
    p.write_text("# Title\n\nbody", encoding="utf-8")
    chunks = ingest_file(str(p), chunk_size=120)
    assert chunks
    assert chunks[0].metadata.source == str(p.resolve())


def test_chunks_to_dicts_returns_serializable_contract(tmp_path: Path) -> None:
    p = tmp_path / "doc.md"
    p.write_text("# Title\n\nbody", encoding="utf-8")
    chunks = chunks_to_dicts(ingest_file(p, chunk_size=120))

    assert chunks
    assert set(chunks[0]) == {"text", "metadata", "content_hash"}
    assert chunks[0]["text"]
    assert chunks[0]["metadata"]["page"] is None
    assert chunks[0]["metadata"]["source"] == str(p.resolve())
    assert len(chunks[0]["content_hash"]) == 64


def test_ingest_markdown_semantic_dummy_embed(tmp_path: Path) -> None:
    p = tmp_path / "s.md"
    p.write_text("a\n\nb\n\nc", encoding="utf-8")

    def embed(batch: list[str]) -> list[list[float]]:
        return [list(map(float, np.random.default_rng(abs(hash(s)) % 2**32).normal(size=4))) for s in batch]

    chunks = ingest_file(
        p,
        chunk_size=50,
        use_semantic_chunking=True,
        embed=embed,
        deduplicate_by_hash=False,
    )
    assert chunks
