"""Practical end-to-end test of the PostgreSQL + pgvector persistence layer.

Inserts deterministic embeddings, runs a cosine-similarity query, and inspects
the resulting indexes to confirm HNSW + vector_cosine_ops are in place.

Usage:
    set TEST_POSTGRES_URL=postgresql+psycopg://rag:ragtest@localhost:55433/ragdb
    python scripts/practical_test_pgvector.py
"""
from __future__ import annotations

import hashlib
import math
import os
import random
import sys

from sqlalchemy import create_engine, text

from py_rag_engine.storage import (
    EmbeddingInput,
    PostgresEmbeddingStore,
    embedding_dimensions_for_model,
)

POSTGRES_URL = os.environ["TEST_POSTGRES_URL"]
MODEL = "bge-m3"
DIM = embedding_dimensions_for_model(MODEL)


def unit_vector(seed: int) -> list[float]:
    """Deterministic unit-norm vector — gives stable cosine similarities."""
    rng = random.Random(seed)
    raw = [rng.gauss(0.0, 1.0) for _ in range(DIM)]
    norm = math.sqrt(sum(v * v for v in raw))
    return [v / norm for v in raw]


def near(base: list[float], jitter_seed: int, sigma: float) -> list[float]:
    """Return a vector near `base` by adding small Gaussian noise, then renorm."""
    rng = random.Random(jitter_seed)
    perturbed = [b + rng.gauss(0.0, sigma) for b in base]
    norm = math.sqrt(sum(v * v for v in perturbed))
    return [v / norm for v in perturbed]


def content_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> int:
    engine = create_engine(POSTGRES_URL, future=True)

    # Fresh state — drop any leftover table from a prior run.
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS embeddings"))

    store = PostgresEmbeddingStore(engine, embedding_model=MODEL)
    store.create_schema()
    print(f"[1/5] Schema created (model={MODEL}, dim={DIM})")

    target = unit_vector(seed=42)
    items = [
        EmbeddingInput(
            text="PostgreSQL com pgvector armazena embeddings para busca semantica.",
            embedding=near(target, jitter_seed=1, sigma=0.02),
            content_hash=content_hash("doc-A"),
            metadata={"source": "manual", "topic": "pgvector", "lang": "pt"},
            embedding_model=MODEL,
        ),
        EmbeddingInput(
            text="A busca por similaridade de cosseno classifica chunks por relevancia.",
            embedding=near(target, jitter_seed=2, sigma=0.05),
            content_hash=content_hash("doc-B"),
            metadata={"source": "manual", "topic": "retrieval", "lang": "pt"},
            embedding_model=MODEL,
        ),
        EmbeddingInput(
            text="Bananas, morangos e maracujas sao frutas tropicais comuns.",
            embedding=unit_vector(seed=999),  # unrelated
            content_hash=content_hash("doc-C"),
            metadata={"source": "manual", "topic": "fruit", "lang": "pt"},
            embedding_model=MODEL,
        ),
        EmbeddingInput(
            text="Index HNSW oferece busca aproximada de vizinhos em alta performance.",
            embedding=near(target, jitter_seed=3, sigma=0.08),
            content_hash=content_hash("doc-D"),
            metadata={"source": "blog", "topic": "pgvector", "lang": "pt"},
            embedding_model=MODEL,
        ),
    ]
    ids = store.add_embeddings(items)
    print(f"[2/5] Inserted {len(ids)} embeddings, ids={ids}")

    # Test 1 — pure cosine ranking. Closest must be doc-A (smallest jitter).
    results = store.similarity_search(target, top_k=4, ef_search=80)
    print("[3/5] Cosine ranking (top_k=4):")
    for rank, r in enumerate(results, start=1):
        print(f"      #{rank}  sim={r.cosine_similarity:.4f}  topic={r.metadata.get('topic')}  text={r.text[:60]!r}")

    assert results[0].metadata["topic"] == "pgvector", "expected doc-A (closest) to win"
    assert results[-1].metadata["topic"] == "fruit", "unrelated fruit doc should rank last"
    for prev, curr in zip(results, results[1:]):
        assert prev.cosine_similarity >= curr.cosine_similarity, "results must be sorted desc"

    # Test 2 — JSONB metadata filter.
    filtered = store.similarity_search(
        target, top_k=5, metadata_filter={"source": "manual"}
    )
    sources = {r.metadata["source"] for r in filtered}
    print(f"[4/5] Metadata-filtered ranking sources={sources} (expected only 'manual')")
    assert sources == {"manual"}, f"metadata filter leaked: {sources}"

    # Test 3 — inspect that the HNSW + cosine index actually exists.
    with engine.connect() as conn:
        index_rows = conn.execute(
            text(
                "SELECT indexname, indexdef FROM pg_indexes "
                "WHERE tablename = 'embeddings' ORDER BY indexname"
            )
        ).mappings().all()
    print("[5/5] Indexes on embeddings:")
    for row in index_rows:
        print(f"      - {row['indexname']}: {row['indexdef']}")

    hnsw_defs = [r["indexdef"] for r in index_rows if "USING hnsw" in r["indexdef"]]
    assert hnsw_defs, "no HNSW index found on embeddings table"
    assert "vector_cosine_ops" in hnsw_defs[0], "HNSW index is not using cosine ops"
    assert any("USING gin" in r["indexdef"] for r in index_rows), "missing GIN index on JSONB metadata"

    # Test 4 — upsert idempotency (same content_hash should overwrite, not duplicate).
    store.add_embeddings([items[0]])
    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM embeddings")).scalar_one()
    assert total == 4, f"expected upsert to keep 4 rows, got {total}"
    print(f"      upsert idempotency OK (rows={total})")

    print("\nAll practical checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
