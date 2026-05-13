from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Sequence
from typing import Any
from urllib.request import Request, urlopen
import json

import pytest
from sqlalchemy import create_engine, text

from py_rag_engine.storage import (
    EMBEDDING_MODEL_DIMENSIONS,
    EmbeddingInput,
    PostgresEmbeddingStore,
    embedding_dimensions_for_model,
)


TEST_POSTGRES_URL = os.getenv("TEST_POSTGRES_URL")
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL")
LM_STUDIO_EMBEDDING_MODEL = os.getenv(
    "LM_STUDIO_EMBEDDING_MODEL",
    "bge-m3",
)
STORAGE_EMBEDDING_MODEL = os.getenv("STORAGE_EMBEDDING_MODEL", LM_STUDIO_EMBEDDING_MODEL)


pytestmark = pytest.mark.skipif(
    not TEST_POSTGRES_URL or not LM_STUDIO_BASE_URL,
    reason="requires TEST_POSTGRES_URL and LM_STUDIO_BASE_URL",
)


def _lm_studio_embeddings(texts: Sequence[str]) -> list[list[float]]:
    payload = json.dumps(
        {
            "model": LM_STUDIO_EMBEDDING_MODEL,
            "input": list(texts),
        }
    ).encode("utf-8")
    request = Request(
        f"{LM_STUDIO_BASE_URL.rstrip('/')}/v1/embeddings",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    last_error: Exception | None = None
    for _ in range(3):
        try:
            with urlopen(request, timeout=60) as response:
                body: dict[str, Any] = json.loads(response.read().decode("utf-8"))
            return [list(map(float, item["embedding"])) for item in body["data"]]
        except OSError as exc:
            last_error = exc
            time.sleep(1)
    raise RuntimeError("LM Studio embeddings endpoint did not return a stable response") from last_error


def _content_hash(text_value: str) -> str:
    return hashlib.sha256(text_value.encode("utf-8")).hexdigest()


def test_lm_studio_embeddings_round_trip_through_postgres_pgvector() -> None:
    if STORAGE_EMBEDDING_MODEL not in EMBEDDING_MODEL_DIMENSIONS:
        pytest.fail(
            "STORAGE_EMBEDDING_MODEL must be one of: "
            f"{', '.join(sorted(EMBEDDING_MODEL_DIMENSIONS))}"
        )

    engine = create_engine(TEST_POSTGRES_URL)
    texts = [
        "PostgreSQL with pgvector stores embeddings for similarity search.",
        "A semantic retrieval system ranks chunks by cosine similarity.",
        "Bananas and strawberries are common fruit.",
    ]
    vectors = _lm_studio_embeddings(texts)
    dimensions = embedding_dimensions_for_model(STORAGE_EMBEDDING_MODEL)
    assert len(vectors[0]) == dimensions

    store = PostgresEmbeddingStore(
        engine,
        embedding_model=STORAGE_EMBEDDING_MODEL,
    )

    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS embeddings"))
    store.create_schema()

    inserted_ids = store.add_embeddings(
        [
            EmbeddingInput(
                text=text_value,
                embedding=vector,
                content_hash=_content_hash(text_value),
                metadata={"source": "integration-test", "chunk_index": index},
                embedding_model=STORAGE_EMBEDDING_MODEL,
            )
            for index, (text_value, vector) in enumerate(zip(texts, vectors, strict=True))
        ]
    )

    query_vector = _lm_studio_embeddings(["How do I search vectors in PostgreSQL?"])[0]
    results = store.similarity_search(
        query_vector,
        top_k=2,
        metadata_filter={"source": "integration-test"},
        ef_search=80,
    )

    with engine.connect() as conn:
        indexes = conn.execute(
            text(
                """
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE tablename = 'embeddings'
                ORDER BY indexname
                """
            )
        ).mappings().all()

    assert len(inserted_ids) == 3
    assert dimensions == len(query_vector)
    assert results
    assert results[0].metadata["source"] == "integration-test"
    assert any("USING hnsw" in row["indexdef"] for row in indexes)
    assert any("vector_cosine_ops" in row["indexdef"] for row in indexes)
