from __future__ import annotations

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.schema import CreateIndex, CreateTable

from py_rag_engine.storage.postgres import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingInput,
    PostgresEmbeddingStore,
    define_embeddings_table,
    embedding_dimensions_for_model,
)


def test_embedding_dimensions_for_supported_models() -> None:
    assert DEFAULT_EMBEDDING_MODEL == "openai-3-small"
    assert embedding_dimensions_for_model("text-embedding-3-small") == 1536
    assert embedding_dimensions_for_model("openai-3-small") == 1536
    assert embedding_dimensions_for_model("bge-m3") == 1024


def test_embedding_dimensions_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="Unsupported embedding model"):
        embedding_dimensions_for_model("unknown")


def test_embeddings_table_uses_pgvector_jsonb_and_hnsw_cosine_index() -> None:
    table = define_embeddings_table(dimensions=3)
    table_sql = str(CreateTable(table).compile(dialect=postgresql.dialect()))
    index_sql = [
        str(CreateIndex(index).compile(dialect=postgresql.dialect()))
        for index in sorted(table.indexes, key=lambda item: item.name)
    ]

    assert "embedding VECTOR(3) NOT NULL" in table_sql
    assert "content_hash VARCHAR(64) NOT NULL" in table_sql
    assert "metadata JSONB DEFAULT '{}'::jsonb NOT NULL" in table_sql
    assert "CONSTRAINT uq_embeddings_model_hash UNIQUE" in table_sql
    assert (
        "CREATE INDEX ix_embeddings_embedding_hnsw_cosine "
        "ON embeddings USING hnsw (embedding vector_cosine_ops)"
    ) in index_sql[0]
    assert "WITH (m = 16, ef_construction = 64)" in index_sql[0]
    assert index_sql[1] == "CREATE INDEX ix_embeddings_metadata_gin ON embeddings USING gin (metadata)"


def test_store_validates_embedding_dimensions() -> None:
    store = PostgresEmbeddingStore(engine=object())  # type: ignore[arg-type]
    item = EmbeddingInput(text="x", embedding=[1.0, 2.0], content_hash="h")

    with pytest.raises(ValueError, match="Expected embedding with 1536 dimensions"):
        store._row_from_input(item)


def test_store_rejects_dimensions_that_do_not_match_model() -> None:
    with pytest.raises(ValueError, match="requires 1536 dimensions"):
        PostgresEmbeddingStore(engine=object(), dimensions=768)  # type: ignore[arg-type]


def test_store_validates_ef_search() -> None:
    store = PostgresEmbeddingStore(engine=object())  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="ef_search must be greater than zero"):
        store.similarity_search([0.0] * 1536, ef_search=0)
