from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Index,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    bindparam,
    func,
    select,
    text as sql_text,
)
from sqlalchemy.dialects.postgresql import JSONB, insert as postgres_insert
from sqlalchemy.engine import Engine

DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_MODEL_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "openai-3-small": 1536,
    "bge-m3": 1024,
}


@dataclass(frozen=True, slots=True)
class EmbeddingInput:
    """Payload ready to be persisted in the embeddings table."""

    text: str
    embedding: Sequence[float]
    content_hash: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    embedding_model: str = DEFAULT_EMBEDDING_MODEL


@dataclass(frozen=True, slots=True)
class SimilaritySearchResult:
    """A row returned by vector similarity search."""

    id: int
    text: str
    metadata: dict[str, Any]
    content_hash: str | None
    embedding_model: str
    cosine_similarity: float


def embedding_dimensions_for_model(model_name: str) -> int:
    """Return the pgvector dimension count for a supported embedding model."""
    try:
        return EMBEDDING_MODEL_DIMENSIONS[model_name]
    except KeyError as exc:
        supported = ", ".join(sorted(EMBEDDING_MODEL_DIMENSIONS))
        raise ValueError(f"Unsupported embedding model '{model_name}'. Supported: {supported}") from exc


def define_embeddings_table(
    *,
    metadata: MetaData | None = None,
    dimensions: int | None = None,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    hnsw_m: int = 16,
    hnsw_ef_construction: int = 64,
) -> Table:
    """Define the PostgreSQL embeddings table with a pgvector HNSW cosine index."""
    if dimensions is None:
        dimensions = embedding_dimensions_for_model(embedding_model)

    table_metadata = metadata if metadata is not None else MetaData()
    table = Table(
        "embeddings",
        table_metadata,
        Column("id", BigInteger, primary_key=True, autoincrement=True),
        Column("embedding_model", String(64), nullable=False),
        Column("content_hash", String(64), nullable=False),
        Column("embedding", Vector(dimensions), nullable=False),
        Column("text", Text, nullable=False),
        Column(
            "metadata",
            JSONB,
            nullable=False,
            server_default=sql_text("'{}'::jsonb"),
        ),
        Column(
            "created_at",
            DateTime(timezone=True),
            nullable=False,
            server_default=func.now(),
        ),
        Column(
            "updated_at",
            DateTime(timezone=True),
            nullable=False,
            server_default=func.now(),
            onupdate=func.now(),
        ),
        UniqueConstraint("embedding_model", "content_hash", name="uq_embeddings_model_hash"),
    )

    Index(
        "ix_embeddings_embedding_hnsw_cosine",
        table.c.embedding,
        postgresql_using="hnsw",
        postgresql_with={"m": hnsw_m, "ef_construction": hnsw_ef_construction},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    Index("ix_embeddings_metadata_gin", table.c.metadata, postgresql_using="gin")
    return table


class PostgresEmbeddingStore:
    """PostgreSQL + pgvector persistence and cosine ANN search."""

    def __init__(
        self,
        engine: Engine,
        *,
        dimensions: int | None = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        metadata: MetaData | None = None,
    ) -> None:
        self.engine = engine
        self.embedding_model = embedding_model
        self.dimensions = dimensions or embedding_dimensions_for_model(embedding_model)
        self.metadata = metadata if metadata is not None else MetaData()
        self.table = define_embeddings_table(
            metadata=self.metadata,
            dimensions=self.dimensions,
            embedding_model=embedding_model,
        )

    def create_schema(self) -> None:
        """Create the vector extension, embeddings table, and HNSW indexes."""
        with self.engine.begin() as conn:
            conn.execute(sql_text("CREATE EXTENSION IF NOT EXISTS vector"))
            self.metadata.create_all(conn)

    def add_embedding(self, item: EmbeddingInput, *, upsert: bool = True) -> int:
        """Persist one embedding and return its database id."""
        ids = self.add_embeddings([item], upsert=upsert)
        return ids[0]

    def add_embeddings(self, items: Sequence[EmbeddingInput], *, upsert: bool = True) -> list[int]:
        """Persist embeddings in a single transaction."""
        if not items:
            return []

        rows = [self._row_from_input(item) for item in items]
        statement = postgres_insert(self.table).values(rows)
        if upsert:
            statement = statement.on_conflict_do_update(
                constraint="uq_embeddings_model_hash",
                set_={
                    "embedding": statement.excluded.embedding,
                    "text": statement.excluded.text,
                    "metadata": statement.excluded.metadata,
                    "updated_at": func.now(),
                },
            )
        statement = statement.returning(self.table.c.id)

        with self.engine.begin() as conn:
            return list(conn.execute(statement).scalars())

    def similarity_search(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int = 5,
        metadata_filter: Mapping[str, Any] | None = None,
        ef_search: int | None = None,
    ) -> list[SimilaritySearchResult]:
        """Return nearest chunks ordered by cosine similarity using pgvector."""
        self._validate_embedding(query_embedding)
        if top_k < 1:
            raise ValueError("top_k must be greater than zero")

        distance = self.table.c.embedding.cosine_distance(bindparam("query_embedding")).label(
            "cosine_distance"
        )
        statement = (
            select(
                self.table.c.id,
                self.table.c.text,
                self.table.c.metadata,
                self.table.c.content_hash,
                self.table.c.embedding_model,
                (1 - distance).label("cosine_similarity"),
            )
            .where(self.table.c.embedding_model == self.embedding_model)
            .order_by(distance)
            .limit(top_k)
        )
        params: dict[str, Any] = {"query_embedding": list(query_embedding)}
        if metadata_filter:
            statement = statement.where(self.table.c.metadata.contains(bindparam("metadata_filter")))
            params["metadata_filter"] = dict(metadata_filter)

        with self.engine.begin() as conn:
            if ef_search is not None:
                conn.execute(sql_text("SET LOCAL hnsw.ef_search = :ef_search"), {"ef_search": ef_search})
            rows = conn.execute(statement, params).mappings()
            return [
                SimilaritySearchResult(
                    id=row["id"],
                    text=row["text"],
                    metadata=dict(row["metadata"]),
                    content_hash=row["content_hash"],
                    embedding_model=row["embedding_model"],
                    cosine_similarity=float(row["cosine_similarity"]),
                )
                for row in rows
            ]

    def _row_from_input(self, item: EmbeddingInput) -> dict[str, Any]:
        self._validate_embedding(item.embedding)
        if item.embedding_model != self.embedding_model:
            raise ValueError(
                f"Store is configured for '{self.embedding_model}', got '{item.embedding_model}'"
            )
        return {
            "embedding_model": item.embedding_model,
            "content_hash": item.content_hash,
            "embedding": list(item.embedding),
            "text": item.text,
            "metadata": dict(item.metadata),
        }

    def _validate_embedding(self, embedding: Sequence[float]) -> None:
        if len(embedding) != self.dimensions:
            raise ValueError(
                f"Expected embedding with {self.dimensions} dimensions, got {len(embedding)}"
            )
