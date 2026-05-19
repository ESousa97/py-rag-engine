from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Column,
    Computed,
    DateTime,
    Index,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    bindparam,
    func,
    literal,
    select,
    text as sql_text,
)
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, insert as postgres_insert
from sqlalchemy.engine import Engine

DEFAULT_EMBEDDING_MODEL = "openai-3-small"
DEFAULT_FTS_LANGUAGE = "english"
_FTS_LANG_RE = re.compile(r"^[a-z_][a-z0-9_]*$", re.IGNORECASE)

EMBEDDING_MODEL_DIMENSIONS = {
    # OpenAI / LM Studio aliases
    "text-embedding-3-small": 1536,
    "openai-3-small": 1536,
    # BGE-M3 — also the default LM Studio embed model
    "bge-m3": 1024,
    "text-embedding-bge-m3": 1024,
    # MiniLM
    "all-MiniLM-L6-v2": 384,
    "all-minilm-l6-v2": 384,
    # Nomic
    "nomic-embed-text-v1": 768,
    "nomic-embed-text-v1.5": 768,
    # MixedBread
    "mxbai-embed-large-v1": 1024,
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


@dataclass(frozen=True, slots=True)
class FTSSearchResult:
    """A row returned by full-text search."""

    id: int
    text: str
    metadata: dict[str, Any]
    content_hash: str | None
    embedding_model: str
    fts_score: float


def _validate_fts_language(lang: str) -> None:
    if not _FTS_LANG_RE.match(lang):
        raise ValueError(
            f"Invalid FTS language '{lang}'. Must be a simple identifier (letters, digits, underscores)."
        )


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
    table_name: str = "embeddings",
    fts_language: str = DEFAULT_FTS_LANGUAGE,
) -> Table:
    """Define a PostgreSQL embeddings table with pgvector HNSW and FTS GIN indexes."""
    _validate_fts_language(fts_language)
    if dimensions is None:
        dimensions = embedding_dimensions_for_model(embedding_model)

    table_metadata = metadata if metadata is not None else MetaData()
    t = table_name  # short alias for name derivation
    table = Table(
        table_name,
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
            "text_search_tsv",
            TSVECTOR,
            Computed(f"to_tsvector('{fts_language}', text)", persisted=True),
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
        UniqueConstraint("embedding_model", "content_hash", name=f"uq_{t}_model_hash"),
    )

    Index(
        f"ix_{t}_embedding_hnsw_cosine",
        table.c.embedding,
        postgresql_using="hnsw",
        postgresql_with={"m": hnsw_m, "ef_construction": hnsw_ef_construction},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
    Index(f"ix_{t}_metadata_gin", table.c.metadata, postgresql_using="gin")
    Index(f"ix_{t}_text_search_gin", table.c.text_search_tsv, postgresql_using="gin")
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
        table_name: str = "embeddings",
        fts_language: str = DEFAULT_FTS_LANGUAGE,
    ) -> None:
        self.engine = engine
        self.embedding_model = embedding_model
        self.fts_language = fts_language
        supported_dimensions = embedding_dimensions_for_model(embedding_model)
        self.dimensions = dimensions or supported_dimensions
        if self.dimensions != supported_dimensions:
            raise ValueError(
                f"Model '{embedding_model}' requires {supported_dimensions} dimensions, "
                f"got {self.dimensions}"
            )
        self.metadata = metadata if metadata is not None else MetaData()
        self.table = define_embeddings_table(
            metadata=self.metadata,
            dimensions=self.dimensions,
            embedding_model=embedding_model,
            table_name=table_name,
            fts_language=fts_language,
        )

    def create_schema(self) -> None:
        """Create the vector extension, embeddings table, HNSW and FTS indexes."""
        t = self.table.name
        with self.engine.begin() as conn:
            conn.execute(sql_text("CREATE EXTENSION IF NOT EXISTS vector"))
            self.metadata.create_all(conn)
            # Idempotent migration for tables created before FTS support was added.
            conn.execute(sql_text(
                f"ALTER TABLE {t} ADD COLUMN IF NOT EXISTS "
                f"text_search_tsv tsvector "
                f"GENERATED ALWAYS AS (to_tsvector('{self.fts_language}', text)) STORED"
            ))
            conn.execute(sql_text(
                f"CREATE INDEX IF NOT EXISTS ix_{t}_text_search_gin "
                f"ON {t} USING gin(text_search_tsv)"
            ))

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
                index_elements=[self.table.c.embedding_model, self.table.c.content_hash],
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
        if ef_search is not None and ef_search < 1:
            raise ValueError("ef_search must be greater than zero")

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
                conn.exec_driver_sql(f"SET LOCAL hnsw.ef_search = {int(ef_search)}")
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

    def fts_search(
        self,
        query_text: str,
        *,
        top_k: int = 5,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[FTSSearchResult]:
        """Return chunks matching query_text ordered by ts_rank_cd (cover-density rank)."""
        if top_k < 1:
            raise ValueError("top_k must be greater than zero")
        if not query_text.strip():
            return []

        meta_clause = ""
        params: dict[str, Any] = {
            "query_text": query_text,
            "model": self.embedding_model,
            "top_k": top_k,
        }
        if metadata_filter:
            meta_clause = "AND e.metadata @> CAST(:meta_filter AS jsonb)"
            params["meta_filter"] = json.dumps(dict(metadata_filter))

        stmt = sql_text(f"""
            WITH q AS (
                SELECT websearch_to_tsquery('{self.fts_language}', :query_text) AS tsq
            )
            SELECT e.id, e.text, e.metadata, e.content_hash, e.embedding_model,
                   ts_rank_cd(e.text_search_tsv, q.tsq) AS fts_score
            FROM {self.table.name} e, q
            WHERE e.embedding_model = :model
              AND e.text_search_tsv IS NOT NULL
              AND e.text_search_tsv @@ q.tsq
              {meta_clause}
            ORDER BY fts_score DESC
            LIMIT :top_k
        """)

        with self.engine.begin() as conn:
            rows = conn.execute(stmt, params).mappings()
            return [
                FTSSearchResult(
                    id=row["id"],
                    text=row["text"],
                    metadata=dict(row["metadata"]),
                    content_hash=row["content_hash"],
                    embedding_model=row["embedding_model"],
                    fts_score=float(row["fts_score"]),
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
