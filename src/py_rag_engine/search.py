from __future__ import annotations

import asyncio
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import text as sql_text
from sqlalchemy.ext.asyncio import AsyncEngine

from py_rag_engine.storage import DEFAULT_EMBEDDING_MODEL, DEFAULT_FTS_LANGUAGE

DEFAULT_RRF_K = 60
DEFAULT_SEARCH_TOP_K = 20
DEFAULT_VECTOR_TOP_K = 20
DEFAULT_FTS_TOP_K = 20

_IDENTIFIER_RE = re.compile(r"^[a-z_][a-z0-9_]*$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class VectorSearchDocument:
    """A document returned by the pgvector branch of hybrid search."""

    id: int
    text: str
    metadata: dict[str, Any]
    content_hash: str | None
    embedding_model: str
    vector_rank: int
    cosine_distance: float
    cosine_similarity: float


@dataclass(frozen=True, slots=True)
class FullTextSearchDocument:
    """A document returned by the PostgreSQL full-text-search branch."""

    id: int
    text: str
    metadata: dict[str, Any]
    content_hash: str | None
    embedding_model: str
    fts_rank: int
    fts_score: float


@dataclass(frozen=True, slots=True)
class HybridSearchDocument:
    """A document fused from vector and full-text rankings with RRF."""

    id: int
    text: str
    metadata: dict[str, Any]
    content_hash: str | None
    embedding_model: str
    rrf_score: float
    confidence_score: float
    vector_rank: int | None
    fts_rank: int | None
    cosine_similarity: float | None
    fts_score: float | None


def _validate_identifier(value: str, *, label: str) -> str:
    if not _IDENTIFIER_RE.match(value):
        raise ValueError(f"Invalid {label} '{value}'. Use a simple SQL identifier.")
    return value


def _vector_literal(query_embedding: Sequence[float]) -> str:
    if not query_embedding:
        raise ValueError("query_embedding must not be empty")

    values: list[str] = []
    for value in query_embedding:
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("query_embedding must contain only finite numbers")
        values.append(f"{numeric:.17g}")
    return f"[{','.join(values)}]"


def reciprocal_rank_fusion(
    vector_results: Sequence[VectorSearchDocument],
    fts_results: Sequence[FullTextSearchDocument],
    *,
    rrf_k: int = DEFAULT_RRF_K,
    top_k: int = DEFAULT_SEARCH_TOP_K,
) -> list[HybridSearchDocument]:
    """Fuse vector and FTS ranked lists with Reciprocal Rank Fusion."""
    if rrf_k < 1:
        raise ValueError("rrf_k must be greater than zero")
    if top_k < 1:
        raise ValueError("top_k must be greater than zero")

    scores: dict[int, float] = {}
    vector_by_id: dict[int, VectorSearchDocument] = {}
    fts_by_id: dict[int, FullTextSearchDocument] = {}

    for rank, result in enumerate(vector_results, start=1):
        scores[result.id] = scores.get(result.id, 0.0) + 1.0 / (rrf_k + rank)
        vector_by_id[result.id] = result

    for rank, result in enumerate(fts_results, start=1):
        scores[result.id] = scores.get(result.id, 0.0) + 1.0 / (rrf_k + rank)
        fts_by_id[result.id] = result

    ranked_ids = sorted(scores, key=lambda doc_id: scores[doc_id], reverse=True)
    output: list[HybridSearchDocument] = []
    for doc_id in ranked_ids[:top_k]:
        vector_result = vector_by_id.get(doc_id)
        fts_result = fts_by_id.get(doc_id)
        source = vector_result or fts_result
        if source is None:  # defensive; ranked_ids only contains populated ids
            continue

        output.append(
            HybridSearchDocument(
                id=doc_id,
                text=source.text,
                metadata=dict(source.metadata),
                content_hash=source.content_hash,
                embedding_model=source.embedding_model,
                rrf_score=scores[doc_id],
                confidence_score=scores[doc_id],
                vector_rank=vector_result.vector_rank if vector_result else None,
                fts_rank=fts_result.fts_rank if fts_result else None,
                cosine_similarity=vector_result.cosine_similarity if vector_result else None,
                fts_score=fts_result.fts_score if fts_result else None,
            )
        )
    return output


class AsyncHybridSearcher:
    """Run pgvector and PostgreSQL FTS searches concurrently and fuse them with RRF."""

    def __init__(
        self,
        engine: AsyncEngine,
        *,
        table_name: str = "embeddings",
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        fts_language: str = DEFAULT_FTS_LANGUAGE,
    ) -> None:
        self.engine = engine
        self.table_name = _validate_identifier(table_name, label="table_name")
        self.embedding_model = embedding_model
        self.fts_language = _validate_identifier(fts_language, label="fts_language")

    async def search(
        self,
        query_text: str,
        query_embedding: Sequence[float],
        *,
        top_k: int = DEFAULT_SEARCH_TOP_K,
        vector_k: int = DEFAULT_VECTOR_TOP_K,
        fts_k: int = DEFAULT_FTS_TOP_K,
        rrf_k: int = DEFAULT_RRF_K,
        metadata_filter: Mapping[str, Any] | None = None,
        ef_search: int | None = None,
    ) -> list[HybridSearchDocument]:
        """Return the top documents from parallel vector + FTS search fused by RRF."""
        if top_k < 1:
            raise ValueError("top_k must be greater than zero")
        if vector_k < 1:
            raise ValueError("vector_k must be greater than zero")
        if fts_k < 1:
            raise ValueError("fts_k must be greater than zero")
        if rrf_k < 1:
            raise ValueError("rrf_k must be greater than zero")
        if ef_search is not None and ef_search < 1:
            raise ValueError("ef_search must be greater than zero")

        vector_task = asyncio.create_task(
            self._vector_search(
                query_embedding,
                top_k=vector_k,
                metadata_filter=metadata_filter,
                ef_search=ef_search,
            )
        )
        fts_task = asyncio.create_task(
            self._full_text_search(
                query_text,
                top_k=fts_k,
                metadata_filter=metadata_filter,
            )
        )
        vector_results, fts_results = await asyncio.gather(vector_task, fts_task)
        return reciprocal_rank_fusion(
            vector_results,
            fts_results,
            rrf_k=rrf_k,
            top_k=top_k,
        )

    async def _vector_search(
        self,
        query_embedding: Sequence[float],
        *,
        top_k: int,
        metadata_filter: Mapping[str, Any] | None,
        ef_search: int | None,
    ) -> list[VectorSearchDocument]:
        vector = _vector_literal(query_embedding)
        metadata_clause = ""
        params: dict[str, Any] = {
            "query_embedding": vector,
            "embedding_model": self.embedding_model,
            "top_k": top_k,
        }
        if metadata_filter:
            metadata_clause = "AND e.metadata @> CAST(:metadata_filter AS jsonb)"
            params["metadata_filter"] = json.dumps(dict(metadata_filter))

        statement = sql_text(f"""
            SELECT
                e.id,
                e.text,
                e.metadata,
                e.content_hash,
                e.embedding_model,
                e.embedding <=> CAST(:query_embedding AS vector) AS cosine_distance,
                1 - (e.embedding <=> CAST(:query_embedding AS vector)) AS cosine_similarity
            FROM {self.table_name} AS e
            WHERE e.embedding_model = :embedding_model
              {metadata_clause}
            ORDER BY e.embedding <=> CAST(:query_embedding AS vector)
            LIMIT :top_k
        """)

        async with self.engine.begin() as conn:
            if ef_search is not None:
                await conn.exec_driver_sql(f"SET LOCAL hnsw.ef_search = {int(ef_search)}")
            rows = (await conn.execute(statement, params)).mappings().all()

        return [
            VectorSearchDocument(
                id=int(row["id"]),
                text=str(row["text"]),
                metadata=dict(row["metadata"]),
                content_hash=row["content_hash"],
                embedding_model=str(row["embedding_model"]),
                vector_rank=rank,
                cosine_distance=float(row["cosine_distance"]),
                cosine_similarity=float(row["cosine_similarity"]),
            )
            for rank, row in enumerate(rows, start=1)
        ]

    async def _full_text_search(
        self,
        query_text: str,
        *,
        top_k: int,
        metadata_filter: Mapping[str, Any] | None,
    ) -> list[FullTextSearchDocument]:
        if not query_text.strip():
            return []

        metadata_clause = ""
        params: dict[str, Any] = {
            "query_text": query_text,
            "embedding_model": self.embedding_model,
            "top_k": top_k,
        }
        if metadata_filter:
            metadata_clause = "AND e.metadata @> CAST(:metadata_filter AS jsonb)"
            params["metadata_filter"] = json.dumps(dict(metadata_filter))

        statement = sql_text(f"""
            WITH q AS (
                SELECT websearch_to_tsquery('{self.fts_language}', :query_text) AS tsq
            )
            SELECT
                e.id,
                e.text,
                e.metadata,
                e.content_hash,
                e.embedding_model,
                ts_rank_cd(e.text_search_tsv, q.tsq) AS fts_score
            FROM {self.table_name} AS e, q
            WHERE e.embedding_model = :embedding_model
              AND e.text_search_tsv IS NOT NULL
              AND e.text_search_tsv @@ q.tsq
              {metadata_clause}
            ORDER BY fts_score DESC
            LIMIT :top_k
        """)

        async with self.engine.begin() as conn:
            rows = (await conn.execute(statement, params)).mappings().all()

        return [
            FullTextSearchDocument(
                id=int(row["id"]),
                text=str(row["text"]),
                metadata=dict(row["metadata"]),
                content_hash=row["content_hash"],
                embedding_model=str(row["embedding_model"]),
                fts_rank=rank,
                fts_score=float(row["fts_score"]),
            )
            for rank, row in enumerate(rows, start=1)
        ]


async def hybrid_search(
    engine: AsyncEngine,
    query_text: str,
    query_embedding: Sequence[float],
    *,
    top_k: int = DEFAULT_SEARCH_TOP_K,
    vector_k: int = DEFAULT_VECTOR_TOP_K,
    fts_k: int = DEFAULT_FTS_TOP_K,
    rrf_k: int = DEFAULT_RRF_K,
    table_name: str = "embeddings",
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    fts_language: str = DEFAULT_FTS_LANGUAGE,
    metadata_filter: Mapping[str, Any] | None = None,
    ef_search: int | None = None,
) -> list[HybridSearchDocument]:
    """Convenience function for one-off async hybrid searches."""
    searcher = AsyncHybridSearcher(
        engine,
        table_name=table_name,
        embedding_model=embedding_model,
        fts_language=fts_language,
    )
    return await searcher.search(
        query_text,
        query_embedding,
        top_k=top_k,
        vector_k=vector_k,
        fts_k=fts_k,
        rrf_k=rrf_k,
        metadata_filter=metadata_filter,
        ef_search=ef_search,
    )
