from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, Request, UploadFile
from sqlalchemy import text as sql_text

from py_rag_engine.clients import LMStudioClient
from py_rag_engine.generation.lm_studio_chat import generate_answer
from py_rag_engine.ingestion.pipeline import ingest_file
from py_rag_engine.retrieval import (
    CrossEncoderReranker,
    retrieve_hybrid_with_rerank,
    retrieve_with_rerank,
)
from py_rag_engine.storage import EmbeddingInput, PostgresEmbeddingStore

from .schemas import (
    DocumentInfo,
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceResult,
)

router = APIRouter()

_ALLOWED_SUFFIXES = {".pdf", ".md", ".markdown"}


def _check_file_type(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in _ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Accepted: {sorted(_ALLOWED_SUFFIXES)}",
        )
    return suffix


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health(request: Request) -> HealthResponse:
    """Check connectivity to PostgreSQL and LM Studio."""
    store: PostgresEmbeddingStore = request.app.state.store
    client: LMStudioClient = request.app.state.lm_client

    try:
        with store.engine.connect() as conn:
            conn.execute(sql_text("SELECT 1"))
        pg_status = "ok"
    except Exception:
        pg_status = "unreachable"

    try:
        client.models()
        lm_status = "ok"
    except Exception:
        lm_status = "unreachable"

    overall = "ok" if pg_status == "ok" and lm_status == "ok" else "degraded"
    return HealthResponse(status=overall, postgres=pg_status, lm_studio=lm_status)


@router.get("/documents", response_model=list[DocumentInfo], tags=["documents"])
def list_documents(request: Request) -> list[DocumentInfo]:
    """List all ingested sources with their chunk count."""
    store: PostgresEmbeddingStore = request.app.state.store
    stmt = sql_text("""
        SELECT metadata->>'source' AS source, COUNT(*) AS chunks
        FROM embeddings
        WHERE embedding_model = :model
          AND metadata ? 'source'
        GROUP BY source
        ORDER BY source
    """)
    with store.engine.connect() as conn:
        rows = conn.execute(stmt, {"model": store.embedding_model}).mappings()
        return [
            DocumentInfo(source=Path(row["source"]).name, chunks=int(row["chunks"]))
            for row in rows
        ]


@router.post("/documents", status_code=201, response_model=IngestResponse, tags=["documents"])
def ingest_document(
    request: Request,
    file: UploadFile = File(..., description="PDF, .md or .markdown file to ingest."),
    chunk_size: int = Query(default=1200, ge=100, le=8000, description="Target characters per chunk."),
) -> IngestResponse:
    """Upload and ingest a PDF or Markdown document.

    The file is chunked, embedded via LM Studio, and stored in PostgreSQL
    with pgvector + full-text search indexes.  Re-uploading the same content
    is idempotent (deduplication by SHA-256 hash).
    """
    suffix = _check_file_type(file.filename)

    content: bytes = file.file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    store: PostgresEmbeddingStore = request.app.state.store
    embed = request.app.state.embed

    tmp_fd, tmp_name = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(tmp_fd, content)
        os.close(tmp_fd)
        chunks = ingest_file(Path(tmp_name), chunk_size=chunk_size)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass

    if not chunks:
        return IngestResponse(source=file.filename or "unknown", chunks_ingested=0, chunk_ids=[])

    vectors: list[list[float]] = embed([c.text for c in chunks])
    inputs = [
        EmbeddingInput(
            text=chunk.text,
            embedding=vec,
            content_hash=chunk.content_hash,
            metadata=chunk.metadata.to_dict(),
            embedding_model=store.embedding_model,
        )
        for chunk, vec in zip(chunks, vectors)
    ]
    ids = store.add_embeddings(inputs)

    return IngestResponse(
        source=file.filename or "unknown",
        chunks_ingested=len(ids),
        chunk_ids=ids,
    )


@router.post("/query", response_model=QueryResponse, tags=["retrieval"])
def query_documents(request: Request, body: QueryRequest) -> QueryResponse:
    """Query the knowledge base with a natural-language question.

    The pipeline:
    1. Embed the question via LM Studio.
    2. **Hybrid** (default): dense ANN (pgvector) + FTS (PostgreSQL tsvector), merged with RRF.
       **Dense-only** (`use_hybrid=false`): ANN search only.
    3. Cross-encoder re-ranking on the fused candidate pool.
    4. Optionally generate a grounded answer with the LM Studio chat model.
    """
    store: PostgresEmbeddingStore = request.app.state.store
    embed = request.app.state.embed
    reranker: CrossEncoderReranker = request.app.state.reranker
    client: LMStudioClient = request.app.state.lm_client

    try:
        query_vec: list[float] = embed([body.question])[0]
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Embedding service unavailable.") from exc

    if body.use_hybrid:
        results = retrieve_hybrid_with_rerank(
            body.question,
            query_vec,
            store,
            reranker,
            top_k=body.top_k,
            metadata_filter=body.metadata_filter,
        )
        retrieval_mode = "hybrid_rerank"
    else:
        results = retrieve_with_rerank(
            body.question,
            query_vec,
            store,
            reranker,
            top_k=body.top_k,
            metadata_filter=body.metadata_filter,
        )
        retrieval_mode = "dense_rerank"

    sources = [
        SourceResult(
            text=r.text,
            source=Path(r.metadata.get("source", "")).name or r.metadata.get("source", ""),
            page=r.metadata.get("page"),
            chunk_index=r.metadata.get("chunk_index"),
            score=round(r.rerank_score, 6),
        )
        for r in results
    ]

    answer: str | None = None
    if body.generate_answer and results:
        try:
            answer = generate_answer(client, body.question, [r.text for r in results])
        except Exception:
            answer = None

    return QueryResponse(answer=answer, sources=sources, retrieval_mode=retrieval_mode)
