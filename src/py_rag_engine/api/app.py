from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from sqlalchemy import create_engine

from py_rag_engine.clients import LMStudioClient
from py_rag_engine.config import LMStudioConfig, PostgresConfig
from py_rag_engine.embeddings.lm_studio_embedder import make_lm_studio_embed
from py_rag_engine.retrieval import CrossEncoderReranker
from py_rag_engine.storage import PostgresEmbeddingStore
from py_rag_engine.storage.postgres import EMBEDDING_MODEL_DIMENSIONS

from .routes import router


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    lm_cfg = LMStudioConfig.from_env()
    pg_cfg = PostgresConfig.from_env()

    # Allow custom / unlisted embedding models by providing dimensions explicitly.
    embed_model = lm_cfg.embed_model
    if embed_model not in EMBEDDING_MODEL_DIMENSIONS:
        dims_env = os.environ.get("EMBEDDING_DIMENSIONS")
        if dims_env is None:
            supported = ", ".join(sorted(EMBEDDING_MODEL_DIMENSIONS))
            raise RuntimeError(
                f"Embedding model '{embed_model}' is not in the known registry. "
                f"Set EMBEDDING_DIMENSIONS=<int> to register it, or choose one of: {supported}"
            )
        EMBEDDING_MODEL_DIMENSIONS[embed_model] = int(dims_env)

    engine = create_engine(pg_cfg.url)
    store = PostgresEmbeddingStore(engine, embedding_model=embed_model)
    store.create_schema()

    client = LMStudioClient(lm_cfg)
    embed = make_lm_studio_embed(client)
    reranker = CrossEncoderReranker()

    app.state.store = store
    app.state.embed = embed
    app.state.reranker = reranker
    app.state.lm_client = client

    yield

    engine.dispose()


def create_app() -> FastAPI:
    """Factory for the py-rag-engine REST API.

    Run with:
        uvicorn "py_rag_engine.api:create_app" --factory --reload
    """
    app = FastAPI(
        title="py-rag-engine",
        description=(
            "RAG API: ingest PDFs and Markdown documents, "
            "then query them with hybrid search (dense ANN + FTS) "
            "and cross-encoder re-ranking."
        ),
        version="0.1.0",
        lifespan=_lifespan,
    )
    app.include_router(router)
    return app
