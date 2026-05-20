from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    postgres: str
    lm_studio: str


class DocumentInfo(BaseModel):
    source: str
    chunks: int


class IngestResponse(BaseModel):
    source: str
    chunks_ingested: int
    chunk_ids: list[int]


class SourceResult(BaseModel):
    text: str
    source: str
    page: int | None
    chunk_index: int | None
    score: float


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(default=5, ge=1, le=20)
    use_hybrid: bool = True
    use_rerank: bool = True
    generate_answer: bool = True
    metadata_filter: dict[str, object] | None = None


class QueryResponse(BaseModel):
    answer: str | None
    sources: list[SourceResult]
    retrieval_mode: str
