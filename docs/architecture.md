# Architecture

This project follows a `src` layout with focused modules:

- `py_rag_engine.domain`: shared domain entities (`DocumentChunk`, `ChunkMetadata`).
- `py_rag_engine.ingestion`: file loading and ingestion orchestration.
- `py_rag_engine.chunking`: recursive and semantic chunking strategies.
- `py_rag_engine.embeddings`: hashing and embedding-related helpers.
- `py_rag_engine.retrieval`: semantic similarity and ranking helpers.
- `py_rag_engine.storage`: PostgreSQL/pgvector persistence and ANN search.

## Ingestion Flow

The ingestion pipeline follows a deterministic sequence:

1. Resolve and validate the input path.
2. Select a loader from the file suffix.
3. Extract text as page-like units (`LoadedPage`).
4. Optionally split each page into semantic paragraph groups with embeddings.
5. Apply recursive character chunking with dynamic overlap.
6. Hash normalized chunk text with SHA-256.
7. Emit `DocumentChunk` objects with text, source metadata, and content hash.

## Data and Generated Outputs

Data and examples are separated from library code:

- `examples/sample_hf.md`
- `data/hf_rows.json`
- `data/gdp_first_rows.json`
- `data/*.metadata.json`

Large source files and generated full outputs are local-only:

- `data/*.pdf`
- `results/`
- `outputs/`

This keeps the repository lightweight while still preserving metadata that helps
the team inspect practical processing results.

## Persistence

`py_rag_engine.storage.postgres` defines the `embeddings` table with SQLAlchemy:

- `embedding`: pgvector `vector(n)`, where `n` is `1536` for
  `text-embedding-3-small`/`openai-3-small` or `1024` for `bge-m3`.
- `text`: original chunk text.
- `metadata`: PostgreSQL `JSONB` for source, page, chunk index, and extra
  attributes.
- `content_hash` and `embedding_model`: deduplication key for idempotent upsert.

The storage setup enables the PostgreSQL `vector` extension and creates an HNSW
index with `vector_cosine_ops`:

```sql
CREATE INDEX ix_embeddings_embedding_hnsw_cosine
ON embeddings USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

Similarity search orders by cosine distance and applies `LIMIT`, which is the
query shape required for PostgreSQL to use pgvector nearest-neighbor indexes.
