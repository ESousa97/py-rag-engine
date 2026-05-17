# Architecture

The project follows a `src` layout with strict per-responsibility modules.
Library code lives under `src/py_rag_engine/`, CLI entry points under
`scripts/`, gold-standard data under `data/`, and JSON reports under
`reports/`.

## Module Map

| Module | Responsibility |
|---|---|
| `py_rag_engine.config` | `LMStudioConfig` / `PostgresConfig` / `EvalConfig` — env-driven, single source of truth |
| `py_rag_engine.domain` | Shared entities (`DocumentChunk`, `ChunkMetadata`) |
| `py_rag_engine.vector_math` | `cosine_similarity` helpers |
| `py_rag_engine.clients` | HTTP adapters (today: `LMStudioClient`, `detect_chat_model`) |
| `py_rag_engine.ingestion` | File loaders (`load_pdf`, `load_markdown`) + ingestion pipeline |
| `py_rag_engine.chunking` | Recursive and semantic chunking strategies |
| `py_rag_engine.embeddings` | Hashing + embedder factories (LM Studio, SentenceTransformer) |
| `py_rag_engine.storage` | PostgreSQL + pgvector persistence and ANN search |
| `py_rag_engine.retrieval` | Dense ranking + cross-encoder re-ranking |
| `py_rag_engine.generation` | LLM-grounded answer generation (LM Studio chat) |
| `py_rag_engine.evaluation` | RAGAS-style metrics and the `EvalRunner` orchestrator |

Each module is a *single responsibility*: clients perform HTTP, embedders
generate vectors, the runner orchestrates. Scripts are thin CLIs (~150 LOC
each) that wire the library together — they do not contain business logic.

## End-to-End Pipeline

```
PDF / Markdown
      │
      ▼
  Ingestion (ingestion.loaders + ingestion.pipeline)
  ├─ Loader picks `.pdf` or `.md`
  ├─ Optional semantic paragraph chunking (chunking.semantic)
  ├─ Recursive character splitting (chunking.recursive)
  └─ SHA-256 dedupe (embeddings.hashing)
      │
      ▼
  Embedding (embeddings.lm_studio_embedder | embeddings.sentence_transformer)
      │  bge-m3 (1024d) via LM Studio, or all-MiniLM-L6-v2 (384d) locally
      ▼
  Storage (storage.postgres.PostgresEmbeddingStore)
      │  pgvector HNSW cosine index, JSONB metadata, idempotent upsert
      ▼
  Dense Recall (storage.similarity_search, top-N candidates)
      │
      ▼
  Re-ranking (retrieval.rerank.CrossEncoderReranker)
      │  ms-marco-MiniLM-L-6-v2, top-K final
      ▼
  Generation (generation.lm_studio_chat)
      │  Grounded answer constrained to retrieved context
      ▼
  Evaluation (evaluation.metrics + evaluation.runner)
      │  Faithfulness / Answer Relevancy / Context Precision per question
      ▼
  JSON Report (reports/eval_report_<UTC>.json)
```

## Ingestion Flow

1. Resolve and validate the input path.
2. Select a loader from the file suffix (`.pdf`, `.md`, `.markdown`).
3. Extract text as page-like units (`LoadedPage`).
4. Optionally split each page into semantic paragraph groups using embeddings.
5. Apply recursive character chunking with dynamic overlap (10–15% of chunk size).
6. Hash normalized chunk text with SHA-256.
7. Emit `DocumentChunk` objects with text, source metadata, and content hash.

## Persistence

`storage.postgres.PostgresEmbeddingStore` defines the embeddings table:

- `embedding`: pgvector `vector(n)`, with `n` from `EMBEDDING_MODEL_DIMENSIONS`
  (1024 for `bge-m3`, 1536 for `openai-3-small`, 384 for `all-MiniLM-L6-v2`).
- `text`: original chunk text.
- `metadata`: PostgreSQL `JSONB` for source, page, chunk_index, and extras.
- `content_hash` + `embedding_model`: idempotent-upsert key.

Indexes:

```sql
CREATE INDEX ix_<t>_embedding_hnsw_cosine
ON <t> USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX ix_<t>_metadata_gin
ON <t> USING gin (metadata);
```

The store supports `table_name=` so the eval pipeline can isolate each
configuration into its own table (`eval_<model>_<chunk>`) without polluting the
default `embeddings` table.

## Evaluation Pipeline

The `EvalRunner` (`evaluation.runner`) runs one configuration in five stages:

1. **Ingest** — `ingest_file(path, chunk_size)` with internal caching so the
   same document is only chunked once per chunk size.
2. **Embed** — calls the supplied `EmbedFn` (LM Studio or SentenceTransformer).
3. **Persist** — drops the per-config table, recreates it, bulk-inserts.
4. **Retrieve + Generate** — for each gold-standard question, runs dense ANN
   search (`top_k=5`, `ef_search=80`) then calls `generate_answer`.
5. **Score** — three RAGAS metrics via `evaluate_samples`, with optional
   fallthrough to the official `ragas` library when
   `EVAL_USE_OFFICIAL_RAGAS=1`.

The top-level driver in `scripts/eval_ragas.py` materialises a list of
(model, chunk_size) configurations, invokes the runner for each, and writes
a single timestamped JSON report aggregating all results.

See [evaluation.md](evaluation.md) for the metric definitions and report
schema.

## Configuration

Three frozen dataclasses concentrate all environment-driven settings:

| Class | Source | Used by |
|---|---|---|
| `LMStudioConfig` | `LM_STUDIO_*` env vars | `LMStudioClient`, scripts |
| `PostgresConfig` | `EVAL_POSTGRES_URL` | scripts |
| `EvalConfig` | `EVAL_*` env vars | `scripts/eval_ragas.py` |

Construct from environment via `LMStudioConfig.from_env()`. Override
fields in tests by passing keyword args directly to the dataclass.

## Data Layout

- `data/eval_document.md` — source document used by the offline evaluation.
- `data/eval_questions.json` — 10 gold-standard Q&A pairs (schema v1).
- `data/gdp_document_0.pdf` — demo PDF (gitignored binary; metadata is versioned).
- `reports/` — timestamped JSON evaluation reports (gitignored).
- `examples/`, `results/` — local working artifacts (gitignored when generated by scripts).
