# py-rag-engine

A Python RAG (Retrieval-Augmented Generation) engine that covers the full
pipeline from document ingestion to re-ranked retrieval. It ingests PDF and
Markdown files, chunks text recursively (with optional semantic splitting),
stores embeddings in PostgreSQL with pgvector, performs dense ANN search, and
re-ranks the candidates with a Cross-Encoder for higher precision.

## Pipeline Overview

```
PDF / Markdown
      │
      ▼
  Ingestion ──► Recursive / Semantic Chunking ──► SHA-256 deduplication
      │
      ▼
  Embedding (LM Studio / any OpenAI-compatible endpoint)
      │
      ▼
  PostgreSQL + pgvector  ◄──  HNSW cosine index (bge-m3 · 1024 dims)
      │
      ▼
  Dense recall  (top-20 candidates via pgvector ANN)
      │
      ▼
  Cross-Encoder re-rank  (ms-marco-MiniLM-L-6-v2, top-5 final)
      │
      ▼
  RerankedResult list ordered by relevance score
```

## Features

- PDF page extraction with `pypdf`, `.md`/`.markdown` ingestion.
- Recursive chunking via `RecursiveCharacterTextSplitter` with dynamic overlap.
- Optional semantic paragraph chunking using cosine similarity between embeddings.
- SHA-256 content hashes for duplicate detection.
- PostgreSQL + pgvector persistence with HNSW cosine index and JSONB metadata.
- Dense ANN search returning `cosine_similarity = 1 − cosine_distance`.
- **Two-stage retrieval**: dense recall (top-20) + Cross-Encoder re-rank (top-5).
- `CrossEncoderReranker` with lazy model loading and injectable `predict` for tests.
- `retrieve_with_rerank` orchestrator that wires the two stages together.
- 27 unit/integration tests — CI runs on Python 3.11 and 3.12.

## Project Layout

```text
py-rag-engine/
├── .cache/                              # local model cache (gitignored)
│   └── ms-marco-MiniLM-L-6-v2/         # Cross-Encoder cloned from HuggingFace
├── data/
│   ├── gdp_document_0.metadata.json     # lightweight chunk metadata (versioned)
│   ├── gdp_first_rows.json              # Hugging Face first-rows response
│   └── hf_rows.json                     # Markdown dataset sample
├── docs/
│   └── architecture.md
├── examples/
│   └── sample_hf.md
├── scripts/
│   ├── demo_rerank.py                   # end-to-end E2E demo (ingest→embed→store→rerank)
│   └── process_document.py             # CLI for PDF/Markdown chunk processing
├── src/
│   └── py_rag_engine/
│       ├── chunking/
│       │   ├── recursive.py            # recursive splitter and dynamic overlap
│       │   └── semantic.py             # embedding-based semantic splitting
│       ├── embeddings/
│       │   └── hashing.py              # SHA-256 content hash helpers
│       ├── ingestion/
│       │   ├── loaders.py              # PDF and Markdown loaders
│       │   └── pipeline.py             # ingestion orchestration
│       ├── retrieval/
│       │   ├── rerank.py               # CrossEncoderReranker + rerank_candidates
│       │   ├── semantic.py             # cosine_similarity helper
│       │   └── service.py              # rank_chunks_by_similarity + retrieve_with_rerank
│       ├── storage/
│       │   └── postgres.py             # PostgresEmbeddingStore (pgvector)
│       ├── domain.py                   # DocumentChunk and ChunkMetadata
│       └── vector_math.py              # numpy cosine similarity
├── tests/
│   ├── test_chunking.py
│   ├── test_embeddings.py
│   ├── test_ingestion.py
│   ├── test_postgres_integration.py    # requires TEST_POSTGRES_URL + LM_STUDIO_BASE_URL
│   ├── test_postgres_storage.py
│   └── test_retrieval.py               # includes re-ranking unit tests
├── pyproject.toml
└── README.md
```

---

## Prerequisites

| Requirement | Version tested | Notes |
|---|---|---|
| Python | 3.11 / 3.12 / 3.14 | 3.11+ required |
| PostgreSQL + pgvector | pg16 + pgvector 0.8.2 | via Docker (see below) |
| LM Studio | any | exposes `/v1/embeddings` on `localhost:1234` |
| Model `gpustack/bge-m3-GGUF` | bge-m3-Q8_0 | loaded inside LM Studio |
| sentence-transformers | 5.4.1 | for Cross-Encoder re-ranking |
| Git LFS | any | to clone the Cross-Encoder weights |

---

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[embeddings]"
pip install pytest
```

The `[embeddings]` extra installs `sentence-transformers`, which provides both
`SentenceTransformer` (for semantic chunking) and `CrossEncoder` (for re-ranking).

---

## Infrastructure Setup

### 1. PostgreSQL + pgvector via Docker

The easiest way to get a pgvector-enabled PostgreSQL is to run the official
`pgvector/pgvector` Docker image.

```powershell
docker run -d `
  --name rag-pgvector `
  -e POSTGRES_PASSWORD=[passowrd] `
  -e POSTGRES_DB=rag `
  -p 5434:5432 `
  pgvector/pgvector:pg16
```

Wait for it to be ready:

```powershell
docker exec rag-pgvector pg_isready -U postgres
# output: /var/run/postgresql:5432 - accepting connections
```

Connection string used throughout this project:

```
postgresql+psycopg://postgres:[password]@localhost:5434/rag
```

To stop and remove the container when done:

```powershell
docker stop rag-pgvector
docker rm rag-pgvector
```

### 2. LM Studio — Embedding Model

1. Download and install [LM Studio](https://lmstudio.ai).
2. Download model **`gpustack/bge-m3-GGUF`** (file `bge-m3-Q8_0.gguf`, ~600 MB).
3. Go to **Developer → Local Server**, select the model, and click **Start Server**.
4. The server exposes `http://localhost:1234/v1/embeddings`.

Verify the server is up:

```powershell
Invoke-RestMethod http://localhost:1234/v1/models | ConvertTo-Json
```

Expected response includes `"id": "text-embedding-bge-m3"`.

### 3. Cross-Encoder Model (local download)

> **Why local?** On some Windows machines Python's `httpx` (used by
> `huggingface_hub`) cannot verify the HuggingFace TLS certificate. Cloning
> via Git sidesteps the issue because Git uses its own CA bundle.

```powershell
mkdir .cache
cd .cache
git clone https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2
cd ..
```

This downloads ~430 MB including `model.safetensors`. The `.cache/` directory
is gitignored so the weights are never committed.

---

## Download the Test PDF

The end-to-end demo uses a geotechnical investigation report from Hugging Face:

```powershell
$rows = Invoke-RestMethod `
  "https://datasets-server.huggingface.co/first-rows?dataset=surgeai%2FGDP.pdf&config=default&split=train"
$url = $rows.rows[0].row.pdf.src
Invoke-WebRequest -Uri $url -OutFile "data\gdp_document_0.pdf"
```

The PDF produces **50 chunks** with the default settings.

---

## End-to-End Demo

`scripts/demo_rerank.py` runs all five pipeline stages in sequence and prints a
comparison table showing how the Cross-Encoder changes the ranking produced by
the dense pgvector search.

### Environment variables

```powershell
$env:PYTHONPATH              = "src"
$env:LM_STUDIO_BASE_URL      = "http://localhost:1234"
$env:LM_STUDIO_EMBEDDING_MODEL = "text-embedding-bge-m3"
$env:DEMO_POSTGRES_URL       = "postgresql+psycopg://postgres:[password]@localhost:5434/rag"
```

### Run

```powershell
python scripts\demo_rerank.py data\gdp_document_0.pdf `
  --query "What is the geotechnical investigation methodology used in this report?" `
  --reranker-model "$PWD\.cache\ms-marco-MiniLM-L-6-v2"
```

Add `--reset-table` on the first run to start from a clean `embeddings` table.
Subsequent runs upsert by `(embedding_model, content_hash)` so the table is
never duplicated.

### Actual output (recorded during development)

```text
==============================================================================
1) INGESTION
==============================================================================
file              = data/gdp_document_0.pdf
chunks produced   = 50
sample chunk[0]   = page=1 chars=234 hash=8276b4d156b6...

==============================================================================
2) EMBEDDING (LM Studio)
==============================================================================
lm_studio_url     = http://localhost:1234
lm_studio_model   = text-embedding-bge-m3
chunks_embedded   = 50 in 9.98s
vector_dim        = 1024

==============================================================================
3) PERSIST (PostgreSQL + pgvector)
==============================================================================
postgres_url      = postgresql+psycopg://postgres:[password]@localhost:5434/rag
storage_model     = bge-m3
rows_upserted     = 50

==============================================================================
4) DENSE RECALL (pgvector, top 20)
==============================================================================
query             = 'What is the geotechnical investigation methodology used in this report?'
candidates_pulled = 20

rank  cos_sim  page  idx  preview
------------------------------------------------------------------------------
   1   0.6405     2    1  GEOTECHNICAL INVESTIGATION REPORT    1. INTRODUCTION
   2   0.6124     1    0  GEOTECHNICAL INVESTIGATION REPORT      1. INTRODUCTION
   3   0.5557     6    8  ANNEXURE-I      1.0 INTRODUCTION       PASSENGER ROPEWA
   4   0.5466     5    7  BH-3  2.00 x 2.00   1.5  15  2.50 X 2.50 15  3.00 X 3.0
   5   0.4996    22   41  continues up to the terminating depth of borehole.  b)
   6   0.4963     3    3  Geological Map of Meghalaya    4. FIELD INVESTIGATION P
   7   0.4943     2    2  of borehole was measured from the existing ground level
   8   0.4911    11   20  deposits. Some of the routine tests were also carried o
   9   0.4853    26   47  BORE/DRILLLOG  ProjectName:Nearshillongpeak,UTPLocation
  10   0.4833    12   21  The above mentioned laboratory tests were conducted as
  ...  (20 candidates pulled total)

==============================================================================
5) RE-RANK (Cross-Encoder, top 5)
==============================================================================
reranker_model    = .cache/ms-marco-MiniLM-L-6-v2
loading cross-encoder (first run downloads ~80MB)...
reranked_in       = 1.80s

rank   rerank  cos_sim  page  idx  preview
------------------------------------------------------------------------------
   1    4.923   0.6124     1    0  GEOTECHNICAL INVESTIGATION REPORT      1. INTRO...
   2    3.233   0.6405     2    1  GEOTECHNICAL INVESTIGATION REPORT    1. INTRO...
   3    1.796   0.5466     5    7  BH-3  2.00 x 2.00   1.5  15  2.50 X 2.50 15...
   4    0.668   0.5557     6    8  ANNEXURE-I      1.0 INTRODUCTION       PASSENGER...
   5    0.501   0.4996    22   41  continues up to the terminating depth of borehole...

==============================================================================
6) DENSE vs RERANK COMPARISON
==============================================================================
dense_top_5        = [2, 1, 9, 8, 42]
rerank_top_5       = [1, 2, 8, 9, 42]
position_shift     = [-1, +1, -1, +1, 0]  (None = new to top 5)
```

### What the output shows

The dense recall ranked the chunk from page 2 (`id=2`, `cos_sim=0.6405`) as
the best match. The Cross-Encoder disagreed: it gave the page 1 chunk (`id=1`,
`rerank_score=4.923`) the highest relevance because its text contains the
full introduction of the methodology section, which better answers the query.

`position_shift` values of `−1` and `+1` on the first two entries show the
re-ranker flipping the top pair, validating that the two stages capture
different signals (semantic proximity vs. precise relevance).

### Demo script flags

| Flag | Default | Description |
|---|---|---|
| `--query` | geotechnical question | User question to retrieve against |
| `--candidate-k` | `20` | Number of dense candidates pulled from pgvector |
| `--top-k` | `5` | Final results after re-ranking |
| `--reset-table` | off | Drop and recreate `embeddings` before inserting |
| `--reranker-model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | HuggingFace model name or local path |
| `--storage-model` | `bge-m3` | Embedding model discriminator stored in the DB |

---

## Usage

### Document Ingestion

```powershell
$env:PYTHONPATH = 'src'
python scripts\process_document.py `
  data\gdp_document_0.pdf `
  --output results\gdp_chunks.json `
  --metadata-output data\gdp_document_0.metadata.json
```

```text
chunks=50
output=results\gdp_chunks.json
metadata_output=data\gdp_document_0.metadata.json
```

### Semantic Chunking

```powershell
$env:PYTHONPATH = 'src'
python -c "
from py_rag_engine.ingestion import ingest_file, make_sentence_transformer_embed
embed = make_sentence_transformer_embed('all-MiniLM-L6-v2')
chunks = ingest_file('examples/sample_hf.md', use_semantic_chunking=True, embed=embed)
print(len(chunks))
"
```

Useful `ingest_file` parameters:

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | `1200` | Target chunk character size |
| `chunk_overlap` | `None` | Fixed overlap; omit for dynamic |
| `overlap_ratio` | `0.12` | Dynamic overlap ratio |
| `use_semantic_chunking` | `False` | Enable embedding-based topic splitting |
| `semantic_similarity_threshold` | `0.55` | Adjacent paragraph similarity cutoff |
| `deduplicate_by_hash` | `True` | Remove duplicate chunks by SHA-256 |

---

## Main API

### Ingestion

```python
from py_rag_engine.ingestion import chunks_to_dicts, ingest_file

chunks = ingest_file("data/gdp_document_0.pdf")
payload = chunks_to_dicts(chunks)
# [{"text": "...", "metadata": {...}, "content_hash": "..."}, ...]
```

### PostgreSQL Persistence

```python
from sqlalchemy import create_engine
from py_rag_engine.storage import EmbeddingInput, PostgresEmbeddingStore

engine = create_engine("postgresql+psycopg://postgres:[password]@localhost:5434/rag")
store = PostgresEmbeddingStore(engine, embedding_model="bge-m3")
store.create_schema()

store.add_embedding(
    EmbeddingInput(
        text="chunk text",
        embedding=[0.0] * 1024,
        metadata={"source": "data/gdp_document_0.pdf", "page": 1, "chunk_index": 0},
        content_hash="sha256-of-chunk",
        embedding_model="bge-m3",
    )
)

results = store.similarity_search([0.0] * 1024, top_k=5, ef_search=80)
```

`create_schema()` runs `CREATE EXTENSION IF NOT EXISTS vector`, creates the
`embeddings` table, and builds these indexes:

```sql
CREATE INDEX ix_embeddings_embedding_hnsw_cosine
ON embeddings USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

CREATE INDEX ix_embeddings_metadata_gin
ON embeddings USING gin (metadata);
```

Supported embedding models and dimensions:

| Model key | Dimensions |
|---|---|
| `bge-m3` | 1024 |
| `openai-3-small` / `text-embedding-3-small` | 1536 |

### Re-ranking Pipeline

```python
from sqlalchemy import create_engine
from py_rag_engine.retrieval import CrossEncoderReranker, retrieve_with_rerank
from py_rag_engine.storage import PostgresEmbeddingStore

engine = create_engine("postgresql+psycopg://postgres:[password]@localhost:5434/rag")
store = PostgresEmbeddingStore(engine, embedding_model="bge-m3")

# Model name or local absolute path
reranker = CrossEncoderReranker(
    model_name="/absolute/path/to/.cache/ms-marco-MiniLM-L-6-v2"
)

results = retrieve_with_rerank(
    query="What is the geotechnical investigation methodology?",
    query_embedding=[...],   # 1024-dim vector from bge-m3
    store=store,
    reranker=reranker,
    candidate_k=20,          # dense recall pool size
    top_k=5,                 # final results after re-ranking
    ef_search=80,
)

for item in results:
    print(f"rerank={item.rerank_score:.3f}  cosine={item.cosine_similarity:.4f}  {item.text[:60]}")
```

`RerankedResult` fields:

| Field | Type | Description |
|---|---|---|
| `id` | `int` | Database row ID |
| `text` | `str` | Chunk text |
| `metadata` | `dict` | Source, page, chunk_index |
| `content_hash` | `str \| None` | SHA-256 of the original chunk |
| `embedding_model` | `str` | Model used to embed this chunk |
| `cosine_similarity` | `float` | Score from pgvector dense search |
| `rerank_score` | `float` | Score from the Cross-Encoder (higher = more relevant) |

To swap models or inject a stub `predict` function in tests:

```python
# Different model
reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-MiniLM-L-12-v2")

# Test stub — no model download needed
reranker = CrossEncoderReranker(predict=lambda pairs: [0.0] * len(pairs))
```

---

## Testing

### Unit tests (no external services required)

```powershell
python -m pytest -q
```

```text
27 passed, 1 skipped
```

The skipped test is `tests/test_postgres_integration.py`, which requires a
real Postgres instance and LM Studio.

### Integration test (full round-trip)

Set environment variables pointing to a running Postgres and LM Studio, then
run the full suite:

```powershell
$env:TEST_POSTGRES_URL        = "postgresql+psycopg://postgres:[password]@localhost:5434/rag"
$env:LM_STUDIO_BASE_URL       = "http://localhost:1234"
$env:LM_STUDIO_EMBEDDING_MODEL = "text-embedding-bge-m3"
$env:STORAGE_EMBEDDING_MODEL   = "bge-m3"
python -m pytest -q
```

```text
28 passed
```

The integration test (`test_lm_studio_embeddings_round_trip_through_postgres_pgvector`)
embeds three sentences, inserts them into the `embeddings` table, runs a
similarity query, and asserts that the HNSW index is present.

---

## Development

| Module | Purpose |
|---|---|
| `src/py_rag_engine/domain.py` | `DocumentChunk`, `ChunkMetadata` |
| `src/py_rag_engine/ingestion/pipeline.py` | Ingest orchestration |
| `src/py_rag_engine/ingestion/loaders.py` | PDF and Markdown loaders |
| `src/py_rag_engine/chunking/recursive.py` | Recursive splitter and dynamic overlap |
| `src/py_rag_engine/chunking/semantic.py` | Embedding-based topic splitting |
| `src/py_rag_engine/embeddings/hashing.py` | SHA-256 content hashing |
| `src/py_rag_engine/storage/postgres.py` | `PostgresEmbeddingStore`, pgvector |
| `src/py_rag_engine/retrieval/rerank.py` | `CrossEncoderReranker`, `rerank_candidates` |
| `src/py_rag_engine/retrieval/service.py` | `retrieve_with_rerank` pipeline |
| `src/py_rag_engine/vector_math.py` | `cosine_similarity` (numpy) |
| `scripts/demo_rerank.py` | E2E demo script |

---

## License

This repository is licensed under the MIT license. See `LICENSE`.
