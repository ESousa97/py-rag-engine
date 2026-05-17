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
│   ├── eval_document.md                 # source doc for RAGAS evaluation
│   ├── eval_questions.json              # 10 gold-standard Q&A pairs (schema v1)
│   ├── gdp_document_0.metadata.json     # demo PDF chunk metadata
│   ├── gdp_first_rows.json              # HF first-rows sample
│   └── hf_rows.json                     # Markdown dataset sample
├── docs/
│   ├── architecture.md                  # module map + pipeline diagram
│   └── evaluation.md                    # RAGAS metrics, modes, report schema
├── examples/
│   └── sample_hf.md
├── reports/                             # eval_report_<UTC>.json (gitignored)
├── scripts/
│   ├── demo_rerank.py                   # E2E demo (ingest→embed→store→rerank)
│   ├── eval_ragas.py                    # offline RAGAS CLI (thin wrapper)
│   └── process_document.py              # CLI for PDF/Markdown chunk processing
├── src/
│   └── py_rag_engine/
│       ├── __init__.py
│       ├── config.py                    # LMStudioConfig / PostgresConfig / EvalConfig
│       ├── domain.py                    # DocumentChunk + ChunkMetadata
│       ├── vector_math.py               # numpy cosine similarity
│       ├── chunking/
│       │   ├── recursive.py             # recursive splitter and dynamic overlap
│       │   └── semantic.py              # embedding-based semantic splitting
│       ├── clients/
│       │   └── lm_studio.py             # LMStudioClient + detect_chat_model
│       ├── embeddings/
│       │   ├── hashing.py               # SHA-256 content hash helpers
│       │   ├── lm_studio_embedder.py    # make_lm_studio_embed(client)
│       │   └── sentence_transformer.py  # make_sentence_transformer_embed(model)
│       ├── evaluation/
│       │   ├── dataset.py               # load_gold_standard(JSON path)
│       │   ├── metrics.py               # faithfulness / answer_relevancy / context_precision
│       │   ├── ragas_official.py        # optional official-library path with fallback
│       │   └── runner.py                # EvalRunner + build_summary
│       ├── generation/
│       │   └── lm_studio_chat.py        # generate_answer grounded in context
│       ├── ingestion/
│       │   ├── loaders.py               # PDF and Markdown loaders
│       │   └── pipeline.py              # ingest_file / ingest_path
│       ├── retrieval/
│       │   ├── rerank.py                # CrossEncoderReranker + rerank_candidates
│       │   └── service.py               # rank_chunks_by_similarity + retrieve_with_rerank
│       └── storage/
│           └── postgres.py              # PostgresEmbeddingStore (pgvector)
├── tests/
│   ├── test_chunking.py
│   ├── test_config.py                   # env loading + dataclass defaults
│   ├── test_embeddings.py
│   ├── test_evaluation.py               # dataset loader + metrics + runner helpers
│   ├── test_generation.py               # prompt assembly + chat dispatch
│   ├── test_ingestion.py
│   ├── test_lm_studio_client.py         # HTTP retry + auto-detect
│   ├── test_postgres_integration.py     # requires TEST_POSTGRES_URL + LM_STUDIO_BASE_URL
│   ├── test_postgres_storage.py
│   └── test_retrieval.py
├── pyproject.toml
└── README.md
```

Each module has one responsibility; scripts are CLI thin-wrappers that
compose the library. See [docs/architecture.md](docs/architecture.md) for
the full module map and pipeline diagram.

---

## Prerequisites

| Requirement | Version tested | Notes |
|---|---|---|
| Python | 3.11 / 3.12 / 3.14 | `>=3.10` per `pyproject.toml` |
| PostgreSQL + pgvector | pg16 + pgvector 0.8.2 | via Docker (see below) |
| LM Studio | any | OpenAI-compatible server on `localhost:1234` |
| Embedding model | `gpustack/bge-m3-GGUF` (bge-m3-Q8_0) | 1024d, multilingual |
| Chat model (for eval) | `Qwen/Qwen2.5-7B-Instruct-GGUF` Q4_K_M | ~4.7 GB, recommended for 12 GB GPUs |
| sentence-transformers | 5.4.x | Cross-Encoder re-ranking + local embeddings |
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
`pgvector/pgvector` Docker image. Set the DB password in your shell once,
then reuse it from the environment — **never commit it to a file**:

```powershell
$env:POSTGRES_PASSWORD = "<choose-a-strong-password>"
$env:POSTGRES_DB       = "rag"
$env:POSTGRES_PORT     = "5434"          # host port to expose

docker run -d `
  --name rag-pgvector `
  -e POSTGRES_PASSWORD=$env:POSTGRES_PASSWORD `
  -e POSTGRES_DB=$env:POSTGRES_DB `
  -p "$($env:POSTGRES_PORT):5432" `
  pgvector/pgvector:pg16
```

Wait for it to be ready:

```powershell
docker exec rag-pgvector pg_isready -U postgres
# output: /var/run/postgresql:5432 - accepting connections
```

The scripts read the connection settings from the environment.
You can either set a full DSN once:

```powershell
$env:EVAL_POSTGRES_URL = "postgresql+psycopg://postgres:$env:POSTGRES_PASSWORD@localhost:$env:POSTGRES_PORT/$env:POSTGRES_DB"
```

…or rely on the standard `POSTGRES_*` parts (`POSTGRES_PASSWORD` is
required; the rest fall back to `postgres` / `localhost` / `5432` / `rag`).

To stop and remove the container when done:

```powershell
docker stop rag-pgvector
docker rm rag-pgvector
```

### 2. LM Studio — Embedding Model & Chat Model

1. Download and install [LM Studio](https://lmstudio.ai).
2. Download the **embedding** model `gpustack/bge-m3-GGUF` (file `bge-m3-Q8_0.gguf`, ~600 MB).
3. Download the **chat** model used by the offline RAGAS evaluation. On a
   12 GB GPU (e.g. RTX 3060) the recommended pick is:
   - `Qwen/Qwen2.5-7B-Instruct-GGUF` → `qwen2.5-7b-instruct-q4_k_m.gguf` (~4.7 GB),
     or the equivalent `bartowski/Qwen2.5-7B-Instruct-GGUF`.
   - **Avoid** 13B+ models — they OOM when loaded alongside `bge-m3` and the
     Cross-Encoder, producing `WinError 10054` mid-eval.
4. Go to **Developer → Local Server**, load **both** models (enable JIT
   loading or pre-load manually), then click **Start Server**.
5. The server exposes `http://localhost:1234/v1/{embeddings,chat/completions,models}`.

Verify the server is up:

```powershell
Invoke-RestMethod http://localhost:1234/v1/models | Select-Object -ExpandProperty data | Format-Table id
```

Expected output includes both `text-embedding-bge-m3` and `qwen2.5-7b-instruct`.

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
$env:PYTHONPATH                = "src"
$env:LM_STUDIO_BASE_URL        = "http://localhost:1234"
$env:LM_STUDIO_EMBEDDING_MODEL = "text-embedding-bge-m3"

# Either set the full DSN…
$env:DEMO_POSTGRES_URL = "postgresql+psycopg://postgres:$env:POSTGRES_PASSWORD@localhost:5434/rag"
# …or rely on POSTGRES_PASSWORD (+ optional POSTGRES_USER/HOST/PORT/DB) set
# earlier in the shell. The script never embeds credentials.
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
postgres_url      = postgresql+psycopg://postgres:***@localhost:5434/rag
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

## Offline RAGAS Evaluation

[scripts/eval_ragas.py](scripts/eval_ragas.py) runs the full pipeline against a
10-question gold-standard set and writes a JSON report comparing chunk sizes
and embedding models. Three RAGAS metrics are computed per question:

- **Faithfulness** — fraction of answer claims entailed by the retrieved context
- **Answer Relevancy** — cosine sim between the question and a reverse-generated
  question derived from the answer
- **Context Precision** — fraction of retrieved chunks judged useful by the LLM

The metrics follow the definitions in the original RAGAS paper. By default the
script uses a manual implementation that talks to LM Studio over plain HTTP —
this avoids the `n>1` and tool-input limitations of the `langchain-openai`
stack. Set `EVAL_USE_OFFICIAL_RAGAS=1` to try the official `ragas` library
first (the script falls back automatically if it fails).

### Setup

Install the eval extra and start the LM Studio server with **both** models
loaded (see "LM Studio — Embedding Model & Chat Model" above):

```powershell
pip install -e ".[eval]"
docker start rag-pgvector
```

### Run modes

| Mode | Env var | Configs × Questions | Wall time¹ |
|---|---|---|---|
| Smoke (sanity check) | `EVAL_SMOKE=1` | 1 × 1 | ~30 s |
| Quick (single model, 3 Qs) | `EVAL_QUICK=1 EVAL_SKIP_MINILM=1` | 3 × 3 | ~3 min |
| Full single-model | `EVAL_SKIP_MINILM=1` | 3 × 10 | ~8 min |
| Full comparison | _(no flags)_ | 6 × 10 | ~15 min |

¹ Qwen2.5-7B-Instruct Q4_K_M on RTX 3060 12 GB.

### Run (PowerShell)

```powershell
$env:PYTHONPATH        = "src"

# Provide credentials via the standard POSTGRES_* parts (or a full DSN in
# $env:EVAL_POSTGRES_URL). No credential lives in the code or the report.
$env:POSTGRES_PASSWORD = "<your-password>"
$env:POSTGRES_PORT     = "5434"        # match the docker run above

# Smoke test first
$env:EVAL_SMOKE = "1"
python scripts\eval_ragas.py

# Then a full run
Remove-Item Env:\EVAL_SMOKE
python scripts\eval_ragas.py
```

The JSON report lands in `reports/eval_report_<UTC-timestamp>.json` with
per-question scores, per-config averages, and an `overall_ranking` block that
ranks configurations by mean score across all three metrics.

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
from py_rag_engine.embeddings import make_sentence_transformer_embed
from py_rag_engine.ingestion import ingest_file
embed = make_sentence_transformer_embed('all-MiniLM-L6-v2')
chunks = ingest_file('data/eval_document.md', use_semantic_chunking=True, embed=embed)
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
from py_rag_engine.config import PostgresConfig
from py_rag_engine.storage import EmbeddingInput, PostgresEmbeddingStore

# Pulls the DSN from EVAL_POSTGRES_URL or the POSTGRES_* env parts.
# Raises RuntimeError if no credentials are available — never hardcoded.
engine = create_engine(PostgresConfig.from_env().url)
store  = PostgresEmbeddingStore(engine, embedding_model="bge-m3")
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
| `all-MiniLM-L6-v2` / `all-minilm-l6-v2` | 384 |

### LM Studio Client

```python
from py_rag_engine.clients import LMStudioClient, detect_chat_model
from py_rag_engine.config import LMStudioConfig

client = LMStudioClient(LMStudioConfig.from_env())     # reads LM_STUDIO_* env vars

# Sanity: list models and pick a 7-8B chat model
chat = detect_chat_model(client)
print(chat)                                            # e.g. "qwen2.5-7b-instruct"

# Warm the LLM into VRAM before a long eval loop
client.warm_up_chat(chat)

# Embed
vectors = client.embed(["hello", "world"])             # 64-batch by default

# Chat
answer = client.chat(
    [{"role": "user", "content": "Reply OK."}],
    model=chat, temperature=0.0, max_tokens=5,
)
```

Retries (`OSError`, `WinError 10054`, malformed JSON) and exponential backoff
are tuned via `LMStudioConfig.retries` and `LMStudioConfig.backoff`. Inject a
fake `http_json` argument for tests.

### Grounded Answer Generation

```python
from py_rag_engine.generation import generate_answer

answer = generate_answer(
    client,
    question="What is HNSW?",
    contexts=["[1] HNSW is a graph-based ANN algorithm.", "[2] Its key knobs are m, ef_construction, ef_search."],
    chat_model="qwen2.5-7b-instruct",
)
```

### Offline Evaluation

```python
from sqlalchemy import create_engine
from py_rag_engine.config import LMStudioConfig, PostgresConfig
from py_rag_engine.clients import LMStudioClient
from py_rag_engine.embeddings import make_lm_studio_embed
from py_rag_engine.evaluation import EvalRunner, build_summary, load_gold_standard

client = LMStudioClient(LMStudioConfig.from_env())
engine = create_engine(PostgresConfig.from_env().url)
gold   = load_gold_standard("data/eval_questions.json")

runner = EvalRunner(
    client=client,
    config=client.config,
    engine=engine,
    chat_model="qwen2.5-7b-instruct",
)

result = runner.run(
    config_id="bge-m3_chunk1024",
    embed_label="BGE-M3 via LM Studio",
    embedding_model="bge-m3",
    chunk_size=1024,
    embed_fn=make_lm_studio_embed(client),
    document_path="data/eval_document.md",
    questions=gold,
)
print(result.metrics)
# {'faithfulness': 0.82, 'answer_relevancy': 0.79, 'context_precision': 0.85, 'mean_retrieval_similarity': 0.61}
```

See [docs/evaluation.md](docs/evaluation.md) for the full metric definitions,
run modes, and report schema.

### Re-ranking Pipeline

```python
import os
from pathlib import Path

from sqlalchemy import create_engine

from py_rag_engine.config import PostgresConfig
from py_rag_engine.retrieval import CrossEncoderReranker, retrieve_with_rerank
from py_rag_engine.storage import PostgresEmbeddingStore

engine = create_engine(PostgresConfig.from_env().url)
store  = PostgresEmbeddingStore(engine, embedding_model="bge-m3")

# Model name (downloaded by sentence-transformers) or a path from env.
# Avoid hardcoding absolute paths — they aren't portable.
reranker = CrossEncoderReranker(
    model_name=os.environ.get(
        "RERANKER_MODEL_PATH",
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ),
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
49 passed, 1 skipped
```

The skipped test is `tests/test_postgres_integration.py`, which requires a
real Postgres instance and LM Studio.

### Integration test (full round-trip)

Set environment variables pointing to a running Postgres and LM Studio, then
run the full suite. The password comes from the shell — never the source tree:

```powershell
$env:POSTGRES_PASSWORD         = "<your-password>"
$env:TEST_POSTGRES_URL         = "postgresql+psycopg://postgres:$env:POSTGRES_PASSWORD@localhost:5434/rag"
$env:LM_STUDIO_BASE_URL        = "http://localhost:1234"
$env:LM_STUDIO_EMBEDDING_MODEL = "text-embedding-bge-m3"
$env:STORAGE_EMBEDDING_MODEL   = "bge-m3"
python -m pytest -q
```

```text
50 passed
```

The integration test (`test_lm_studio_embeddings_round_trip_through_postgres_pgvector`)
embeds three sentences, inserts them into the `embeddings` table, runs a
similarity query, and asserts that the HNSW index is present.

---

## Development

Library modules grouped by responsibility:

| Module | Purpose |
|---|---|
| `config.py` | `LMStudioConfig` / `PostgresConfig` / `EvalConfig` (env-driven) |
| `clients/lm_studio.py` | `LMStudioClient` HTTP wrapper + `detect_chat_model` |
| `domain.py` | `DocumentChunk`, `ChunkMetadata` |
| `vector_math.py` | `cosine_similarity` (numpy) |
| `ingestion/pipeline.py` | Ingest orchestration (`ingest_file`, `ingest_path`) |
| `ingestion/loaders.py` | PDF and Markdown loaders |
| `chunking/recursive.py` | Recursive splitter and dynamic overlap |
| `chunking/semantic.py` | Embedding-based topic splitting |
| `embeddings/hashing.py` | SHA-256 content hashing |
| `embeddings/lm_studio_embedder.py` | `make_lm_studio_embed(client)` |
| `embeddings/sentence_transformer.py` | `make_sentence_transformer_embed(model)` |
| `storage/postgres.py` | `PostgresEmbeddingStore`, pgvector HNSW |
| `retrieval/rerank.py` | `CrossEncoderReranker`, `rerank_candidates` |
| `retrieval/service.py` | `retrieve_with_rerank` pipeline |
| `generation/lm_studio_chat.py` | `generate_answer` grounded in context |
| `evaluation/dataset.py` | `load_gold_standard(JSON)` |
| `evaluation/metrics.py` | Faithfulness / answer_relevancy / context_precision |
| `evaluation/ragas_official.py` | Optional official-library adapter |
| `evaluation/runner.py` | `EvalRunner`, `build_summary`, `safe_table_name` |
| `scripts/demo_rerank.py` | E2E demo CLI |
| `scripts/eval_ragas.py` | Offline RAGAS CLI |
| `scripts/process_document.py` | Standalone chunking CLI |

---

## License

This repository is licensed under the MIT license. See `LICENSE`.
