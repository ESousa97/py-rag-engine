# py-rag-engine

A Python RAG engine focused on document ingestion for PDF and Markdown files. It
loads documents, extracts text, applies recursive chunking with dynamic overlap,
optionally detects semantic topic breaks with embeddings, and emits deduplicated
chunks ready for indexing.

## Features

- PDF, `.md`, and `.markdown` ingestion.
- PDF page extraction with `pypdf`.
- Recursive chunking through `RecursiveCharacterTextSplitter`.
- Dynamic overlap based on the configured chunk size.
- Optional semantic paragraph chunking with user-provided embeddings.
- SHA-256 content hashes for duplicate detection.
- JSON-ready output with `text`, `metadata`, and `content_hash`.

## Project Layout

```text
PY-RAG-ENGINE/
├── data/
│   ├── gdp_document_0.metadata.json # generated metadata for practical review
│   ├── gdp_first_rows.json          # Hugging Face first-rows response
│   └── hf_rows.json                 # Markdown dataset sample
├── docs/
│   └── architecture.md
├── examples/
│   └── sample_hf.md
├── scripts/
│   └── process_document.py          # local CLI for PDF/Markdown processing
├── src/
│   └── py_rag_engine/
│       ├── chunking/
│       │   ├── recursive.py         # recursive splitter and dynamic overlap
│       │   └── semantic.py          # embedding-based semantic splitting
│       ├── embeddings/
│       │   └── hashing.py           # content hash helpers
│       ├── ingestion/
│       │   ├── loaders.py           # PDF and Markdown loaders
│       │   └── pipeline.py          # ingestion orchestration
│       ├── retrieval/
│       ├── domain.py                # DocumentChunk and ChunkMetadata
│       └── vector_math.py
├── tests/
│   ├── test_chunking.py
│   ├── test_embeddings.py
│   ├── test_ingestion.py
│   └── test_retrieval.py
├── pyproject.toml
└── README.md
```

Generated document files and full processing outputs are intentionally kept out
of Git:

- `data/*.pdf`
- `results/`
- `outputs/`

Metadata JSON files under `data/` are kept versionable so the team can review
practical processing results without committing large documents or full chunk
payloads.

## Installation

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
pip install pytest
```

Install optional embedding support when semantic chunking should use
`sentence-transformers`:

```powershell
pip install -e ".[embeddings]"
```

## Download the Test PDF

The Hugging Face endpoint returns dataset rows with signed PDF URLs:

```powershell
Invoke-WebRequest `
  -Uri "https://datasets-server.huggingface.co/first-rows?dataset=surgeai%2FGDP.pdf&config=default&split=train" `
  -OutFile "data\gdp_first_rows.json"
```

Download the first PDF referenced by the response:

```powershell
$rows = Get-Content -Raw data\gdp_first_rows.json | ConvertFrom-Json
$url = $rows.rows[0].row.pdf.src
Invoke-WebRequest -Uri $url -OutFile "data\gdp_document_0.pdf"
```

## Usage

### Process a PDF

```powershell
$env:PYTHONPATH='src'
python scripts\process_document.py `
  data\gdp_document_0.pdf `
  --output results\gdp_chunks.json `
  --metadata-output data\gdp_document_0.metadata.json
```

Expected output for the current test PDF:

```text
chunks=50
output=results\gdp_chunks.json
metadata_output=data\gdp_document_0.metadata.json
```

`results\gdp_chunks.json` contains the full chunk list, including text. It is
ignored by Git because it can become large.

`data\gdp_document_0.metadata.json` contains lightweight metadata suitable for
team review:

```json
{
  "chunk_count": 50,
  "unique_hash_count": 50,
  "sources": ["data\\gdp_document_0.pdf"],
  "pages": [1, 2, 3],
  "chunks": [
    {
      "content_hash": "8276b4d156b68d5ced56231850149a1e8ded0133bfab80375275e9702e45dbcb",
      "source": "data\\gdp_document_0.pdf",
      "page": 1,
      "chunk_index": 0,
      "text_chars": 234
    }
  ]
}
```

### Inspect the Full Chunk Output

```powershell
$env:PYTHONPATH='src'
Get-Content -Raw results\gdp_chunks.json
```

Each chunk has this shape:

```json
[
  {
    "text": "GEOTECHNICAL INVESTIGATION REPORT ...",
    "metadata": {
      "source": "C:\\Users\\...\\data\\gdp_document_0.pdf",
      "page": 1,
      "chunk_index": 0,
      "extra": {}
    },
    "content_hash": "8276b4d156b68d5ced56231850149a1e8ded0133bfab80375275e9702e45dbcb"
  }
]
```

### Process Markdown

```powershell
$env:PYTHONPATH='src'
python scripts\process_document.py `
  examples\sample_hf.md `
  --output results\sample_hf_chunks.json `
  --metadata-output data\sample_hf.metadata.json
```

For Markdown documents, `metadata.page` is `null` because the source format has
no page boundaries.

### Quick Count Without Writing Files

```powershell
$env:PYTHONPATH='src'
python -c "from py_rag_engine.ingestion import ingest_file; chunks = ingest_file('data/gdp_document_0.pdf'); print(len(chunks))"
```

This command prints only the number of generated chunks.

### Semantic Chunking

Semantic chunking requires an embedding function. With `sentence-transformers`:

```powershell
$env:PYTHONPATH='src'
python -c "from py_rag_engine.ingestion import ingest_file, make_sentence_transformer_embed; embed = make_sentence_transformer_embed('all-MiniLM-L6-v2'); chunks = ingest_file('examples/sample_hf.md', use_semantic_chunking=True, embed=embed); print(len(chunks))"
```

Useful parameters:

- `chunk_size`: target chunk size, default `1200`.
- `chunk_overlap`: fixed overlap. When omitted, dynamic overlap is used.
- `overlap_ratio`: dynamic overlap ratio, default `0.12`.
- `use_semantic_chunking`: enables embedding-based topic break detection.
- `semantic_similarity_threshold`: paragraph similarity threshold.
- `deduplicate_by_hash`: removes duplicated chunks, default `True`.

## Main API

```python
from py_rag_engine.ingestion import chunks_to_dicts, ingest_file

chunks = ingest_file("data/gdp_document_0.pdf")
payload = chunks_to_dicts(chunks)
```

`payload` is a list of dictionaries:

```python
[
    {
        "text": "...",
        "metadata": {
            "source": "...",
            "page": 1,
            "chunk_index": 0,
            "extra": {},
        },
        "content_hash": "...",
    }
]
```

## Testing

Run the full test suite:

```powershell
python -m pytest -q
```

Expected result in the current project state:

```text
12 passed
```

Run a quick PDF ingestion check:

```powershell
$env:PYTHONPATH='src'
python -c "from py_rag_engine.ingestion import ingest_file; chunks = ingest_file('data/gdp_document_0.pdf'); print(len(chunks), len({c.content_hash for c in chunks}))"
```

The first number is the total chunk count. The second number is the number of
unique content hashes.

## Development

- Source code: `src/py_rag_engine/`.
- Tests: `tests/`.
- Domain models: `src/py_rag_engine/domain.py`.
- Ingestion pipeline: `src/py_rag_engine/ingestion/pipeline.py`.
- Document loaders: `src/py_rag_engine/ingestion/loaders.py`.
- Recursive chunking: `src/py_rag_engine/chunking/recursive.py`.
- Semantic chunking: `src/py_rag_engine/chunking/semantic.py`.

## License

This repository is licensed under the MIT license. See `LICENSE`.
