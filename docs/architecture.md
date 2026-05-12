# Architecture

This project follows a `src` layout with focused modules:

- `py_rag_engine.domain`: shared domain entities (`DocumentChunk`, `ChunkMetadata`).
- `py_rag_engine.ingestion`: file loading and ingestion orchestration.
- `py_rag_engine.chunking`: recursive and semantic chunking strategies.
- `py_rag_engine.embeddings`: hashing and embedding-related helpers.
- `py_rag_engine.retrieval`: semantic similarity and ranking helpers.

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
