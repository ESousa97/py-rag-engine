# Architecture

This project follows a `src` layout with focused modules:

- `py_rag_engine.domain`: shared domain entities (`DocumentChunk`, `ChunkMetadata`).
- `py_rag_engine.ingestion`: file loading and ingestion orchestration.
- `py_rag_engine.chunking`: recursive and semantic chunking strategies.
- `py_rag_engine.embeddings`: hashing and embedding-related helpers.
- `py_rag_engine.retrieval`: semantic similarity and ranking helpers.

Data and examples are separated from library code:

- `examples/sample_hf.md`
- `data/hf_rows.json`
