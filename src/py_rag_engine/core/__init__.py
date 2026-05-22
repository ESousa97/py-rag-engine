"""Core orchestration layer for the RAG engine.

The `core` package hosts pipeline-spanning components that sit *above* the
single-purpose modules (retrieval, embeddings, storage). Today it ships the
query-ingestion router; future additions (caching, guardrails, telemetry) will
live here too.
"""
