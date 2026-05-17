"""End-to-end RAGAS evaluation runner.

Wires every library piece together for one configuration:

    Document  ─► ingest_file (chunk_size)
        │
        ▼
    Chunks    ─► embed (LM Studio or SentenceTransformer)
        │
        ▼
    Vectors   ─► PostgresEmbeddingStore.add_embeddings (isolated table per config)
        │
        ▼
    Question  ─► similarity_search → contexts
        │
        ▼
    Contexts  ─► generate_answer (chat completions)
        │
        ▼
    Answer    ─► evaluate_samples (faithfulness / answer_relevancy / context_precision)
"""
from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import text as sql_text
from sqlalchemy.engine import Engine

from py_rag_engine.clients import LMStudioClient
from py_rag_engine.config import LMStudioConfig
from py_rag_engine.embeddings import EmbedFn
from py_rag_engine.evaluation.dataset import GoldQuestion
from py_rag_engine.evaluation.metrics import (
    EvalSample,
    MetricScores,
    evaluate_samples,
)
from py_rag_engine.evaluation.ragas_official import try_official_ragas
from py_rag_engine.generation import generate_answer
from py_rag_engine.ingestion import ingest_file
from py_rag_engine.storage import EmbeddingInput, PostgresEmbeddingStore


@dataclass(frozen=True, slots=True)
class ConfigResult:
    """Aggregate metrics + per-question detail for one configuration."""

    id: str
    embedding_model: str
    embedding_label: str
    chunk_size: int
    num_chunks: int
    embed_time_sec: float
    metrics: dict[str, float | None]
    per_question: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id":              self.id,
            "embedding_model": self.embedding_model,
            "embedding_label": self.embedding_label,
            "chunk_size":      self.chunk_size,
            "num_chunks":      self.num_chunks,
            "embed_time_sec":  self.embed_time_sec,
            "metrics":         self.metrics,
            "per_question":    self.per_question,
        }


def safe_table_name(model_name: str, chunk_size: int) -> str:
    """Build a PostgreSQL-safe table name like `eval_bge_m3_512`."""
    safe = re.sub(r"[^a-z0-9]+", "_", model_name.lower()).strip("_")
    return f"eval_{safe}_{chunk_size}"


@dataclass(slots=True)
class EvalRunner:
    """Runs the eval pipeline for a single (model, chunk_size) pair."""

    client: LMStudioClient
    config: LMStudioConfig
    engine: Engine
    chat_model: str
    top_k: int = 5
    ef_search: int = 80
    use_official_ragas: bool = False

    # Cached embedding doc, so we don't re-ingest the same file for each chunk
    # size when the caller invokes multiple configs in a row.
    _last_ingest_key: tuple[str, int] | None = field(default=None, init=False)
    _last_ingest_value: list = field(default_factory=list, init=False)

    # ── Public API ───────────────────────────────────────────────────────────

    def run(
        self,
        *,
        config_id: str,
        embed_label: str,
        embedding_model: str,
        chunk_size: int,
        embed_fn: EmbedFn,
        document_path: Path,
        questions: Sequence[GoldQuestion],
    ) -> ConfigResult:
        print(f"\n{'─'*70}")
        print(f"Config : {config_id}")
        print(f"  model : {embed_label}")
        table = safe_table_name(embedding_model, chunk_size)
        print(f"  chunk : {chunk_size} chars  |  table: {table}")

        chunks = self._ingest(document_path, chunk_size)
        vectors, embed_time = self._embed(chunks, embed_fn)
        store = self._persist(chunks, vectors, embedding_model, table)
        samples = self._retrieve_and_generate(store, embed_fn, questions)
        scores = self._score(samples)
        return self._summarise(
            config_id=config_id,
            embed_label=embed_label,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            num_chunks=len(chunks),
            embed_time=embed_time,
            samples=samples,
            scores=scores,
        )

    # ── Pipeline steps ───────────────────────────────────────────────────────

    def _ingest(self, document_path: Path, chunk_size: int) -> list:
        key = (str(document_path), chunk_size)
        if self._last_ingest_key == key:
            print("  [1/5] Ingesting document… (cached)")
            return self._last_ingest_value
        print("  [1/5] Ingesting document…")
        chunks = ingest_file(document_path, chunk_size=chunk_size)
        print(f"        {len(chunks)} chunks produced")
        self._last_ingest_key = key
        self._last_ingest_value = chunks
        return chunks

    def _embed(self, chunks: list, embed_fn: EmbedFn) -> tuple[list[list[float]], float]:
        print("  [2/5] Embedding chunks…")
        t0      = time.perf_counter()
        vectors = embed_fn([c.text for c in chunks])
        t_embed = time.perf_counter() - t0
        print(f"        {len(vectors)} vectors  dim={len(vectors[0])}  {t_embed:.1f}s")
        return vectors, t_embed

    def _persist(
        self,
        chunks: list,
        vectors: list[list[float]],
        embedding_model: str,
        table: str,
    ) -> PostgresEmbeddingStore:
        print("  [3/5] Storing in PostgreSQL…")
        with self.engine.begin() as conn:
            conn.execute(sql_text(f"DROP TABLE IF EXISTS {table} CASCADE"))
        store = PostgresEmbeddingStore(
            self.engine, embedding_model=embedding_model, table_name=table,
        )
        store.create_schema()
        store.add_embeddings([
            EmbeddingInput(
                text=chunk.text,
                embedding=vec,
                content_hash=chunk.content_hash,
                metadata={
                    "source": chunk.metadata.source,
                    "page": chunk.metadata.page,
                    "chunk_index": chunk.metadata.chunk_index,
                },
                embedding_model=embedding_model,
            )
            for chunk, vec in zip(chunks, vectors, strict=True)
        ])
        print(f"        {len(chunks)} rows stored")
        return store

    def _retrieve_and_generate(
        self,
        store: PostgresEmbeddingStore,
        embed_fn: EmbedFn,
        questions: Sequence[GoldQuestion],
    ) -> list[tuple[EvalSample, float]]:
        print("  [4/5] Retrieving contexts & generating answers…")
        samples: list[tuple[EvalSample, float]] = []
        for i, qa in enumerate(questions, 1):
            q_vec   = embed_fn([qa.question])[0]
            results = store.similarity_search(q_vec, top_k=self.top_k, ef_search=self.ef_search)
            contexts = [r.text for r in results]
            mean_sim = (sum(r.cosine_similarity for r in results) / len(results)) if results else 0.0
            answer   = generate_answer(self.client, qa.question, contexts, chat_model=self.chat_model)
            print(f"        Q{i:02d}: {qa.question[:60]}…")
            sample = EvalSample(
                question=qa.question,
                answer=answer,
                contexts=contexts,
                ground_truth=qa.ground_truth,
            )
            samples.append((sample, round(mean_sim, 4)))
        return samples

    def _score(
        self,
        samples: list[tuple[EvalSample, float]],
    ) -> list[MetricScores]:
        print("  [5/5] Running RAGAS metrics (may take several minutes)…")
        sample_objs = [s for s, _ in samples]
        if self.use_official_ragas:
            print("  [info] EVAL_USE_OFFICIAL_RAGAS=1 — trying official ragas first…")
            official = try_official_ragas(sample_objs, config=self.config, chat_model=self.chat_model)
            if official is not None:
                return official
        return evaluate_samples(self.client, self.client.embed, sample_objs, chat_model=self.chat_model)

    def _summarise(
        self,
        *,
        config_id: str,
        embed_label: str,
        embedding_model: str,
        chunk_size: int,
        num_chunks: int,
        embed_time: float,
        samples: list[tuple[EvalSample, float]],
        scores: list[MetricScores],
    ) -> ConfigResult:
        per_q = [
            {
                "question":               sample.question,
                "answer":                  sample.answer,
                "mean_cosine_similarity":  mean_sim,
                "faithfulness":            sc.faithfulness,
                "answer_relevancy":        sc.answer_relevancy,
                "context_precision":       sc.context_precision,
            }
            for (sample, mean_sim), sc in zip(samples, scores)
        ]

        def avg(key: str) -> float | None:
            vals = [p[key] for p in per_q if p[key] is not None]
            return round(sum(vals) / len(vals), 4) if vals else None

        return ConfigResult(
            id=config_id,
            embedding_model=embedding_model,
            embedding_label=embed_label,
            chunk_size=chunk_size,
            num_chunks=num_chunks,
            embed_time_sec=round(embed_time, 2),
            metrics={
                "faithfulness":              avg("faithfulness"),
                "answer_relevancy":          avg("answer_relevancy"),
                "context_precision":         avg("context_precision"),
                "mean_retrieval_similarity": avg("mean_cosine_similarity"),
            },
            per_question=per_q,
        )


# ── Report summary ──────────────────────────────────────────────────────────


def build_summary(configs: list[ConfigResult]) -> dict[str, Any]:
    """Aggregate per-config metrics into the report `summary` block."""
    def best(metric: str) -> dict | None:
        valid = [(c.metrics.get(metric), c.id) for c in configs
                 if c.metrics.get(metric) is not None]
        if not valid:
            return None
        score, cid = max(valid)
        return {"config": cid, "score": score}

    def avg_score(c: ConfigResult) -> float:
        vals = [v for v in c.metrics.values() if v is not None]
        return sum(vals) / len(vals) if vals else 0.0

    ranked = sorted(configs, key=avg_score, reverse=True)
    return {
        "best_faithfulness":         best("faithfulness"),
        "best_answer_relevancy":     best("answer_relevancy"),
        "best_context_precision":    best("context_precision"),
        "best_retrieval_similarity": best("mean_retrieval_similarity"),
        "overall_ranking": [
            {"rank": i + 1, "config": c.id, "avg_score": round(avg_score(c), 4)}
            for i, c in enumerate(ranked)
        ],
    }
