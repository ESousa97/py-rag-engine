from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from py_rag_engine.clients import LMStudioClient
from py_rag_engine.config import LMStudioConfig
from py_rag_engine.evaluation import (
    EvalSample,
    build_summary,
    evaluate_samples,
    load_gold_standard,
)
from py_rag_engine.evaluation.runner import ConfigResult, safe_table_name


# ── Dataset loader ──────────────────────────────────────────────────────────


def test_load_gold_standard_happy_path(tmp_path: Path):
    payload = {
        "schema_version": 1,
        "questions": [
            {"id": "a", "question": "Q1", "ground_truth": "A1"},
            {"id": "b", "question": "Q2", "ground_truth": "A2"},
        ],
    }
    p = tmp_path / "qs.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    gold = load_gold_standard(p)
    assert len(gold) == 2
    assert gold[0].id == "a"
    assert gold[1].question == "Q2"


def test_load_gold_standard_rejects_unsupported_schema(tmp_path: Path):
    p = tmp_path / "qs.json"
    p.write_text(json.dumps({"schema_version": 99, "questions": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported schema_version"):
        load_gold_standard(p)


def test_load_gold_standard_detects_duplicate_ids(tmp_path: Path):
    payload = {
        "schema_version": 1,
        "questions": [
            {"id": "a", "question": "Q1", "ground_truth": "A1"},
            {"id": "a", "question": "Q2", "ground_truth": "A2"},
        ],
    }
    p = tmp_path / "qs.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_gold_standard(p)


def test_loaded_real_gold_has_ten_questions():
    project_root = Path(__file__).resolve().parents[1]
    gold = load_gold_standard(project_root / "data" / "eval_questions.json")
    assert len(gold) == 10
    assert {q.id for q in gold} >= {"rag-definition", "faithfulness", "hnsw"}


# ── Metrics with fake LLM ───────────────────────────────────────────────────


def _fake_chat_client(*replies: str) -> LMStudioClient:
    queue = list(replies)

    def fake_http(url: str, payload: Any = None, *, timeout: int | None = None) -> Any:
        if not queue:
            raise AssertionError(f"unexpected call to {url}")
        return {"choices": [{"message": {"content": queue.pop(0)}}]}

    return LMStudioClient(LMStudioConfig(chat_model="dummy"), http_json=fake_http)


def test_evaluate_samples_parses_scores():
    client = _fake_chat_client(
        '{"score": 0.8}',         # faithfulness
        "What is RAG?",           # reverse-question for ans_rel
        '{"score": 0.9}',         # context_precision
    )
    embeds = [[1.0, 0.0], [1.0, 0.0]]                            # cosine = 1.0

    def fake_embed(_: list[str]) -> list[list[float]]:
        return embeds

    sample = EvalSample(
        question="Q",
        answer="A",
        contexts=["c1", "c2"],
        ground_truth="GT",
    )
    out = evaluate_samples(client, fake_embed, [sample])
    assert out[0].faithfulness == 0.8
    assert out[0].context_precision == 0.9
    assert out[0].answer_relevancy == 1.0


def test_evaluate_samples_handles_garbled_score():
    client = _fake_chat_client(
        "I cannot answer.",                # faithfulness — no JSON
        "What is X?",                       # reverse-question — embed fails
        '{"score": "not a number"}',        # context_precision — bad JSON
    )

    def fake_embed(_: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding offline")

    sample = EvalSample(question="Q", answer="A", contexts=["c"], ground_truth="GT")
    out = evaluate_samples(client, fake_embed, [sample])
    assert out[0].faithfulness is None
    assert out[0].answer_relevancy is None
    assert out[0].context_precision is None


# ── Runner helpers ──────────────────────────────────────────────────────────


def test_safe_table_name_normalises():
    assert safe_table_name("BGE-M3!", 512) == "eval_bge_m3_512"
    assert safe_table_name("all-MiniLM-L6-v2", 1024) == "eval_all_minilm_l6_v2_1024"


def test_build_summary_ranks_by_average():
    results = [
        ConfigResult(
            id="a", embedding_model="m", embedding_label="m", chunk_size=512,
            num_chunks=10, embed_time_sec=1.0,
            metrics={"faithfulness": 0.6, "answer_relevancy": 0.6,
                     "context_precision": 0.6, "mean_retrieval_similarity": 0.6},
            per_question=[],
        ),
        ConfigResult(
            id="b", embedding_model="m", embedding_label="m", chunk_size=1024,
            num_chunks=10, embed_time_sec=1.0,
            metrics={"faithfulness": 0.9, "answer_relevancy": 0.9,
                     "context_precision": 0.9, "mean_retrieval_similarity": 0.9},
            per_question=[],
        ),
    ]
    summary = build_summary(results)
    assert summary["overall_ranking"][0]["config"] == "b"
    assert summary["best_faithfulness"]["config"] == "b"
