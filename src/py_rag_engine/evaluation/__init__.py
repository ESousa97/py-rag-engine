"""Offline RAGAS-style evaluation primitives.

Public API:
  - `load_gold_standard`          load Q&A pairs from JSON
  - `evaluate_samples`            score a list of samples (faithfulness, etc.)
  - `try_official_ragas`          optional official-library path with fallback
  - `EvalRunner`                  orchestrates ingest→embed→store→retrieve→generate→score
  - `build_summary`               aggregate per-config metrics for the report
"""
from py_rag_engine.evaluation.dataset import GoldQuestion, load_gold_standard
from py_rag_engine.evaluation.metrics import EvalSample, MetricScores, evaluate_samples
from py_rag_engine.evaluation.ragas_official import try_official_ragas
from py_rag_engine.evaluation.runner import (
    ConfigResult,
    EvalRunner,
    build_summary,
    safe_table_name,
)

__all__ = [
    "ConfigResult",
    "EvalRunner",
    "EvalSample",
    "GoldQuestion",
    "MetricScores",
    "build_summary",
    "evaluate_samples",
    "load_gold_standard",
    "safe_table_name",
    "try_official_ragas",
]
