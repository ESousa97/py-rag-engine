"""Optional adapter for the official `ragas` library.

The library expects a LangChain-wrapped LLM and embeddings. Because LM Studio
rejects `n>1` sampling and certain non-string tool inputs, the integration is
fragile — callers should use `try_official_ragas` and fall back to the manual
implementation in `metrics.py` on any failure.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from py_rag_engine.config import LMStudioConfig
from py_rag_engine.evaluation.metrics import EvalSample, MetricScores


def _coerce_score(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        if f != f:                # NaN
            return None
        return round(max(0.0, min(1.0, f)), 4)
    except (TypeError, ValueError):
        return None


def try_official_ragas(
    samples: Sequence[EvalSample],
    *,
    config: LMStudioConfig,
    chat_model: str,
) -> list[MetricScores] | None:
    """Try the official `ragas` library; return None if it's unavailable or fails."""
    try:
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from ragas import evaluate as ragas_evaluate
        from ragas.dataset_schema import EvaluationDataset, SingleTurnSample
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.metrics import AnswerRelevancy, ContextPrecision, Faithfulness
    except ImportError as exc:
        print(f"  [info] official ragas not available ({exc}); using manual path.")
        return None

    try:
        llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=chat_model,
                base_url=f"{config.base_url}/v1",
                api_key="lm-studio",
                temperature=0.0,
                max_retries=2,
                timeout=180,
            )
        )
        embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model=config.embed_model,
                base_url=f"{config.base_url}/v1",
                api_key="lm-studio",
            )
        )
        dataset = EvaluationDataset(samples=[
            SingleTurnSample(
                user_input=s.question,
                response=s.answer,
                retrieved_contexts=list(s.contexts),
                reference=s.ground_truth,
            )
            for s in samples
        ])
        # strictness=1 disables the n>1 sampling that LM Studio rejects.
        result = ragas_evaluate(
            dataset,
            metrics=[
                Faithfulness(),
                AnswerRelevancy(strictness=1),
                ContextPrecision(),
            ],
            llm=llm,
            embeddings=embeddings,
            show_progress=False,
            raise_exceptions=False,
        )
        rows = result.to_pandas().to_dict(orient="records")
        return [
            MetricScores(
                faithfulness=_coerce_score(r.get("faithfulness")),
                answer_relevancy=_coerce_score(r.get("answer_relevancy")),
                context_precision=_coerce_score(r.get("context_precision")),
            )
            for r in rows
        ]
    except Exception as exc:
        print(f"  [warn] official ragas raised {type(exc).__name__}: {exc}")
        print("         Falling back to manual implementation.")
        return None
