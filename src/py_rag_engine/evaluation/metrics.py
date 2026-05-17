"""RAGAS-style metrics computed through an LM Studio judge.

Three metrics following the original RAGAS paper:
  - **Faithfulness**       fraction of answer claims entailed by the context
  - **Answer Relevancy**   cosine similarity between the user's question and a
                           question reverse-generated from the answer
  - **Context Precision**  fraction of retrieved chunks judged useful

This module is independent of any specific transport — it works against any
`LMStudioClient` and any `embed(texts)` callable. Score values are clamped to
[0, 1] and rounded to 4 decimals; failures return `None` with a printed cause
so the caller can distinguish "model unreachable" from "model output garbled".
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from py_rag_engine.clients import LMStudioClient

EmbedFn = Callable[[list[str]], list[list[float]]]


@dataclass(frozen=True, slots=True)
class EvalSample:
    """One Q&A pair under evaluation, with its retrieved context and answer."""

    question: str
    answer: str
    contexts: list[str]
    ground_truth: str


@dataclass(frozen=True, slots=True)
class MetricScores:
    faithfulness: float | None
    answer_relevancy: float | None
    context_precision: float | None


# ── LLM-as-judge primitives ──────────────────────────────────────────────────


def _llm_json_score(
    client: LMStudioClient,
    prompt: str,
    *,
    chat_model: str | None = None,
    label: str = "score",
) -> float | None:
    """Run a judge prompt and parse `{"score": <float 0-1>}` from the reply."""
    try:
        raw = client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a strict RAG evaluator. Respond with a single JSON "
                        'object: {"score": <float between 0 and 1>}. '
                        "No commentary, no markdown, just the JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model=chat_model,
            temperature=0.0,
            max_tokens=80,
        )
    except Exception as exc:
        print(f"        [{label} http-error] {type(exc).__name__}: {exc}")
        return None

    match = re.search(r"\{[^{}]*\}", raw)
    if not match:
        print(f"        [{label} parse-error] no JSON object in: {raw[:120]!r}")
        return None
    try:
        obj = json.loads(match.group(0))
        v = float(obj.get("score", float("nan")))
        if v != v:                # NaN
            print(f"        [{label} parse-error] score is NaN in: {raw[:120]!r}")
            return None
        return round(max(0.0, min(1.0, v)), 4)
    except (ValueError, json.JSONDecodeError) as exc:
        print(f"        [{label} parse-error] {exc}: {raw[:120]!r}")
        return None


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = math.sqrt(sum(x * x for x in a))
    nb  = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def _format_context_block(contexts: Sequence[str]) -> str:
    return "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))


# ── Per-metric scorers ───────────────────────────────────────────────────────


def faithfulness(
    client: LMStudioClient, sample: EvalSample, *, chat_model: str | None = None, label: str = "faith"
) -> float | None:
    """Fraction of answer claims supported by the retrieved context."""
    prompt = (
        f"Question: {sample.question}\n\n"
        f"Answer: {sample.answer}\n\n"
        f"Context:\n{_format_context_block(sample.contexts)}\n\n"
        "Break the answer into atomic factual statements. For each statement, "
        "decide whether it is fully supported by the context above. "
        "Return only the fraction of supported statements as a JSON score "
        "between 0 (none supported) and 1 (all supported)."
    )
    return _llm_json_score(client, prompt, chat_model=chat_model, label=label)


def answer_relevancy(
    client: LMStudioClient,
    embed: EmbedFn,
    sample: EvalSample,
    *,
    chat_model: str | None = None,
    label: str = "ans_rel",
) -> float | None:
    """Cosine similarity between the original question and a reverse-generated one."""
    try:
        reverse_q = client.chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Given an answer, write ONE concise question that the answer "
                        "would be a direct reply to. Reply with only the question."
                    ),
                },
                {"role": "user", "content": f"Answer: {sample.answer}\n\nQuestion:"},
            ],
            model=chat_model,
            temperature=0.0,
            max_tokens=80,
        ).lstrip("Q:").strip()
        embeds = embed([sample.question, reverse_q])
        return round(max(0.0, min(1.0, _cosine(embeds[0], embeds[1]))), 4)
    except Exception as exc:
        print(f"        [{label} error] {type(exc).__name__}: {exc}")
        return None


def context_precision(
    client: LMStudioClient,
    sample: EvalSample,
    *,
    chat_model: str | None = None,
    label: str = "ctx_prec",
) -> float | None:
    """Fraction of retrieved chunks an LLM judge considers useful."""
    prompt = (
        f"Question: {sample.question}\n\n"
        f"Reference answer: {sample.ground_truth}\n\n"
        f"Retrieved contexts:\n{_format_context_block(sample.contexts)}\n\n"
        "For each retrieved context, judge whether it contains information useful "
        "to answer the question correctly. Return only the fraction of useful "
        "contexts as a JSON score between 0 and 1."
    )
    return _llm_json_score(client, prompt, chat_model=chat_model, label=label)


def evaluate_samples(
    client: LMStudioClient,
    embed: EmbedFn,
    samples: Sequence[EvalSample],
    *,
    chat_model: str | None = None,
) -> list[MetricScores]:
    """Compute all three metrics for every sample."""
    out: list[MetricScores] = []
    for i, s in enumerate(samples, 1):
        faith   = faithfulness(client, s,   chat_model=chat_model, label=f"Q{i:02d} faith")
        ans_rel = answer_relevancy(client, embed, s, chat_model=chat_model, label=f"Q{i:02d} ans_rel")
        ctx_p   = context_precision(client, s, chat_model=chat_model, label=f"Q{i:02d} ctx_prec")
        scores  = MetricScores(faithfulness=faith, answer_relevancy=ans_rel, context_precision=ctx_p)
        out.append(scores)
        print(
            f"        scored Q{i:02d}:  faith={faith}  ans_rel={ans_rel}  ctx_prec={ctx_p}"
        )
    return out
