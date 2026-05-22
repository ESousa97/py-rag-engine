"""Intent classifier for incoming user questions.

Two implementations live here:

* `LLMIntentClassifier` — calls a structured LLM via `core.router.llm` and
  returns a validated `RoutingDecision`. This is the default in production.
* `heuristic_classify` — pure-Python fallback used when no LLM is configured
  or when the LLM call fails. Cheap, deterministic, good enough for tests.

The classifier is intentionally a single-shot, low-temperature call: routing
mistakes are usually recoverable downstream (worst case: we run a needless
sub-query decomposition), so we trade marginal accuracy for latency.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from py_rag_engine.core.router.llm import ChatFn, call_structured
from py_rag_engine.core.router.schemas import QueryIntent, RoutingDecision

_CLASSIFIER_SYSTEM_PROMPT = (
    "You classify user questions for a Retrieval-Augmented Generation engine. "
    "You are precise, terse, and always reply with a JSON object that matches "
    "the supplied schema."
)

_CLASSIFIER_USER_TEMPLATE = (
    "Classify the following user question.\n\n"
    "Definitions:\n"
    "- simple_factual: one fact, no chaining (e.g. 'What is the capital of France?').\n"
    "- complex_multi_hop: needs several facts combined to answer (e.g. "
    "'How does X affect Y when Z changes?').\n"
    "- comparative: explicitly compares two or more entities/options.\n"
    "- conversational: chit-chat, greetings, clarifications.\n"
    "- out_of_scope: not answerable from a knowledge base of documents.\n\n"
    "Pick the BEST intent. Set complexity_score between 0 and 1. "
    "Set needs_decomposition=true when the question genuinely benefits from "
    "being split into <=3 independent sub-questions.\n\n"
    "Question: {question}"
)

# Heuristic markers — kept small on purpose. Order matters: comparative is
# checked before multi-hop because comparative questions often contain the
# multi-hop conjunctions too.
_COMPARATIVE_PATTERNS = (
    r"\bvs\.?\b",
    r"\bversus\b",
    r"\bcompared?\b",
    r"\bcomparison\b",
    r"\bdifference(?:s)? between\b",
    r"\bbetter than\b",
)

_MULTI_HOP_MARKERS = (
    r"\band\b.{0,120}\band\b",  # two `and`s in same question → likely multi-part
    r"\balso\b",
    r"\bin addition\b",
    r"\bas well as\b",
    r"\bafter\b",
    r"\bbefore\b",
    r"\bbecause\b",
    r"\bwhich\b.+\bthat\b",
    r"\bhow does\b.+\bdiffer\b",
    r"\bdiffer(?:ence|s)?\b.+\bfrom\b",
    r"\bwhen should\b",
)

_CONVERSATIONAL_PATTERNS = (
    r"^\s*(hi|hello|hey|thanks|thank you|good (morning|afternoon|evening))\b",
    r"^\s*(how are you|what can you do)\b",
)


def _matches_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(p, text, flags=re.IGNORECASE) for p in patterns)


def heuristic_classify(question: str) -> RoutingDecision:
    """LLM-free classifier used as a fallback or in tests.

    Decision tree:
      1. Conversational greeting → conversational, score 0.0.
      2. Comparative markers → comparative, score 0.7.
      3. Multi-hop markers or multiple `?` / many words → complex_multi_hop.
      4. Default → simple_factual.
    """
    stripped = question.strip()
    if not stripped:
        # Empty input is conversational by convention — caller decides what to do.
        return RoutingDecision(
            intent=QueryIntent.CONVERSATIONAL,
            complexity_score=0.0,
            reasoning="empty input",
        )

    if _matches_any(stripped, _CONVERSATIONAL_PATTERNS):
        return RoutingDecision(
            intent=QueryIntent.CONVERSATIONAL,
            complexity_score=0.0,
            reasoning="matched conversational greeting pattern",
        )

    if _matches_any(stripped, _COMPARATIVE_PATTERNS):
        return RoutingDecision(
            intent=QueryIntent.COMPARATIVE,
            complexity_score=0.7,
            reasoning="contains comparative markers (vs / compare / difference)",
            needs_decomposition=True,
        )

    word_count = len(re.findall(r"\w+", stripped))
    question_marks = stripped.count("?")
    multi_hop_match = _matches_any(stripped, _MULTI_HOP_MARKERS)

    if multi_hop_match or question_marks >= 2 or word_count >= 25:
        score = min(1.0, 0.5 + 0.02 * max(0, word_count - 15) + 0.1 * question_marks)
        return RoutingDecision(
            intent=QueryIntent.COMPLEX_MULTI_HOP,
            complexity_score=round(score, 3),
            reasoning=(
                f"heuristic flagged complexity: words={word_count}, "
                f"question_marks={question_marks}, multi_hop_match={multi_hop_match}"
            ),
            needs_decomposition=True,
        )

    return RoutingDecision(
        intent=QueryIntent.SIMPLE_FACTUAL,
        complexity_score=0.1,
        reasoning="single-fact pattern, no decomposition markers",
    )


@dataclass(slots=True)
class LLMIntentClassifier:
    """LLM-backed classifier with deterministic heuristic fallback.

    Pass any `ChatFn` (LM Studio, OpenAI, Anthropic — see `core.router.llm`).
    """

    chat: ChatFn
    temperature: float = 0.0
    max_tokens: int = 300
    system_prompt: str = _CLASSIFIER_SYSTEM_PROMPT
    fallback_on_error: bool = True

    def classify(self, question: str) -> RoutingDecision:
        try:
            return call_structured(
                self.chat,
                system=self.system_prompt,
                user=_CLASSIFIER_USER_TEMPLATE.format(question=question),
                response_model=RoutingDecision,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception:
            if not self.fallback_on_error:
                raise
            return heuristic_classify(question)


__all__ = ["LLMIntentClassifier", "heuristic_classify"]
