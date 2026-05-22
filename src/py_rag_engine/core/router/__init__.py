"""Query-ingestion router: classify intent, decompose, retrieve in parallel, fuse.

Public surface:
  * `QueryRouter`                — top-level orchestrator (async + sync).
  * `LLMIntentClassifier`        — LLM-backed classifier.
  * `heuristic_classify`         — pure-Python fallback classifier.
  * `LLMSubQueryGenerator`       — LLM-backed decomposer (max 3 sub-queries).
  * `heuristic_decompose`        — pure-Python fallback decomposer.
  * `multi_query_rrf`            — fuse N ranked lists with Reciprocal Rank Fusion.
  * `run_parallel_retrieval`     — embed-then-retrieve-then-fuse helper.
  * `call_structured` / `ChatFn` — generic structured-output LLM call.
  * Pydantic schemas: `RoutingDecision`, `SubQuery`, `SubQueryPlan`,
    `FusedRetrievalResult`, `RoutedQueryResult`, `QueryIntent`.
"""
from py_rag_engine.core.router.classifier import (
    LLMIntentClassifier,
    heuristic_classify,
)
from py_rag_engine.core.router.decomposer import (
    LLMSubQueryGenerator,
    heuristic_decompose,
)
from py_rag_engine.core.router.fusion import (
    DEFAULT_MULTI_QUERY_RRF_K,
    DEFAULT_MULTI_QUERY_TOP_K,
    multi_query_rrf,
)
from py_rag_engine.core.router.llm import (
    ChatFn,
    call_structured,
    chat_fn_from_anthropic,
    chat_fn_from_openai,
)
from py_rag_engine.core.router.orchestrator import (
    IntentClassifier,
    QueryRouter,
    SubQueryGenerator,
)
from py_rag_engine.core.router.parallel import (
    EmbedBatchFn,
    RetrieveFn,
    embed_sub_queries,
    retrieve_in_parallel,
    run_parallel_retrieval,
)
from py_rag_engine.core.router.schemas import (
    MAX_SUB_QUERIES,
    FusedRetrievalResult,
    QueryIntent,
    RoutedQueryResult,
    RoutingDecision,
    SubQuery,
    SubQueryPlan,
)

__all__ = [
    "DEFAULT_MULTI_QUERY_RRF_K",
    "DEFAULT_MULTI_QUERY_TOP_K",
    "MAX_SUB_QUERIES",
    "ChatFn",
    "EmbedBatchFn",
    "FusedRetrievalResult",
    "IntentClassifier",
    "LLMIntentClassifier",
    "LLMSubQueryGenerator",
    "QueryIntent",
    "QueryRouter",
    "RetrieveFn",
    "RoutedQueryResult",
    "RoutingDecision",
    "SubQuery",
    "SubQueryGenerator",
    "SubQueryPlan",
    "call_structured",
    "chat_fn_from_anthropic",
    "chat_fn_from_openai",
    "embed_sub_queries",
    "heuristic_classify",
    "heuristic_decompose",
    "multi_query_rrf",
    "retrieve_in_parallel",
    "run_parallel_retrieval",
]
