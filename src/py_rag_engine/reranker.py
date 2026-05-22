from __future__ import annotations

import importlib
from pathlib import Path
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from py_rag_engine.search import HybridSearchDocument

DEFAULT_FLASHRANK_MODEL = "ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANK_TOP_K = 5


class FlashrankRankerProtocol(Protocol):
    def rerank(self, request: Any) -> list[dict[str, Any]]:
        ...


@dataclass(frozen=True, slots=True)
class RerankedDocument:
    """A hybrid-search document with an updated FlashRank relevance score."""

    id: int
    text: str
    metadata: dict[str, Any]
    content_hash: str | None
    embedding_model: str
    confidence_score: float
    rerank_score: float
    rrf_score: float | None = None
    cosine_similarity: float | None = None
    fts_score: float | None = None


def _get_value(document: Any, name: str, default: Any = None) -> Any:
    if isinstance(document, Mapping):
        return document.get(name, default)
    return getattr(document, name, default)


class FlashrankReranker:
    """Lazy FlashRank wrapper for local re-ranking of RRF candidates."""

    def __init__(
        self,
        model_name: str = DEFAULT_FLASHRANK_MODEL,
        *,
        cache_dir: str | None = None,
        max_length: int = 512,
        model_file_name: str | None = None,
        ranker: FlashrankRankerProtocol | None = None,
    ) -> None:
        if max_length < 1:
            raise ValueError("max_length must be greater than zero")
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.model_file_name = model_file_name
        self._ranker = ranker
        self._request_cls: type | None = None

    def rerank(
        self,
        query: str,
        documents: Sequence[Any],
        *,
        top_k: int = DEFAULT_RERANK_TOP_K,
    ) -> list[RerankedDocument]:
        """Re-rank RRF candidates and return the top documents with new scores."""
        if top_k < 1:
            raise ValueError("top_k must be greater than zero")
        if not documents:
            return []

        ranker, request_cls = self._resolve_flashrank()
        passages = [
            {
                "id": _get_value(document, "id"),
                "text": str(_get_value(document, "text", "")),
                "meta": {"candidate_index": index},
            }
            for index, document in enumerate(documents)
        ]
        request = request_cls(query=query, passages=passages)
        ranked_passages = ranker.rerank(request)

        output: list[RerankedDocument] = []
        for passage in ranked_passages[:top_k]:
            source = self._source_document(passage, documents)
            score = float(passage.get("score", 0.0))
            output.append(
                RerankedDocument(
                    id=int(_get_value(source, "id")),
                    text=str(_get_value(source, "text", "")),
                    metadata=dict(_get_value(source, "metadata", {}) or {}),
                    content_hash=_get_value(source, "content_hash"),
                    embedding_model=str(_get_value(source, "embedding_model", "")),
                    confidence_score=score,
                    rerank_score=score,
                    rrf_score=_optional_float(_get_value(source, "rrf_score")),
                    cosine_similarity=_optional_float(_get_value(source, "cosine_similarity")),
                    fts_score=_optional_float(_get_value(source, "fts_score")),
                )
            )
        return output

    def _resolve_flashrank(self) -> tuple[FlashrankRankerProtocol, type]:
        if self._ranker is not None:
            if self._request_cls is None:
                self._request_cls = _SimpleRerankRequest
            return self._ranker, self._request_cls

        try:
            from flashrank import Ranker, RerankRequest
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "flashrank is required for FlashrankReranker; install with: "
                "pip install 'py-rag-engine[rerank]'"
            ) from exc

        self._register_local_model_if_needed()
        kwargs: dict[str, Any] = {
            "model_name": self.model_name,
            "max_length": self.max_length,
        }
        if self.cache_dir is not None:
            kwargs["cache_dir"] = self.cache_dir

        try:
            self._ranker = Ranker(**kwargs)
        except KeyError as exc:  # pragma: no cover
            raise ValueError(
                f"flashrank does not know model '{self.model_name}'. "
                "Use a FlashRank-supported model or a local cache compatible with that model name."
            ) from exc
        self._request_cls = RerankRequest
        return self._ranker, self._request_cls

    def _register_local_model_if_needed(self) -> None:
        try:
            config = importlib.import_module("flashrank.Config")
            ranker_module = importlib.import_module("flashrank.Ranker")
        except ImportError:
            return

        if self.model_name in config.model_file_map:
            return

        model_file = self.model_file_name
        if model_file is None:
            cache_dir = Path(self.cache_dir or config.default_cache_dir)
            model_file = _find_local_onnx_file(cache_dir / self.model_name)
        if model_file is None:
            return

        normalized_model_file = str(Path(model_file)).replace("\\", "/")
        config.model_file_map[self.model_name] = normalized_model_file
        ranker_module.model_file_map[self.model_name] = normalized_model_file

    @staticmethod
    def _source_document(passage: Mapping[str, Any], documents: Sequence[Any]) -> Any:
        meta = passage.get("meta")
        if isinstance(meta, Mapping):
            index = meta.get("candidate_index")
            if isinstance(index, int) and 0 <= index < len(documents):
                return documents[index]

        passage_id = passage.get("id")
        for document in documents:
            if _get_value(document, "id") == passage_id:
                return document
        raise ValueError("flashrank returned a passage that was not in the candidate list")


class _SimpleRerankRequest:
    """Small request shim used by tests that inject a fake ranker."""

    def __init__(
        self,
        *,
        query: str | None = None,
        passages: list[dict[str, Any]] | None = None,
    ) -> None:
        self.query = query
        self.passages = passages or []


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _find_local_onnx_file(model_dir: Path) -> str | None:
    if not model_dir.exists():
        return None

    preferred = (
        "flashrank-MiniLM-L-6-v2.onnx",
        "model_quantized.onnx",
        "model.onnx",
        "model_fp16.onnx",
        "onnx/model_quantized.onnx",
        "onnx/model.onnx",
        "onnx/model_fp16.onnx",
    )
    for relative_name in preferred:
        if (model_dir / relative_name).exists():
            return relative_name

    candidates = sorted(model_dir.rglob("*.onnx"))
    if not candidates:
        return None
    return candidates[0].relative_to(model_dir).as_posix()


def rerank_documents(
    query: str,
    documents: Sequence[HybridSearchDocument | Mapping[str, Any]],
    *,
    top_k: int = DEFAULT_RERANK_TOP_K,
    reranker: FlashrankReranker | None = None,
) -> list[RerankedDocument]:
    """Re-rank the 20 RRF candidates and keep the 5 most relevant by default."""
    active_reranker = reranker or FlashrankReranker()
    return active_reranker.rerank(query, documents, top_k=top_k)
