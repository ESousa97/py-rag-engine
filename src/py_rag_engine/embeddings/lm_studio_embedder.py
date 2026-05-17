"""Embedding adapter backed by an LM Studio HTTP endpoint."""
from __future__ import annotations

from collections.abc import Callable

from py_rag_engine.clients import LMStudioClient

EmbedFn = Callable[[list[str]], list[list[float]]]


def make_lm_studio_embed(client: LMStudioClient, *, model: str | None = None) -> EmbedFn:
    """Return an `embed(texts) -> vectors` callable bound to an LM Studio client."""
    def _embed(texts: list[str]) -> list[list[float]]:
        return client.embed(texts, model=model)
    return _embed
