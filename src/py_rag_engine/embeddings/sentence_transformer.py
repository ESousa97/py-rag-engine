"""Embedding adapter backed by a local `sentence-transformers` model.

Lazy-imports torch / sentence_transformers so callers that never invoke
`make_sentence_transformer_embed` don't pay the ~3 s import cost.
"""
from __future__ import annotations

from collections.abc import Callable

EmbedFn = Callable[[list[str]], list[list[float]]]


def make_sentence_transformer_embed(
    model_name: str,
    *,
    batch_size: int = 32,
    device: str | None = None,
) -> EmbedFn:
    """Return an `embed(texts) -> vectors` callable using a local ST model.

    `device` is auto-detected when omitted (`cuda` if available, else `cpu`).
    """
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "sentence-transformers is required for make_sentence_transformer_embed; "
            "install with: pip install 'py-rag-engine[embeddings]'"
        ) from exc

    resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer(model_name, device=resolved_device)

    def _embed(texts: list[str]) -> list[list[float]]:
        vecs = model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            device=resolved_device,
            batch_size=batch_size,
        )
        return [v.tolist() for v in vecs]

    return _embed
