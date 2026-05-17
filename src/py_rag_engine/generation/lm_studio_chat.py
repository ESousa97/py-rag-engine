"""LLM-based answer generation grounded in retrieved context.

The standard RAG prompt asks the model to answer using ONLY the supplied
context. This module is intentionally tiny — it just composes the prompt and
delegates the HTTP call to `LMStudioClient`.
"""
from __future__ import annotations

from collections.abc import Sequence

from py_rag_engine.clients import LMStudioClient

DEFAULT_GENERATION_PROMPT = (
    "You are a helpful assistant. Answer the question using ONLY the information "
    "provided in the context below. Be concise and precise. Do not add any "
    "information that is not explicitly stated in the context."
)


def build_generation_prompt(
    question: str,
    contexts: Sequence[str],
    *,
    system: str = DEFAULT_GENERATION_PROMPT,
) -> list[dict[str, str]]:
    """Assemble the chat messages for a grounded answer."""
    ctx_block = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    user = (
        f"{system}\n\n"
        f"Context:\n{ctx_block}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    return [{"role": "user", "content": user}]


def generate_answer(
    client: LMStudioClient,
    question: str,
    contexts: Sequence[str],
    *,
    chat_model: str | None = None,
    temperature: float = 0.05,
    max_tokens: int = 400,
) -> str:
    """Generate a grounded answer for `question` given retrieved `contexts`."""
    messages = build_generation_prompt(question, contexts)
    return client.chat(
        messages,
        model=chat_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
