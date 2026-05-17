from __future__ import annotations

from typing import Any

import pytest

from py_rag_engine.clients import LMStudioClient, detect_chat_model
from py_rag_engine.config import LMStudioConfig


def _fake_http_factory(responses: list[Any]) -> Any:
    calls: list[tuple[str, Any, int | None]] = []

    def _fake(url: str, payload: Any = None, *, timeout: int | None = None) -> Any:
        calls.append((url, payload, timeout))
        if not responses:
            raise AssertionError(f"unexpected call to {url} with {payload}")
        return responses.pop(0)

    _fake.calls = calls
    return _fake


def test_models_returns_list_of_dicts():
    fake = _fake_http_factory([{"data": [{"id": "qwen2.5-7b"}, {"id": "bge"}]}])
    client = LMStudioClient(LMStudioConfig(), http_json=fake)
    assert [m["id"] for m in client.models()] == ["qwen2.5-7b", "bge"]


def test_embed_batches_texts():
    cfg  = LMStudioConfig(embed_batch_size=2)
    fake = _fake_http_factory([
        {"data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}]},
        {"data": [{"embedding": [0.5, 0.6]}]},
    ])
    client = LMStudioClient(cfg, http_json=fake)
    out    = client.embed(["a", "b", "c"], model="bge-m3")
    assert out == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]
    # First batch had 2 texts, second batch had 1
    assert fake.calls[0][1]["input"] == ["a", "b"]
    assert fake.calls[1][1]["input"] == ["c"]


def test_chat_requires_model_config():
    client = LMStudioClient(LMStudioConfig(chat_model=""), http_json=lambda *a, **kw: {})
    with pytest.raises(ValueError, match="chat_model not configured"):
        client.chat([{"role": "user", "content": "hi"}])


def test_chat_returns_message_content():
    fake = _fake_http_factory([{"choices": [{"message": {"content": "  hello  "}}]}])
    client = LMStudioClient(LMStudioConfig(chat_model="m"), http_json=fake)
    assert client.chat([{"role": "user", "content": "hi"}]) == "hello"


def test_detect_picks_preferred_pattern_first():
    fake = _fake_http_factory([{"data": [
        {"id": "qwen2.5-14b-instruct"},
        {"id": "qwen2.5-7b-instruct"},
        {"id": "text-embedding-bge-m3"},
    ]}])
    client = LMStudioClient(LMStudioConfig(), http_json=fake)
    assert detect_chat_model(client) == "qwen2.5-7b-instruct"


def test_detect_falls_back_to_size_band():
    # No preferred pattern matches; should pick the 8B over 14B and over 3B.
    fake = _fake_http_factory([{"data": [
        {"id": "some-model-14b"},
        {"id": "some-model-8b"},
        {"id": "some-model-3b"},
    ]}])
    client = LMStudioClient(LMStudioConfig(), http_json=fake)
    assert detect_chat_model(client) == "some-model-8b"


def test_detect_filters_embedding_hints():
    fake = _fake_http_factory([{"data": [
        {"id": "text-embedding-bge-m3"},
        {"id": "text-embedding-nomic-embed"},
        {"id": "qwen2.5-7b-instruct"},
    ]}])
    client = LMStudioClient(LMStudioConfig(), http_json=fake)
    assert detect_chat_model(client) == "qwen2.5-7b-instruct"


def test_warm_up_returns_false_on_failure():
    def fail(*a, **kw): raise RuntimeError("nope")
    client = LMStudioClient(LMStudioConfig(chat_model="m"), http_json=fail)
    assert client.warm_up_chat() is False
