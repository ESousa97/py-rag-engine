from __future__ import annotations

from typing import Any

from py_rag_engine.clients import LMStudioClient
from py_rag_engine.config import LMStudioConfig
from py_rag_engine.generation import build_generation_prompt, generate_answer


def test_build_generation_prompt_enumerates_contexts():
    msgs = build_generation_prompt("What is X?", ["chunk one", "chunk two"])
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    content = msgs[0]["content"]
    assert "[1] chunk one" in content
    assert "[2] chunk two" in content
    assert "Question: What is X?" in content


def test_generate_answer_uses_chat_client():
    captured: dict[str, Any] = {}

    def fake_http(url: str, payload: Any = None, *, timeout: int | None = None) -> Any:
        captured["url"] = url
        captured["payload"] = payload
        return {"choices": [{"message": {"content": "X is the answer."}}]}

    client = LMStudioClient(LMStudioConfig(chat_model="dummy"), http_json=fake_http)
    answer = generate_answer(client, "What is X?", ["X is well-known."])
    assert answer == "X is the answer."
    assert captured["payload"]["temperature"] == 0.05
    assert captured["payload"]["max_tokens"] == 400
    assert "X is well-known." in captured["payload"]["messages"][0]["content"]
