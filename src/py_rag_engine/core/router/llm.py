"""Thin adapter for *structured* LLM calls.

Goal: let the router consume Pydantic-typed responses from any chat backend
(OpenAI, Anthropic, LM Studio) without hard-depending on a specific SDK.

Strategy:
  * Define a `ChatFn` protocol — `(messages, *, temperature, max_tokens) -> str`.
    `LMStudioClient.chat` already matches; OpenAI's `client.chat.completions.create`
    and Anthropic's `client.messages.create` can be adapted in two lines.
  * `call_structured` injects a JSON-Schema-flavoured user prompt, parses the
    response, and validates against the supplied Pydantic model.
  * On parse failure we retry with a *repair* prompt that quotes the offending
    output and the validation error — much cheaper than blowing up.
"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

ChatMessages = list[dict[str, str]]

T = TypeVar("T", bound=BaseModel)


class ChatFn(Protocol):
    """Subset of the OpenAI / LM Studio chat-completion signature."""

    def __call__(
        self,
        messages: ChatMessages,
        *,
        temperature: float = ...,
        max_tokens: int = ...,
    ) -> str: ...


_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)


def _extract_json_payload(raw: str) -> str:
    """Best-effort recovery of the JSON object/array embedded in `raw`.

    Models love to wrap their output in code fences or chatty preamble.
    We try, in order:
      1. fenced ```json ... ``` block,
      2. first balanced `{...}` or `[...]` we can find by bracket counting.
    """
    fenced = _FENCED_JSON_RE.search(raw)
    if fenced:
        return fenced.group(1).strip()

    for opener, closer in (("{", "}"), ("[", "]")):
        start = raw.find(opener)
        if start == -1:
            continue
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(raw)):
            ch = raw[i]
            if escape:
                escape = False
                continue
            if ch == "\\":
                escape = True
                continue
            if ch == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return raw[start : i + 1]
    return raw.strip()


def _format_schema_block(model_cls: type[BaseModel]) -> str:
    """Render the Pydantic model's JSON Schema as a prompt-friendly block."""
    schema = model_cls.model_json_schema()
    return json.dumps(schema, indent=2, sort_keys=True)


def _wrap_user_prompt(
    user_prompt: str,
    model_cls: type[BaseModel],
    *,
    extra_instructions: str | None = None,
) -> str:
    schema_block = _format_schema_block(model_cls)
    instructions = (
        "Reply with a SINGLE JSON object that conforms strictly to the schema "
        "below. Do NOT wrap it in markdown fences. Do NOT add commentary."
    )
    if extra_instructions:
        instructions = f"{instructions}\n{extra_instructions}"
    return (
        f"{user_prompt}\n\n"
        f"JSON Schema (output MUST validate against this):\n{schema_block}\n\n"
        f"{instructions}"
    )


def call_structured(
    chat: ChatFn,
    *,
    system: str,
    user: str,
    response_model: type[T],
    temperature: float = 0.0,
    max_tokens: int = 600,
    max_repair_attempts: int = 1,
) -> T:
    """Run a single chat call and decode the response into `response_model`.

    Args:
        chat: Anything matching `ChatFn` — typically `LMStudioClient.chat`.
        system: System message setting the role/persona.
        user: User instruction (the schema block is appended automatically).
        response_model: Pydantic class the response must validate against.
        max_repair_attempts: Extra retries with the parse error fed back in.

    Raises:
        ValueError: when the LLM cannot produce valid JSON within the budget.
    """
    messages: ChatMessages = [
        {"role": "system", "content": system},
        {"role": "user", "content": _wrap_user_prompt(user, response_model)},
    ]
    last_error: Exception | None = None
    last_raw: str = ""

    for attempt in range(max_repair_attempts + 1):
        raw = chat(messages, temperature=temperature, max_tokens=max_tokens)
        last_raw = raw
        payload = _extract_json_payload(raw)
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            last_error = exc
        else:
            try:
                return response_model.model_validate(data)
            except ValidationError as exc:
                last_error = exc

        if attempt >= max_repair_attempts:
            break

        # Inject a repair turn explaining what went wrong.
        messages.append({"role": "assistant", "content": raw})
        messages.append(
            {
                "role": "user",
                "content": (
                    "Your previous reply could not be parsed. "
                    f"Error: {last_error}\n"
                    "Reply again with ONLY a valid JSON object matching the schema."
                ),
            }
        )

    raise ValueError(
        f"LLM failed to produce valid {response_model.__name__} JSON after "
        f"{max_repair_attempts + 1} attempt(s). Last error: {last_error}. "
        f"Last raw output: {last_raw[:200]!r}"
    )


def chat_fn_from_openai(client: Any, *, model: str) -> ChatFn:
    """Adapt an `openai.OpenAI` client into a `ChatFn`.

    Kept here as a convenience so external callers don't repeat the boilerplate.
    """

    def _chat(
        messages: ChatMessages,
        *,
        temperature: float = 0.0,
        max_tokens: int = 600,
    ) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content or ""

    return _chat


def chat_fn_from_anthropic(client: Any, *, model: str) -> ChatFn:
    """Adapt an `anthropic.Anthropic` client into a `ChatFn`.

    Anthropic's API takes the system prompt out-of-band, so we split it off
    before forwarding.
    """

    def _chat(
        messages: ChatMessages,
        *,
        temperature: float = 0.0,
        max_tokens: int = 600,
    ) -> str:
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        non_system = [m for m in messages if m["role"] != "system"]
        resp = client.messages.create(
            model=model,
            system="\n".join(system_parts) if system_parts else "",
            messages=non_system,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # `resp.content` is a list of content blocks; take the first text block.
        for block in getattr(resp, "content", []):
            text = getattr(block, "text", None)
            if text:
                return text
        return ""

    return _chat


__all__ = [
    "ChatFn",
    "ChatMessages",
    "call_structured",
    "chat_fn_from_anthropic",
    "chat_fn_from_openai",
]
