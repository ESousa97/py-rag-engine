"""Gold-standard question/answer dataset loader.

Reads `data/eval_questions.json` (schema_version 1):

    {
      "schema_version": 1,
      "questions": [
        {"id": "...", "question": "...", "ground_truth": "..."},
        ...
      ]
    }
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class GoldQuestion:
    id: str
    question: str
    ground_truth: str


SUPPORTED_SCHEMA_VERSIONS = (1,)


def load_gold_standard(path: str | Path) -> list[GoldQuestion]:
    """Load and validate the JSON gold-standard dataset."""
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(p)

    payload = json.loads(p.read_text(encoding="utf-8"))
    version = payload.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Unsupported schema_version {version!r} in {p}. "
            f"Supported: {SUPPORTED_SCHEMA_VERSIONS}"
        )

    raw = payload.get("questions", [])
    if not raw:
        raise ValueError(f"No questions found in {p}")

    out: list[GoldQuestion] = []
    seen_ids: set[str] = set()
    for i, item in enumerate(raw, 1):
        try:
            qid = item["id"]
            q   = item["question"]
            gt  = item["ground_truth"]
        except KeyError as exc:
            raise ValueError(f"Question #{i} missing required key {exc}") from exc
        if qid in seen_ids:
            raise ValueError(f"Duplicate question id {qid!r} at #{i}")
        seen_ids.add(qid)
        out.append(GoldQuestion(id=qid, question=q, ground_truth=gt))
    return out
