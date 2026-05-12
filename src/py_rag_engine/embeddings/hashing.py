from __future__ import annotations

import hashlib


def normalize_for_hash(text: str) -> str:
    return " ".join(text.split())


def content_sha256(text: str) -> str:
    normalized = normalize_for_hash(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
