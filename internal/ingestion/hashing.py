from __future__ import annotations

import hashlib


def normalize_for_hash(text: str) -> str:
    """Collapse whitespace so equivalent content maps to the same hash."""
    return " ".join(text.split())


def content_sha256(text: str) -> str:
    """Deterministic SHA-256 hex digest of UTF-8 normalized text."""
    normalized = normalize_for_hash(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
