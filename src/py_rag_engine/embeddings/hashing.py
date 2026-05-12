from __future__ import annotations

import hashlib


def normalize_for_hash(text: str) -> str:
    """Collapse whitespace so equivalent text hashes consistently."""
    return " ".join(text.split())


def content_sha256(text: str) -> str:
    """Return a SHA-256 hash for normalized chunk content."""
    normalized = normalize_for_hash(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
