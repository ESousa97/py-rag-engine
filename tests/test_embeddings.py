from __future__ import annotations

from py_rag_engine.embeddings.hashing import content_sha256, normalize_for_hash


def test_content_hash_stable() -> None:
    a = content_sha256("  hello\n\nworld  ")
    b = content_sha256("hello world")
    assert a == b


def test_normalize_for_hash() -> None:
    assert normalize_for_hash("a\n\nb") == "a b"
