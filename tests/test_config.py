from __future__ import annotations

import pytest

from py_rag_engine.config import EvalConfig, LMStudioConfig, PostgresConfig


def test_lm_studio_defaults():
    cfg = LMStudioConfig()
    assert cfg.base_url == "http://localhost:1234"
    assert cfg.embed_model == "text-embedding-bge-m3"
    assert cfg.chat_model == ""
    assert cfg.embed_batch_size == 64
    assert cfg.retries == 5


def test_lm_studio_from_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LM_STUDIO_BASE_URL", "http://example:9999/")
    monkeypatch.setenv("LM_STUDIO_CHAT_MODEL", "qwen2.5-7b-instruct")
    monkeypatch.setenv("LM_STUDIO_EMBED_BATCH", "16")
    cfg = LMStudioConfig.from_env()
    assert cfg.base_url == "http://example:9999"           # trailing slash stripped
    assert cfg.chat_model == "qwen2.5-7b-instruct"
    assert cfg.embed_batch_size == 16


def test_postgres_from_full_url(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EVAL_POSTGRES_URL", "postgresql+psycopg://u:p@h:5/db")
    assert PostgresConfig.from_env().url == "postgresql+psycopg://u:p@h:5/db"


def test_postgres_assembles_from_parts(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("EVAL_POSTGRES_URL", raising=False)
    monkeypatch.setenv("POSTGRES_PASSWORD", "s3cret")
    monkeypatch.setenv("POSTGRES_USER", "rag")
    monkeypatch.setenv("POSTGRES_HOST", "db.local")
    monkeypatch.setenv("POSTGRES_PORT", "6543")
    monkeypatch.setenv("POSTGRES_DB", "ragdb")
    cfg = PostgresConfig.from_env()
    assert cfg.url == "postgresql+psycopg://rag:s3cret@db.local:6543/ragdb"


def test_postgres_uses_part_defaults(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("EVAL_POSTGRES_URL", raising=False)
    for var in ("POSTGRES_USER", "POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DB"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("POSTGRES_PASSWORD", "p")
    cfg = PostgresConfig.from_env()
    assert cfg.url == "postgresql+psycopg://postgres:p@localhost:5432/rag"


def test_postgres_requires_credentials(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("EVAL_POSTGRES_URL", raising=False)
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    with pytest.raises(RuntimeError, match="credentials missing"):
        PostgresConfig.from_env()


def test_postgres_named_var_overrides(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("EVAL_POSTGRES_URL", raising=False)
    monkeypatch.setenv("DEMO_POSTGRES_URL", "postgresql+psycopg://a:b@c:1/d")
    assert PostgresConfig.from_env(var="DEMO_POSTGRES_URL").url == "postgresql+psycopg://a:b@c:1/d"


def test_postgres_construct_with_explicit_url():
    """Direct construction with a URL still works (for tests / lib use)."""
    cfg = PostgresConfig(url="postgresql+psycopg://x:y@z:1/q")
    assert cfg.url == "postgresql+psycopg://x:y@z:1/q"


def test_eval_flags(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EVAL_SMOKE", "1")
    monkeypatch.setenv("EVAL_USE_OFFICIAL_RAGAS", "1")
    cfg = EvalConfig.from_env()
    assert cfg.smoke is True
    assert cfg.use_official_ragas is True
    assert cfg.preferred_chat_patterns[0] == "qwen2.5-7b"
