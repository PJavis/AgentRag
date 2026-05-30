"""make_async_openai — gating behaviour around LANGFUSE_ENABLED."""
from __future__ import annotations

from openai import AsyncOpenAI

from src.agentrag.common import langfuse_client
from src.agentrag.common.langfuse_client import init_langfuse, make_async_openai
from src.agentrag.config import settings


def test_disabled_returns_plain_async_openai(monkeypatch):
    monkeypatch.setattr(settings, "LANGFUSE_ENABLED", False)
    client = make_async_openai(api_key="sk-test", base_url=None)
    assert type(client) is AsyncOpenAI


def test_enabled_but_langfuse_missing_falls_back_to_plain(monkeypatch):
    # Simulate langfuse not installed: import inside make_async_openai raises.
    monkeypatch.setattr(settings, "LANGFUSE_ENABLED", True)
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "langfuse.openai" or name.startswith("langfuse"):
            raise ImportError("no langfuse")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    client = make_async_openai(api_key="sk-test", base_url=None)
    assert type(client) is AsyncOpenAI


def test_init_langfuse_exports_env(monkeypatch):
    monkeypatch.setattr(settings, "LANGFUSE_ENABLED", True)
    monkeypatch.setattr(settings, "LANGFUSE_PUBLIC_KEY", "pk-123")
    monkeypatch.setattr(settings, "LANGFUSE_SECRET_KEY", "sk-456")
    monkeypatch.setattr(settings, "LANGFUSE_HOST", "http://lf.local:3000")
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    init_langfuse()
    import os

    assert os.environ["LANGFUSE_PUBLIC_KEY"] == "pk-123"
    assert os.environ["LANGFUSE_SECRET_KEY"] == "sk-456"
    assert os.environ["LANGFUSE_HOST"] == "http://lf.local:3000"


def test_disabled_init_is_noop(monkeypatch):
    monkeypatch.setattr(settings, "LANGFUSE_ENABLED", False)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    init_langfuse()
    import os

    assert "LANGFUSE_PUBLIC_KEY" not in os.environ
