"""Tiny Ollama client (stdlib only) so everything runs against local open-weight models."""

from __future__ import annotations

import json
import os
import urllib.request


def _base_url() -> str:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    return host if host.startswith("http") else f"http://{host}"


def is_available(timeout: float = 0.5) -> bool:
    try:
        with urllib.request.urlopen(f"{_base_url()}/api/tags", timeout=timeout):
            return True
    except Exception:
        return False


def chat(model: str, messages: list[dict], tools: list[dict] | None = None,
         fmt: dict | str | None = None, timeout: float = 180) -> dict:
    payload: dict = {"model": model, "messages": messages, "stream": False,
                     "options": {"temperature": 0}}
    if tools:
        payload["tools"] = tools
    if fmt:
        payload["format"] = fmt
    req = urllib.request.Request(
        f"{_base_url()}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def text_model() -> str:
    return os.environ.get("PAYDAY_TEXT_MODEL", "qwen2.5:7b")


def vision_model() -> str:
    return os.environ.get("PAYDAY_VISION_MODEL", "qwen2.5vl:7b")
