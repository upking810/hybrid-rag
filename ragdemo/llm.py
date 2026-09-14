"""Thin wrapper around the chat API, with a JSON-mode helper."""
from __future__ import annotations

import json

from . import config
from .embeddings import client


def chat(system: str, user: str, temperature: float = 0.0) -> str:
    resp = client().chat.completions.create(
        model=config.CHAT_MODEL,
        temperature=temperature,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content or ""


def chat_json(system: str, user: str) -> dict:
    resp = client().chat.completions.create(
        model=config.CHAT_MODEL,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    return json.loads(resp.choices[0].message.content or "{}")
