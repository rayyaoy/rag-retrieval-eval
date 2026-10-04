"""读 passages.jsonl。"""
from __future__ import annotations

import json
from pathlib import Path

from .passages import Passage


def load_passages(path: str | Path) -> list[Passage]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(Passage(**json.loads(line)))
    ids = [p.pid for p in out]
    if len(set(ids)) != len(ids):
        raise ValueError("段落 ID 重复")
    return out
