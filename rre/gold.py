"""金标：题目（questions.jsonl，人写）→ 金标段落 ID（goldset.jsonl，机器解析）。

题目字段：
  id        唯一编号
  type      crosslingual | terminology | number_source | stale
  query     自然问句（向量 / BM25 / 融合 / 精排的输入）
  keywords  用户在关键词检索框里会敲的词（基线的输入）
  evidence  [{"doc": 文档 ID 前缀, "text": 证据串}]：证据串必须在段落正文里逐字出现
  stale_evidence（仅 stale 题）：同一公司上一年年报里对应的旧说法

解析规则（resolve）：金标 = 所有 doc_id 以 doc 开头、且正文（空白压缩后）包含证据串的段落。
  - 非 stale 题的 doc 一般写到公司一级（如 edgar-nvda），两年年报里一字不差的同一句话都算相关；
  - stale 题的 doc 写到具体年份，旧年份的对应段落进 stale 列表，不算金标。
证据串一个都对不上、或 stale 与金标重叠，check 直接报错，不静默修。
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Sequence

from .passages import Passage

TYPES = ("crosslingual", "terminology", "number_source", "stale")
REQUIRED = ("id", "type", "query", "keywords", "evidence")
WS = re.compile(r"\s+")


def norm(s: str) -> str:
    return WS.sub(" ", s).strip()


def resolve(evidence: Iterable[dict], passages: Sequence[Passage]) -> tuple[list[str], list[str]]:
    """返回 (命中的段落 ID 列表（语料顺序、去重）, 没有命中的证据串列表)。"""
    hits: list[str] = []
    missing: list[str] = []
    for ev in evidence:
        t = norm(ev["text"])
        found = [p.pid for p in passages if p.doc_id.startswith(ev["doc"]) and t in norm(p.body)]
        if not found:
            missing.append(f"{ev['doc']} :: {t[:60]}")
        hits.extend(found)
    seen, out = set(), []
    order = {p.pid: i for i, p in enumerate(passages)}
    for pid in sorted(set(hits), key=lambda x: order[x]):
        if pid not in seen:
            seen.add(pid)
            out.append(pid)
    return out, missing


def build_item(q: dict, passages: Sequence[Passage]) -> tuple[dict, list[str]]:
    errs = [f"缺字段 {k}" for k in REQUIRED if not q.get(k)]
    if errs:
        return q, errs
    if q["type"] not in TYPES:
        errs.append(f"未知题型 {q['type']}")
    gold, miss = resolve(q["evidence"], passages)
    errs += [f"证据串未命中 {m}" for m in miss]
    item = {k: q[k] for k in ("id", "type", "query", "keywords", "evidence")}
    item["gold"] = gold
    if q["type"] == "stale":
        if not q.get("stale_evidence"):
            errs.append("stale 题缺 stale_evidence")
        else:
            stale, miss2 = resolve(q["stale_evidence"], passages)
            errs += [f"旧证据串未命中 {m}" for m in miss2]
            if set(stale) & set(gold):
                errs.append("stale 与金标重叠")
            item["stale_evidence"] = q["stale_evidence"]
            item["stale"] = stale
    if not gold:
        errs.append("金标为空")
    return item, errs


def build_goldset(questions: Sequence[dict], passages: Sequence[Passage]):
    items, problems, seen = [], {}, set()
    for q in questions:
        item, errs = build_item(q, passages)
        if q.get("id") in seen:
            errs.append("id 重复")
        seen.add(q.get("id"))
        if errs:
            problems[q.get("id", "?")] = errs
        items.append(item)
    return items, problems


def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def dump_jsonl(items: Iterable[dict], path: str | Path) -> None:
    Path(path).write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in items),
                          encoding="utf-8")
