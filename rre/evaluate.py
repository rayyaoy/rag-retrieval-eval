"""评测公共部分：读金标、逐题跑系统、计时、按题型出表、过期内容指标、运行环境指纹。"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Sequence

from .metrics import percentile, query_metrics, summarize

TYPES = ["crosslingual", "terminology", "number_source", "stale"]
TYPE_ZH = {"crosslingual": "跨语言", "terminology": "术语同义", "number_source": "数字出处",
           "stale": "过期内容", "ALL": "全部"}


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_gold(path: str | Path) -> list[dict]:
    items = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    for it in items:
        if not it.get("gold"):
            raise ValueError(f"{it.get('id')} 没有金标，先跑 goldset/build_gold.py")
    return items


def run_system(fn: Callable[[dict], list[str]], gold: Sequence[dict], warmup: int = 5):
    """先预热 warmup 题（不计时、不计分），再每题跑一次：计时那次的结果就是评分结果。"""
    for it in gold[:warmup]:
        fn(it)
    ranked, lat = {}, []
    for it in gold:
        t = time.perf_counter()
        r = fn(it)
        lat.append((time.perf_counter() - t) * 1000)
        ranked[it["id"]] = r
    return ranked, lat


def score(ranked: dict[str, list[str]], gold: Sequence[dict]):
    per_q, table = {}, defaultdict(list)
    for it in gold:
        m = query_metrics(ranked[it["id"]], it["gold"])
        per_q[it["id"]] = m
        table["ALL"].append(m)
        table[it["type"]].append(m)
    return per_q, {c: summarize(rows) for c, rows in table.items()}


def latency_summary(lat: Sequence[float]) -> dict:
    return {"p50": round(percentile(lat, .5), 1), "p95": round(percentile(lat, .95), 1), "n": len(lat)}


def stale_metrics(ranked: dict[str, list[str]], gold: Sequence[dict]) -> dict:
    """只看 stale 题。
    stale_top1        第 1 名就是旧年份段落的题目占比
    stale_in_top5     前 5 名里旧年份段落占的比例（按题平均）
    first_is_stale    在「新 ∪ 旧」相关段落里，排最前的是旧段落的占比（= 顺着结果读，先读到旧值的概率）
    current_found@5   前 5 名里有最新年份段落的题目占比
    """
    items = [it for it in gold if it["type"] == "stale"]
    if not items:
        return {}
    acc = defaultdict(float)
    for it in items:
        r, g, s = ranked[it["id"]], set(it["gold"]), set(it["stale"])
        top5 = r[:5]
        acc["stale_top1"] += bool(r) and r[0] in s
        acc["stale_in_top5"] += sum(p in s for p in top5) / 5
        first = next((p for p in r if p in g or p in s), None)
        acc["first_is_stale"] += first is not None and first in s
        acc["current_found@5"] += any(p in g for p in top5)
    n = len(items)
    return {"n": n, **{k: round(v / n, 3) for k, v in acc.items()}}


def env_fingerprint(cfg) -> dict:
    info = {"python": platform.python_version(), "machine": platform.machine(),
            "system": platform.system()}
    try:
        import torch
        info["torch"] = torch.__version__
    except Exception:  # noqa: BLE001
        pass
    for mod in ("sentence_transformers", "qdrant_client", "jieba"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            pass
    info["models"] = {"embedder": cfg["models"]["embedder"], "reranker": cfg["models"]["reranker"]}
    info["qdrant_location"] = cfg["qdrant"]["location"]
    return info


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
