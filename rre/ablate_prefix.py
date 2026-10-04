#!/usr/bin/env python3
"""消融：段落前缀（「文档名 > 小节路径」）对向量检索的贡献。

同一批段落、同一个模型，只差送进模型的文本带不带前缀：
  with_prefix  编码 前缀 + 换行 + 正文（主评测用的就是这个）
  no_prefix    只编码正文，写进另一个 collection（{collection}_noprefix）
逐题配对：ΔRecall@5 = 有前缀 − 无前缀，配对 bootstrap 95% 区间（10,000 次，种子 0）。

    python -m rre.ablate_prefix
"""
from __future__ import annotations

import time

from .config import ROOT, load_config


def _rel(p):
    try:
        return p.relative_to(ROOT)
    except ValueError:
        return p
from .evaluate import TYPES, load_gold, score, stale_metrics, write_json
from .metrics import paired_bootstrap_ci
from .models import STEmbedder
from .search import build_retriever


def main() -> int:
    cfg = load_config()
    gold = load_gold(cfg.path("goldset"))
    m = cfg["models"]
    emb = STEmbedder(m["embedder"], m["max_seq_length"], m["device"], m["batch_size"])
    out, per, stale = {}, {}, {}
    for name, wp in (("with_prefix", True), ("no_prefix", False)):
        t = time.time()
        r = build_retriever(cfg, need_vector=True, with_prefix=wp, embedder=emb)
        ranked = {it["id"]: r.vector(it["query"]) for it in gold}
        per[name], out[name] = score(ranked, gold)
        stale[name] = stale_metrics(ranked, gold)
        r.client.close()
        print(f"  {name:<12} R@5 {out[name]['ALL']['recall@5']:.3f}  ({time.time() - t:.0f}s, {r.info})")
    ids = [it["id"] for it in gold]
    diffs = [per["with_prefix"][i]["recall"] - per["no_prefix"][i]["recall"] for i in ids]
    lo, hi = paired_bootstrap_ci(diffs)
    res = {"run_at": time.strftime("%Y-%m-%d %H:%M:%S"), "metrics": out,
           "delta_recall5": round(sum(diffs) / len(diffs), 3), "boot95": [round(lo, 3), round(hi, 3)],
           "by_type_delta": {t: round(out["with_prefix"][t]["recall@5"] - out["no_prefix"][t]["recall@5"], 3)
                             for t in TYPES if t in out["with_prefix"]},
           "stale": stale}
    path = cfg.path("results_dir") / "ablate_prefix.json"
    write_json(path, res)
    print(f"ΔRecall@5（有前缀 − 无前缀）= {res['delta_recall5']:+.3f}，95% [{lo:+.3f}, {hi:+.3f}]")
    print(f"写入 {_rel(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
