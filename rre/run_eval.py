#!/usr/bin/env python3
"""主评测：五路对照 + 三个预注册变体，同一进程、同一份索引、同一套金标题。

    python -m rre.run_eval                 # 全部系统
    python -m rre.run_eval --no-rerank     # 跳过需要精排模型的系统（hybrid / V1 / V3）

输出（results/public/）：
  eval.json            各系统 × 题型的 Recall@5 / Hit@5 / MRR@10、延迟 P50/P95、过期内容指标、
                       相对纯向量的配对 bootstrap 区间与预注册换线条件逐条判定、运行环境与输入文件指纹
  per_query.jsonl      逐题：金标、每个系统的第一个金标名次与前 5 名（公开语料，可复现）
"""
from __future__ import annotations

import argparse
import json
import time

from .config import ROOT, load_config


def _rel(p):
    try:
        return p.relative_to(ROOT)
    except ValueError:
        return p
from .evaluate import (TYPES, env_fingerprint, latency_summary, load_gold, run_system, score,
                       sha256_file, stale_metrics, write_json)
from .metrics import paired_bootstrap_ci, switch_decision
from .search import build_retriever

PREREG = ROOT / "prereg" / "2026-10-04-public.md"
BASELINE = "vector"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()
    cfg = load_config()
    gold = load_gold(cfg.path("goldset"))
    print(f"金标题 {len(gold)} 道：" + "，".join(f"{t} {sum(g['type'] == t for g in gold)}" for t in TYPES))

    t0 = time.time()
    r = build_retriever(cfg, need_vector=True, need_rerank=not args.no_rerank)
    load_s = time.time() - t0
    print(f"检索器就绪 {load_s:.1f}s  {r.info}")

    systems = {
        "keyword": lambda it: r.keyword(it["keywords"]),
        "bm25": lambda it: r.bm25_search(it["query"]),
        "vector": lambda it: r.vector(it["query"]),
        "rrf": lambda it: r.rrf(it["query"]),
    }
    if not args.no_rerank:
        systems["hybrid"] = lambda it: r.hybrid(it["query"])
        systems["V1_vec_rerank"] = lambda it: r.vec_rerank(it["query"])
    systems["V2_wrrf"] = lambda it: r.wrrf(it["query"])
    if not args.no_rerank:
        systems["V3_route"] = lambda it: r.route(it["query"])

    ranked, out_sys, per_q = {}, {}, {}
    for name, fn in systems.items():
        t = time.time()
        ranked[name], lat = run_system(fn, gold, args.warmup)
        per_q[name], tables = score(ranked[name], gold)
        out_sys[name] = {"metrics": tables, "latency_ms": latency_summary(lat),
                         "stale": stale_metrics(ranked[name], gold)}
        m = tables["ALL"]
        print(f"  {name:<14} R@5 {m['recall@5']:.3f}  Hit@5 {m['hit@5']:.3f}  MRR {m['mrr@10']:.3f}  "
              f"P50 {out_sys[name]['latency_ms']['p50']:.0f}ms  ({time.time() - t:.0f}s)")

    ids = [it["id"] for it in gold]
    base = out_sys[BASELINE]["metrics"]
    comps = {}
    for name in systems:
        if name == BASELINE:
            continue
        diffs = [per_q[name][i]["recall"] - per_q[BASELINE][i]["recall"] for i in ids]
        lo, hi = paired_bootstrap_ci(diffs)
        m = out_sys[name]["metrics"]
        delta = sum(diffs) / len(diffs)
        worst = min(m[t]["recall@5"] - base[t]["recall@5"] for t in TYPES if t in m)
        dec = switch_decision(delta, lo, worst, out_sys[name]["latency_ms"]["p95"])
        comps[name] = {"delta_recall5": round(delta, 3), "boot95": [round(lo, 3), round(hi, 3)],
                       "worst_type_delta": round(worst, 3), **dec}

    meta = {"run_at": time.strftime("%Y-%m-%d %H:%M:%S"), "n_gold": len(gold),
            "n_passages": len(r.passages), "load_seconds": round(load_s, 1), "index": r.info,
            "env": env_fingerprint(cfg), "retrieval_cfg": r.r,
            "inputs_sha256": {"goldset": sha256_file(cfg.path("goldset")),
                              "passages": sha256_file(cfg.passages_path),
                              "prereg": sha256_file(PREREG) if PREREG.exists() else None},
            "baseline": BASELINE, "warmup": args.warmup,
            "latency_note": "单次查询耗时，进程内、模型已加载，每系统先预热 5 题；keyword 在内存里扫全部段落"}
    res = cfg.path("results_dir")
    write_json(res / "eval.json", {"meta": meta, "systems": out_sys, "vs_vector": comps})
    with open(res / "per_query.jsonl", "w", encoding="utf-8") as f:
        for it in gold:
            row = {"id": it["id"], "type": it["type"], "gold": it["gold"], "stale": it.get("stale", [])}
            for name in systems:
                rk = ranked[name][it["id"]]
                row[name] = {"first_gold_rank": per_q[name][it["id"]]["first_gold_rank"],
                             "recall5": round(per_q[name][it["id"]]["recall"], 3), "top5": rk[:5]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"写入 {_rel(res)}/eval.json、per_query.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
