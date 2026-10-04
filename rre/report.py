#!/usr/bin/env python3
"""把 results/public/*.json 渲染成 results/public/RESULTS.md（README 的表从这里抄，不手填数字）。

    python -m rre.report
"""
from __future__ import annotations

import json

from .config import ROOT, load_config


def _rel(p):
    try:
        return p.relative_to(ROOT)
    except ValueError:
        return p
from .evaluate import TYPE_ZH, TYPES


def ms(x: float) -> str:
    return f"{x:,.0f}" if x >= 10 else f"{x:.1f}"

MAIN = ["keyword", "bm25", "vector", "rrf", "hybrid"]
VARIANTS = ["rrf", "hybrid", "V1_vec_rerank", "V2_wrrf", "V3_route"]
NAME_ZH = {"keyword": "关键词加权计数（基线）", "bm25": "BM25（jieba）", "vector": "向量（BGE-M3）",
           "rrf": "RRF 融合（不精排）", "hybrid": "RRF + bge-reranker-v2-m3",
           "V1_vec_rerank": "V1 精排池只取向量前 20", "V2_wrrf": "V2 加权 RRF（BM25 权重 0.5）",
           "V3_route": "V3 问句带数字走混合，否则走向量"}
INPUT = {"keyword": "关键词", "bm25": "问句", "vector": "问句", "rrf": "问句", "hybrid": "问句"}


def render(ev: dict, ab: dict | None) -> str:
    S, C, meta = ev["systems"], ev["vs_vector"], ev["meta"]
    L = [f"# 公开语料评测结果\n",
         f"> 自动生成（`python -m rre.report`），数据来自 `eval.json`（{meta['run_at']}）"
         + (f"与 `ablate_prefix.json`（{ab['run_at']}）" if ab else "") + "。",
         f"> 金标题 {meta['n_gold']} 道，段落 {meta['n_passages']} 个；指标按段落计。"
         f"设备 {meta['index'].get('device')} / {meta['index'].get('dtype')}；"
         f"{meta['env'].get('system')} {meta['env'].get('machine')}，torch {meta['env'].get('torch')}。\n"]
    ceil = S["vector"]["metrics"]["ALL"]["recall@5_ceiling"]
    L += ["## 主表（五路对照）\n",
          f"| 系统 | 输入 | Recall@5（天花板 {ceil:.3f}） | Hit@5 | MRR@10 | P50 ms | P95 ms |",
          "|---|---|---|---|---|---|---|"]
    for n in MAIN:
        if n not in S:
            continue
        m, l = S[n]["metrics"]["ALL"], S[n]["latency_ms"]
        L.append(f"| {NAME_ZH[n]} | {INPUT[n]} | {m['recall@5']:.3f} | {m['hit@5']:.3f} | "
                 f"{m['mrr@10']:.3f} | {ms(l['p50'])} | {ms(l['p95'])} |")
    L += ["", "## 按题型（Recall@5 / MRR@10）\n",
          "| 题型 | n | " + " | ".join(NAME_ZH[n] for n in MAIN if n in S) + " |",
          "|---|---|" + "---|" * sum(n in S for n in MAIN)]
    for t in TYPES:
        cells = [f"{S[n]['metrics'][t]['recall@5']:.3f} / {S[n]['metrics'][t]['mrr@10']:.3f}"
                 for n in MAIN if n in S]
        L.append(f"| {TYPE_ZH[t]} | {S['vector']['metrics'][t]['n']} | " + " | ".join(cells) + " |")
    L += ["", "Hit@5 按题型：\n", "| 题型 | " + " | ".join(NAME_ZH[n] for n in MAIN if n in S) + " |",
          "|---|" + "---|" * sum(n in S for n in MAIN)]
    for t in TYPES:
        L.append(f"| {TYPE_ZH[t]} | " + " | ".join(f"{S[n]['metrics'][t]['hit@5']:.3f}" for n in MAIN if n in S) + " |")

    L += ["", "## 相对纯向量：配对 bootstrap 与预注册换线条件\n",
          "条件（全部满足才换线）：ΔRecall@5 ≥ 0.03 且 95% 区间下界 > 0；任一题型掉分不超过 0.03；P95 < 1,000 ms。\n",
          "| 系统 | Recall@5 | ΔR@5 [95% 区间] | 最差题型 Δ | " + " | ".join(TYPE_ZH[t] for t in TYPES)
          + " | P50 / P95 ms | 满足换线 |", "|---|---|---|---|" + "---|" * len(TYPES) + "---|---|"]
    vm = S["vector"]["metrics"]
    L.append(f"| {NAME_ZH['vector']}（基线） | {vm['ALL']['recall@5']:.3f} | — | — | "
             + " | ".join(f"{vm[t]['recall@5']:.3f}" for t in TYPES)
             + f" | {ms(S['vector']['latency_ms']['p50'])} / {ms(S['vector']['latency_ms']['p95'])} | — |")
    for n in VARIANTS:
        if n not in C:
            continue
        c, m, l = C[n], S[n]["metrics"], S[n]["latency_ms"]
        failed = [k for k, v in c["checks"].items() if not v]
        verdict = "是" if c["switch"] else "否（" + "、".join(failed) + "）"
        L.append(f"| {NAME_ZH[n]} | {m['ALL']['recall@5']:.3f} | {c['delta_recall5']:+.3f} "
                 f"[{c['boot95'][0]:+.3f}, {c['boot95'][1]:+.3f}] | {c['worst_type_delta']:+.3f} | "
                 + " | ".join(f"{m[t]['recall@5']:.3f}" for t in TYPES)
                 + f" | {ms(l['p50'])} / {ms(l['p95'])} | {verdict} |")

    L += ["", "## 过期内容（同一公司两年年报并存，16 题）\n",
          "| 系统 | 第 1 名是旧年份段落 | 相关段落里先读到旧年份 | 前 5 名中旧段落占比 | 前 5 名含最新年份段落 |",
          "|---|---|---|---|---|"]
    for n in MAIN + ["V1_vec_rerank"]:
        if n in S and S[n]["stale"]:
            s = S[n]["stale"]
            L.append(f"| {NAME_ZH[n]} | {s['stale_top1']:.3f} | {s['first_is_stale']:.3f} | "
                     f"{s['stale_in_top5']:.3f} | {s['current_found@5']:.3f} |")
    if ab:
        L += ["", "## 分段前缀消融（纯向量）\n",
              "| 版本 | Recall@5 | Hit@5 | MRR@10 | " + " | ".join(f"{TYPE_ZH[t]} R@5" for t in TYPES) + " |",
              "|---|---|---|---|" + "---|" * len(TYPES)]
        for n, zh in (("with_prefix", "有前缀（文档名 > 小节路径）"), ("no_prefix", "无前缀（只编正文）")):
            m = ab["metrics"][n]
            L.append(f"| {zh} | {m['ALL']['recall@5']:.3f} | {m['ALL']['hit@5']:.3f} | {m['ALL']['mrr@10']:.3f} | "
                     + " | ".join(f"{m[t]['recall@5']:.3f}" for t in TYPES) + " |")
        if ab.get("stale"):
            s1, s0 = ab["stale"]["with_prefix"], ab["stale"]["no_prefix"]
            L.append(f"\n过期内容题上「第 1 名是旧年份段落」：有前缀 {s1['stale_top1']:.3f}，无前缀 {s0['stale_top1']:.3f}；"
                     f"「相关段落里先读到旧年份」：有前缀 {s1['first_is_stale']:.3f}，无前缀 {s0['first_is_stale']:.3f}。")
        L.append(f"\nΔRecall@5（有 − 无）= {ab['delta_recall5']:+.3f}，配对 bootstrap 95% 区间 "
                 f"[{ab['boot95'][0]:+.3f}, {ab['boot95'][1]:+.3f}]。")
    L += ["", "## 输入指纹\n", "```", json.dumps(meta["inputs_sha256"], indent=1), "```", ""]
    return "\n".join(L)


def main() -> int:
    cfg = load_config()
    d = cfg.path("results_dir")
    ev = json.loads((d / "eval.json").read_text())
    abp = d / "ablate_prefix.json"
    ab = json.loads(abp.read_text()) if abp.exists() else None
    (d / "RESULTS.md").write_text(render(ev, ab), encoding="utf-8")
    print(f"写入 {_rel((d / 'RESULTS.md'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
