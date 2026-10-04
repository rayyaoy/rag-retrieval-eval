"""指标、配对 bootstrap、预注册判定条件。纯函数，无第三方依赖，单元测试覆盖。

指标都按「段落」计（检索单位 = 段落，金标 = 段落 ID 集合）：
  Recall@5  = |前5 ∩ 金标| / |金标|，按题平均。金标多于 5 段的题满分到不了 1，另报天花板。
  Hit@5     = 前 5 里至少有一个金标的题目占比。
  MRR@10    = 1 / 第一个金标的名次（前 10 没有记 0），按题平均。
  延迟       = 单次查询耗时（进程内、模型已加载），取 P50 / P95。
"""
from __future__ import annotations

import random
from typing import Iterable, Sequence

K_RECALL = 5
K_MRR = 10


def query_metrics(ranked: Sequence[str], gold: Iterable[str], k_recall: int = K_RECALL,
                  k_mrr: int = K_MRR) -> dict:
    gold = set(gold)
    if not gold:
        raise ValueError("金标为空")
    top = set(ranked[:k_recall])
    hit_set = gold & top
    rr = 0.0
    for i, p in enumerate(ranked[:k_mrr], 1):
        if p in gold:
            rr = 1.0 / i
            break
    first = next((i for i, p in enumerate(ranked, 1) if p in gold), None)
    return {"recall": len(hit_set) / len(gold), "hit": 1.0 if hit_set else 0.0, "rr": rr,
            "first_gold_rank": first, "ceiling": min(len(gold), k_recall) / len(gold)}


def summarize(rows: Sequence[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    return {"n": n,
            "recall@5": round(sum(r["recall"] for r in rows) / n, 3),
            "hit@5": round(sum(r["hit"] for r in rows) / n, 3),
            "mrr@10": round(sum(r["rr"] for r in rows) / n, 3),
            "recall@5_ceiling": round(sum(r["ceiling"] for r in rows) / n, 3)}


def percentile(xs: Sequence[float], q: float) -> float:
    """最近秩法（与原框架一致）：排序后取 round(q*(n-1)) 位置。"""
    if not xs:
        raise ValueError("空序列")
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def paired_bootstrap_ci(diffs: Sequence[float], n_boot: int = 10_000, seed: int = 0,
                        alpha: float = 0.05) -> tuple[float, float]:
    """配对 bootstrap：diffs 是逐题差值（系统 A − 系统 B），按题有放回重抽 n_boot 次，
    取均值分布的 [alpha/2, 1-alpha/2] 分位数。固定种子，结果可复现。"""
    k = len(diffs)
    if k == 0:
        raise ValueError("空序列")
    rng = random.Random(seed)
    means = sorted(sum(diffs[rng.randrange(k)] for _ in range(k)) / k for _ in range(n_boot))
    lo = means[int((alpha / 2) * n_boot)]
    hi = means[int((1 - alpha / 2) * n_boot) - 1]
    return lo, hi


# 预注册的换线条件（prereg/ 下的文件写明，跑数前定死）
SWITCH_RULES = {"min_delta": 0.03, "max_type_drop": 0.03, "max_p95_ms": 1000.0}


def switch_decision(delta: float, ci_low: float, worst_type_delta: float, p95_ms: float,
                    rules: dict = SWITCH_RULES) -> dict:
    """全部满足才把基线（纯向量）换成候选系统。

    1. 总 Recall@5 比基线高 >= min_delta，且配对 bootstrap 95% 区间下界 > 0；
    2. 任一题型 Recall@5 都不比基线低超过 max_type_drop；
    3. 单次查询 P95 < max_p95_ms。
    """
    checks = {
        "delta>=min": delta >= rules["min_delta"] - 1e-12,
        "ci_low>0": ci_low > 0,
        "no_type_drop": worst_type_delta >= -rules["max_type_drop"] - 1e-12,
        "p95<max": p95_ms < rules["max_p95_ms"],
    }
    return {"checks": checks, "switch": all(checks.values())}
