"""RRF（Reciprocal Rank Fusion）融合。

score(d) = Σ_i w_i / (k + rank_i(d))，rank 从 1 起；只在某一路出现的文档只加那一路。
同分时按「首次出现的列表顺序、列表内名次」稳定排序，保证结果确定。
"""
from __future__ import annotations

from typing import Sequence


def rrf(rank_lists: Sequence[Sequence[str]], k: int = 60,
        weights: Sequence[float] | None = None) -> tuple[list[str], dict[str, float]]:
    if weights is not None and len(weights) != len(rank_lists):
        raise ValueError("weights 与 rank_lists 长度不一致")
    score: dict[str, float] = {}
    order: dict[str, int] = {}
    for li, lst in enumerate(rank_lists):
        w = 1.0 if weights is None else float(weights[li])
        for r, doc in enumerate(lst, 1):
            score[doc] = score.get(doc, 0.0) + w / (k + r)
            order.setdefault(doc, len(order))
    ranked = sorted(score, key=lambda d: (-score[d], order[d]))
    return ranked, score
