"""五路检索 + 三个预注册变体。所有方法返回按分数降序的段落 ID 列表。

  keyword   加权关键词计数（基线，输入是关键词）
  bm25      BM25 + jieba（输入是问句）
  vector    BGE-M3 稠密向量 + Qdrant
  rrf       BM25 前 N + 向量前 N，RRF(k=60) 融合，不精排
  hybrid    rrf 后取前 rerank_top 个送 bge-reranker-v2-m3 精排；其后的保持融合顺序
  V1 vec_rerank  精排池只取向量前 rerank_top（不掺 BM25）
  V2 wrrf        加权 RRF：向量权重 1、BM25 权重 0.5，不精排
  V3 route       问句含阿拉伯数字 → hybrid；否则 → vector
"""
from __future__ import annotations

import re
from typing import Sequence

from .bm25 import BM25
from .fusion import rrf
from .index import vector_search
from .keyword import keyword_scores
from .passages import Passage

HAS_DIGIT = re.compile(r"\d")
DEFAULT_RCFG = {"vector_candidates": 100, "topn_each": 20, "rrf_k": 60, "rerank_top": 20}


class Retriever:
    def __init__(self, passages: Sequence[Passage], embedder=None, client=None,
                 collection: str | None = None, reranker=None, rcfg: dict | None = None,
                 bm25: BM25 | None = None):
        self.passages = list(passages)
        self.by_id = {p.pid: p for p in self.passages}
        self.embedder, self.client, self.collection = embedder, client, collection
        self.reranker = reranker
        self.r = {**DEFAULT_RCFG, **(rcfg or {})}
        self.bm25 = bm25 or BM25.from_texts([p.pid for p in self.passages],
                                            [p.text for p in self.passages])

    # ---- 单路 ----
    def keyword(self, keywords: Sequence[str]) -> list[str]:
        return [pid for pid, _ in keyword_scores(keywords, self.passages)]

    def bm25_search(self, query: str) -> list[str]:
        return [pid for pid, _ in self.bm25.search(query)]

    def vector(self, query: str, limit: int | None = None) -> list[str]:
        q = self.embedder.encode([query], progress=False)[0]
        return [pid for pid, _ in vector_search(self.client, self.collection, q,
                                                limit or self.r["vector_candidates"])]

    # ---- 融合与精排 ----
    def _rerank(self, query: str, pids: Sequence[str]) -> list[str]:
        scores = self.reranker.score(query, [self.by_id[p].text for p in pids])
        order = sorted(range(len(pids)), key=lambda i: (-scores[i], i))
        return [pids[i] for i in order]

    def rrf(self, query: str) -> list[str]:
        n = self.r["topn_each"]
        order, _ = rrf([self.bm25_search(query)[:n], self.vector(query, n)], k=self.r["rrf_k"])
        return order

    def hybrid(self, query: str) -> list[str]:
        order = self.rrf(query)
        k = self.r["rerank_top"]
        return self._rerank(query, order[:k]) + order[k:]

    # ---- 预注册变体 ----
    def vec_rerank(self, query: str) -> list[str]:
        v = self.vector(query)
        k = self.r["rerank_top"]
        return self._rerank(query, v[:k]) + v[k:]

    def wrrf(self, query: str, w_bm25: float = 0.5) -> list[str]:
        n = self.r["topn_each"]
        order, _ = rrf([self.vector(query, n), self.bm25_search(query)[:n]],
                       k=self.r["rrf_k"], weights=[1.0, w_bm25])
        return order

    def route(self, query: str) -> list[str]:
        return self.hybrid(query) if HAS_DIGIT.search(query) else self.vector(query)


def build_retriever(cfg, need_vector: bool = True, need_rerank: bool = False,
                    with_prefix: bool = True, embedder=None, reranker=None, progress: bool = True):
    """按配置拼出 Retriever：读段落 → 编码（命中缓存则跳过）→ 灌 Qdrant → 可选加载精排模型。"""
    from .corpus import load_passages
    from .index import encode_cached, ensure_collection, open_qdrant, texts_for
    from .models import CrossEncoderReranker, STEmbedder

    passages = load_passages(cfg.passages_path)
    m = cfg["models"]
    client = coll = None
    info = {}
    if need_vector:
        embedder = embedder or STEmbedder(m["embedder"], m["max_seq_length"], m["device"], m["batch_size"])
        texts = texts_for(passages, with_prefix)
        vecs, key, secs = encode_cached(embedder, texts, cfg.work_dir, m["max_seq_length"], progress)
        client = open_qdrant(cfg["qdrant"]["location"], cfg.work_dir)
        coll = cfg["qdrant"]["collection"] + ("" if with_prefix else "_noprefix")
        rebuilt = ensure_collection(client, coll, vecs, [p.pid for p in passages], key, cfg.work_dir)
        info = {"vector_cache_id": key, "encode_seconds_this_run": round(secs, 1), "collection": coll,
                "collection_rebuilt": rebuilt, "device": getattr(embedder, "device", "?"),
                "dtype": getattr(embedder, "dtype", "?")}
    if need_rerank:
        reranker = reranker or CrossEncoderReranker(m["reranker"], m["reranker_max_length"], m["device"])
    r = Retriever(passages, embedder, client, coll, reranker, cfg["retrieval"])
    r.info = info
    return r
