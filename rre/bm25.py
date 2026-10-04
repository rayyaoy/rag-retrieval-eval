"""Okapi BM25（k1=1.5, b=0.75），中文用 jieba 分词。倒排索引自己写，不依赖第三方 BM25 库。

分词规则（tokenize）：
  1. NFKC 全角转半角，转小写。
  2. 数字串整体保留：「130,497」「36,000」「2025-01-26」「1.5」「hbm3e」这类串整体抠出来
     当一个 token（去掉千分位逗号、去掉百分号）——数字出处题靠它；
     jieba 会把「9,138.83」切碎，碎片 IDF 很低。
  3. 其余文本走 jieba.lcut_for_search（搜索引擎模式，长词再切出子词），文档和查询用同一个函数。
  4. 丢弃：纯标点 / 空白、单个拉丁字母、停用词。中文单字不整体丢（「股」「价」有信息量）。
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Sequence

import jieba

jieba.setLogLevel(60)

ZH_STOP = set("""
的 了 和 与 及 或 而 也 都 就 还 又 在 是 为 对 从 把 被 让 给 向 于 以 由 其 之 这 那 这个 那个 这些 那些
一个 一种 一些 有 没有 没 不 很 更 最 再 已 已经 会 能 可以 可 要 需要 应 应该 该 得 地 着 过 吗 呢 吧 啊
我 我们 你 你们 他 她 它 他们 自己 什么 怎么 怎么样 如何 哪 哪些 哪个 多少 几 为什么 是否 是不是 有没有
如果 因为 所以 但是 但 然后 以及 并 并且 等 等等 即 即使 则 且 到底 到 上 下 中 里 后 前 时 时候 之后 之前
个 次 种 条 篇 份 些 每 各 此 本 其中 这样 那样 还是 只 只是 就是 还有
""".split())
EN_STOP = set("""
a an the and or but if then else of to in on at by for with from as is are was were be been being it its
this that these those i you he she we they me my your our their do does did not no so than too very can
will just about into over under up down out off what which who whom how why when where there here all any
""".split())
SYM_STOP = {"|", "-", "--", "---", "*", "**", "#", ">", "`", "·", "—", "…", "•"}
STOPWORDS = ZH_STOP | EN_STOP | SYM_STOP

NUM_RE = re.compile(r"[a-z0-9]*\d[a-z0-9]*(?:[.\-/:,]\d[a-z0-9]*)*%?")
PUNCT_RE = re.compile(r"^[\W_]+$", re.U)


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    toks: list[str] = []
    pos = 0
    for m in NUM_RE.finditer(text):
        toks.extend(_words(text[pos:m.start()]))
        toks.append(m.group(0).replace(",", "").rstrip("%"))
        pos = m.end()
    toks.extend(_words(text[pos:]))
    return toks


def _words(seg: str) -> list[str]:
    out = []
    for w in jieba.lcut_for_search(seg):
        w = w.strip()
        if not w or w in STOPWORDS or PUNCT_RE.match(w):
            continue
        if len(w) == 1 and w.isascii():
            continue
        out.append(w)
    return out


class BM25:
    def __init__(self, ids: Sequence[str], docs_tokens: Sequence[Sequence[str]],
                 k1: float = 1.5, b: float = 0.75):
        if len(ids) != len(docs_tokens):
            raise ValueError("ids 与 docs_tokens 长度不一致")
        self.k1, self.b = k1, b
        self.ids = list(ids)
        self.N = len(docs_tokens)
        self.dl = [len(t) for t in docs_tokens]
        self.avgdl = sum(self.dl) / max(1, self.N)
        self.post: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, toks in enumerate(docs_tokens):
            for t, tf in Counter(toks).items():
                self.post[t].append((i, tf))
        self.idf = {t: math.log(1 + (self.N - len(p) + 0.5) / (len(p) + 0.5))
                    for t, p in self.post.items()}

    @classmethod
    def from_texts(cls, ids: Sequence[str], texts: Sequence[str], **kw) -> "BM25":
        return cls(ids, [tokenize(t) for t in texts], **kw)

    def scores(self, query: str) -> dict[int, float]:
        sc: dict[int, float] = defaultdict(float)
        for t in set(tokenize(query)):
            if t not in self.post:
                continue
            idf = self.idf[t]
            for i, tf in self.post[t]:
                norm = tf + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl)
                sc[i] += idf * tf * (self.k1 + 1) / norm
        return sc

    def search(self, query: str, limit: int | None = None) -> list[tuple[str, float]]:
        """返回 [(id, score)]，分数降序；同分按语料顺序。"""
        res = sorted(self.scores(query).items(), key=lambda x: (-x[1], x[0]))
        out = [(self.ids[i], s) for i, s in res]
        return out[:limit] if limit else out
