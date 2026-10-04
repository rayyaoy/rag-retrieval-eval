"""基线：加权关键词计数（原知识库检索工具的打分逻辑，改成段落粒度）。

原工具按文件打分：文件名命中 +100、标题行命中 +10、正文行命中 +1（每行只算一次）、
全部关键词都出现再加 50×关键词数，PDF 打七折。公开版检索单位是段落、没有文件名，
所以改成：前缀（文档名 > 小节路径）命中每个关键词 +10，正文每行命中 +1，
全部关键词都出现（前缀或正文）再加 50×关键词数。输入是用户会敲的关键词，不是问句。
同分保持语料顺序（与原工具的稳定排序一致）。
"""
from __future__ import annotations

import re
from typing import Sequence

from .passages import Passage


def keyword_scores(keywords: Sequence[str], passages: Sequence[Passage]) -> list[tuple[str, int]]:
    pats = [re.compile(re.escape(k), re.I) for k in keywords if k.strip()]
    if not pats:
        return []
    out = []
    for p in passages:
        score, matched = 0, set()
        for pat in pats:
            if pat.search(p.prefix):
                score += 10
                matched.add(pat.pattern)
        for line in p.body.splitlines():
            for pat in pats:
                if pat.search(line):
                    matched.add(pat.pattern)
                    score += 1
                    break
        # 「全部关键词都出现」：前缀命中也算，因为段落的前缀就相当于原工具的「文件名 + 标题」。
        # 注意正文每行只记第一个命中的关键词（原工具如此），同一行里的第二个关键词不计入 matched。
        if len(matched) == len(pats):
            score += 50 * len(pats)
        if score > 0:
            out.append((p.pid, score))
    out.sort(key=lambda x: -x[1])          # 稳定排序：同分保持语料顺序
    return out
