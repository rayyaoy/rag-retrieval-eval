"""块 → 段落（检索单位）。段落 ID = f"{doc_id}#{序号:04d}"，金标答案就是这些 ID。

分段规则（两类语料相同）：
  1. 按结构切小节：10-K 按 Item 与粗体小标题；维基按 h2/h3/h4。
  2. 小节内按块顺序拼接，超过 max_chars 就收一段；段与段之间**不重叠**
     （金标是段落 ID，重叠会让同一句话同时落在两段里，Recall 的分母就说不清了）。
  3. 单块超长：先按句末标点切，仍超长再硬切。
  4. 去空白后不足 min_chars 的段丢弃。
  5. 每段带一个前缀「文档名 > 小节路径」，建向量时与正文一起送进模型；
     前缀单独存字段，消融实验（ablate_prefix）只编正文。
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from .htmlblocks import Block

ITEM_RE = re.compile(r"^item\s*(\d{1,2}[a-c]?)\s*[\.:]\s*(.*)$", re.I)
EDGAR_SCOPE = ["1", "1A", "1B", "1C", "2", "3", "4", "5", "6", "7", "7A"]
EDGAR_NOISE = [re.compile(p, re.I) for p in (
    r"^table of contents$", r"^\d{1,3}$", r"form 10-k \| \d+$", r"^\|?\s*\d{4} form 10-k",
    r"^part i{1,3}v?\.?$", r"^part i{1,3}v?\s*[-–—]", r"^\d{1,3} \| \d{4} (form )?10-k$",
    r"\| \d{4} form 10-k \| \d{1,3}$",
)]
WIKI_STOP = {"参考文献", "参考资料", "参考", "外部链接", "注释", "参见", "延伸阅读", "脚注",
             "相关条目", "来源", "注解", "引用", "外部連結", "參考文獻", "参考来源"}
SENT_RE = re.compile(r"(?<=[\.!?;。！？；])\s+|(?<=[。！？；])")


@dataclass
class Passage:
    pid: str
    doc_id: str
    source: str        # edgar | wiki | fixture
    lang: str          # en | zh
    title: str         # 文档名
    section: str       # 小节路径
    body: str

    @property
    def prefix(self) -> str:
        return f"{self.title} > {self.section}" if self.section else self.title

    @property
    def text(self) -> str:
        """送进向量模型 / BM25 的文本 = 前缀 + 换行 + 正文。"""
        return f"{self.prefix}\n{self.body}"

    def to_json(self) -> dict:
        return asdict(self)


def split_long(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    out, cur = [], ""
    for s in (x for x in SENT_RE.split(text) if x):
        if len(s) > max_chars:                     # 一句就超长：硬切
            if cur:
                out.append(cur)
                cur = ""
            out.extend(s[i:i + max_chars] for i in range(0, len(s), max_chars))
            continue
        sep = "" if (not cur or re.search(r"[一-鿿]$", cur)) else " "
        if cur and len(cur) + len(sep) + len(s) > max_chars:
            out.append(cur)
            cur = s
        else:
            cur = f"{cur}{sep}{s}" if cur else s
    if cur:
        out.append(cur)
    return out


def pack(sections, max_chars: int, min_chars: int):
    """sections = [(小节路径, [块文本...])] -> [(小节路径, 段正文)]"""
    out = []
    for path, texts in sections:
        cur: list[str] = []
        size = 0
        for t in texts:
            for piece in split_long(t, max_chars):
                if cur and size + len(piece) + 1 > max_chars:
                    out.append((path, "\n".join(cur)))
                    cur, size = [], 0
                cur.append(piece)
                size += len(piece) + 1
        if cur:
            out.append((path, "\n".join(cur)))
    return [(p, b) for p, b in out if len(re.sub(r"\s", "", b)) >= min_chars]


def _clip(s: str, n: int = 80) -> str:
    return s if len(s) <= n else s[:n].rstrip() + "…"


def edgar_sections(blocks: list[Block]):
    """只保留 Item 1–7A；Item 8 起停止。目录表（表格行）里的 Item 不算标题。"""
    sections, item, sub, buf = [], None, "", []

    def close():
        if item and buf:
            sections.append((" > ".join(x for x in (item, sub) if x), list(buf)))
        buf.clear()

    for b in blocks:
        t = b.text.strip()
        if any(r.search(t) for r in EDGAR_NOISE):
            continue
        m = ITEM_RE.match(t) if b.kind != "row" else None
        if m and len(t) < 200:
            num = m.group(1).upper()
            close()
            if num in EDGAR_SCOPE:
                item, sub = _clip(f"Item {num}. {m.group(2).strip()}".rstrip(". ")), ""
                continue
            if item:                       # 越过 7A 之后的第一个 Item（通常是 8）就结束
                break
            continue
        if item is None:
            continue
        if b.kind == "p" and b.bold and len(t) < 150:
            close()
            sub = _clip(t.rstrip(":"))
            continue
        buf.append(t)
    close()
    return sections


def wiki_sections(blocks: list[Block]):
    sections, path, buf = [], [], []

    def close():
        if buf:
            sections.append((" > ".join(path), list(buf)))
        buf.clear()

    for b in blocks:
        t = b.text.strip()
        if b.kind == "h" and b.level >= 2:
            close()
            if t in WIKI_STOP:
                break
            path = path[: b.level - 2] + [_clip(t, 60)]
            continue
        if b.kind == "h":
            continue
        buf.append(t)
    close()
    return sections


def make_passages(doc_id, source, lang, title, sections, max_chars=900, min_chars=20):
    return [Passage(f"{doc_id}#{i:04d}", doc_id, source, lang, title, path, body)
            for i, (path, body) in enumerate(pack(sections, max_chars, min_chars))]
