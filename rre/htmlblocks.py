"""把 HTML 拆成「块」：段落 / 标题 / 表格行。只用标准库 html.parser。

每个块记录：text（空白已压缩）、kind（p / h / row）、level（h1–h6 的级别，非标题为 0）、
bold（块内全部文字都是粗体——10-K 的小标题没有 h 标签，只能靠这个认）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from html.parser import HTMLParser

BLOCK_TAGS = {"div", "p", "li", "ul", "ol", "dl", "dt", "dd", "table", "tbody", "thead",
              "blockquote", "section", "br", "h1", "h2", "h3", "h4", "h5", "h6", "tr",
              "figure", "figcaption", "center"}
VOID_TAGS = {"br", "img", "hr", "meta", "link", "input", "col", "area", "base", "wbr", "source"}
BOLD_RE = re.compile(r"font-weight:\s*(bold|[6-9]00)", re.I)
HIDDEN_RE = re.compile(r"display:\s*none", re.I)
WS_RE = re.compile(r"\s+")


@dataclass
class Block:
    text: str
    kind: str          # p | h | row
    level: int = 0
    bold: bool = False


class _Parser(HTMLParser):
    def __init__(self, skip):
        super().__init__(convert_charrefs=True)
        self.skip = skip               # skip(tag, attrs_dict) -> bool
        self.stack = []                # [(tag, skipped, bold)]
        self.blocks: list[Block] = []
        self.runs: list[tuple[str, bool]] = []
        self.row: list[str] | None = None
        self.cell: list[str] | None = None
        self.heading = 0

    # ---- 状态 ----
    def _skipping(self):
        return any(s for _, s, _ in self.stack)

    def _bold(self):
        return any(b for _, _, b in self.stack)

    def _flush(self):
        text = WS_RE.sub(" ", "".join(t for t, _ in self.runs)).strip()
        if text:
            vis = [(t, b) for t, b in self.runs if t.strip()]
            bold = bool(vis) and all(b for _, b in vis)
            kind = "h" if self.heading else "p"
            self.blocks.append(Block(text, kind, self.heading, bold))
        self.runs = []

    # ---- 回调 ----
    def handle_starttag(self, tag, attrs):
        a = {k: (v or "") for k, v in attrs}
        if tag in VOID_TAGS:
            if tag == "br" and not self._skipping():
                if self.cell is not None:
                    self.cell.append(" ")
                else:
                    self._flush()
            return
        skipped = self.skip(tag, a) or bool(HIDDEN_RE.search(a.get("style", "")))
        bold = tag in ("b", "strong") or bool(BOLD_RE.search(a.get("style", "")))
        if self._skipping():
            self.stack.append((tag, True, bold))
            return
        if tag == "tr":
            self._flush()
            self.row = []
        elif tag in ("td", "th") and self.row is not None:
            self.cell = []
        elif tag in BLOCK_TAGS and self.row is None:
            self._flush()
            if re.fullmatch(r"h[1-6]", tag):
                self.heading = int(tag[1])
        self.stack.append((tag, skipped, bold))

    def handle_endtag(self, tag):
        if tag in VOID_TAGS:
            return
        # 找到最近的同名标签出栈（容忍不规范嵌套）
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                was_skipping = self._skipping()
                del self.stack[i:]
                if was_skipping and self._skipping():
                    return
                if was_skipping:            # 刚离开跳过区
                    return
                break
        else:
            return
        if tag in ("td", "th") and self.cell is not None and self.row is not None:
            self.row.append(WS_RE.sub(" ", "".join(self.cell)).strip())
            self.cell = None
        elif tag == "tr" and self.row is not None:
            cells = [c for c in self.row if c]
            if cells:
                text = " | ".join(cells)
                text = re.sub(r"\$ \| ", "$", text)
                text = re.sub(r" \| (%|\))", r"\1", text)
                text = re.sub(r"\( \| ", "(", text)
                self.blocks.append(Block(text, "row"))
            self.row = None
        elif tag in BLOCK_TAGS and self.row is None:
            self._flush()
            if re.fullmatch(r"h[1-6]", tag):
                self.heading = 0

    def handle_data(self, data):
        if self._skipping():
            return
        if self.cell is not None:
            self.cell.append(data)
        elif self.row is not None:
            return                          # 行内但不在单元格里的空白
        else:
            self.runs.append((data, self._bold()))

    def close(self):
        super().close()
        self._flush()


def html_to_blocks(html: str, skip=lambda tag, attrs: False) -> list[Block]:
    p = _Parser(skip)
    p.feed(html)
    p.close()
    return p.blocks
