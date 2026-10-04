#!/usr/bin/env python3
"""原始 HTML → data/passages.jsonl（检索单位，带稳定段落 ID）。

    python data/build_passages.py

输出确定：同样的原始文件、同样的 chunk 参数，段落 ID 与内容逐字相同。
passages.jsonl 的 sha256 记进 data/manifest.json，便于核对复现。
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))
from rre.config import load_config  # noqa: E402
from rre.htmlblocks import html_to_blocks  # noqa: E402
from rre.passages import edgar_sections, make_passages, wiki_sections  # noqa: E402
from _http import check_or_record  # noqa: E402

WIKI_SKIP_CLASSES = ("infobox", "navbox", "reflist", "references", "reference", "mw-editsection",
                     "metadata", "ambox", "toc", "hatnote", "noprint", "sidebar", "mw-references-wrap",
                     "thumb", "gallery", "mw-empty-elt", "noexcerpt", "printfooter", "catlinks",
                     "vertical-navbox", "plainlinks", "mbox-small", "mw-kartographer", "geo")


def edgar_skip(tag, attrs):
    return tag in ("ix:header", "script", "style", "head")


def wiki_skip(tag, attrs):
    if tag in ("style", "script", "sup", "math", "annotation"):
        return True
    cls = attrs.get("class", "").split()
    return any(c in WIKI_SKIP_CLASSES or c.startswith(("navbox", "infobox")) for c in cls)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    mx, mn = cfg["chunk"]["max_chars"], cfg["chunk"]["min_chars"]
    with open(ROOT / "data" / "sources.toml", "rb") as fh:
        src = tomllib.load(fh)
    out = []
    for f in src["edgar"]:
        html = (ROOT / "data" / "raw" / "edgar" / f"{f['doc_id']}.htm").read_text(errors="ignore")
        secs = edgar_sections(html_to_blocks(html, edgar_skip))
        title = f"{f['company']} Form 10-K {f['fiscal_year']}"
        out += make_passages(f["doc_id"], "edgar", "en", title, secs, mx, mn)
    for a in src["wiki"]:
        d = json.loads((ROOT / "data" / "raw" / "wiki" / f"{a['doc_id']}.json").read_text())
        secs = wiki_sections(html_to_blocks(d["html"], wiki_skip))
        out += make_passages(a["doc_id"], "wiki", "zh", f"维基百科：{a['label']}", secs, mx, mn)

    dest = cfg.passages_path
    data = "".join(json.dumps(p.to_json(), ensure_ascii=False) + "\n" for p in out)
    dest.write_text(data)
    digest = hashlib.sha256(data.encode()).hexdigest()
    st = check_or_record("passages_sha256", f"max{mx}-min{mn}", digest, args.record)
    by_doc = collections.Counter(p.doc_id for p in out)
    lens = sorted(len(p.body) for p in out)
    for k, v in by_doc.items():
        print(f"  {k:<24} {v:>5} 段")
    print(f"共 {len(out)} 段（edgar {sum(p.source == 'edgar' for p in out)} / "
          f"wiki {sum(p.source == 'wiki' for p in out)}）；正文字数 P50 {lens[len(lens)//2]} "
          f"P95 {lens[int(len(lens)*.95)]} max {lens[-1]}")
    print(f"写入 {dest.relative_to(ROOT)}  sha256 {digest[:16]}…  manifest {st}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
