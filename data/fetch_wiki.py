#!/usr/bin/env python3
"""按 revision id 下载中文维基条目的渲染 HTML（zh-cn 字词转换）到 data/raw/wiki/。

    python data/fetch_wiki.py

用 MediaWiki parse API（oldid 钉死版本，variant=zh-cn）。条目正文是 CC BY-SA 4.0，
不随仓库分发，署名信息见 NOTICE。注意：同一 revision 在不同日期渲染，
「X 年前」之类按当天计算的模板输出可能不同，所以维基文件的 sha256 只做提示、不做硬校验；
金标证据串由 goldset/check_gold.py 逐字核对。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import tomllib
import urllib.parse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _http import ROOT, check_or_record, get, sha256  # noqa: E402

RAW = ROOT / "data" / "raw" / "wiki"
API = "https://zh.wikipedia.org/w/api.php"
UA = "rag-retrieval-eval/0.1 (retrieval evaluation research)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", action="store_true")
    args = ap.parse_args()
    with open(ROOT / "data" / "sources.toml", "rb") as fh:
        arts = tomllib.load(fh)["wiki"]
    RAW.mkdir(parents=True, exist_ok=True)
    for a in arts:
        dest = RAW / f"{a['doc_id']}.json"
        if not dest.exists():
            q = urllib.parse.urlencode({
                "action": "parse", "format": "json", "oldid": a["revid"], "prop": "text|revid|displaytitle",
                "variant": "zh-cn", "disableeditsection": 1, "disabletoc": 1})
            d = json.loads(get(f"{API}?{q}", {"User-Agent": UA}))
            if "error" in d:
                print(f"  {a['doc_id']}: {d['error']}")
                return 1
            p = d["parse"]
            out = {"doc_id": a["doc_id"], "title": a["title"], "revid": p["revid"],
                   "html": p["text"]["*"]}
            assert out["revid"] == a["revid"], f"revid 不一致：{out['revid']} != {a['revid']}"
            dest.write_text(json.dumps(out, ensure_ascii=False))
            time.sleep(2)                       # 对维基 API 客气一点
        body = dest.read_bytes()
        st = check_or_record("wiki_sha256_hint", dest.name, sha256(body), args.record)
        print(f"  {a['doc_id']:<24} rev {a['revid']}  {len(body):>9,} bytes  sha256 {st}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
