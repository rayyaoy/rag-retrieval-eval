#!/usr/bin/env python3
"""下载 sources.toml 里钉死的 10-K 原始 HTML 到 data/raw/edgar/。

    SEC_USER_AGENT="Your Name you@example.com" python data/fetch_edgar.py
    python data/fetch_edgar.py --wayback-only      # 直接走互联网档案馆的原样快照

SEC 要求自动化请求在 User-Agent 里写明身份（https://www.sec.gov/os/accessing-edgar-data），
所以必须设 SEC_USER_AGENT。某些出口 IP 会被 SEC 整体拒绝（403 "Undeclared Automated Tool"），
这时退回 web.archive.org 的原样快照（URL 带 id_，返回未改写的原始字节）。
两条路下载到的文件都用 data/manifest.json 里的 sha256 核对。
"""
from __future__ import annotations

import argparse
import os
import sys
import tomllib
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _http import ROOT, check_or_record, get, sha256  # noqa: E402

RAW = ROOT / "data" / "raw" / "edgar"


def sec_url(f: dict) -> str:
    return (f"https://www.sec.gov/Archives/edgar/data/{f['cik']}/"
            f"{f['accession'].replace('-', '')}/{f['document']}")


def fetch_one(f: dict, ua: str | None, wayback_only: bool) -> tuple[bytes, str]:
    url = sec_url(f)
    if ua and not wayback_only:
        try:
            return get(url, {"User-Agent": ua}), "sec.gov"
        except urllib.error.HTTPError as e:
            print(f"  sec.gov 返回 {e.code}，改走 web.archive.org")
        except Exception as e:  # noqa: BLE001
            print(f"  sec.gov 失败（{e!r}），改走 web.archive.org")
    wb = f"https://web.archive.org/web/2026id_/{url}"
    return get(wb, {"User-Agent": ua or "rag-retrieval-eval (research)"}, timeout=180), "web.archive.org"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wayback-only", action="store_true")
    ap.add_argument("--record", action="store_true", help="manifest 里没有记录时写入（维护者用）")
    args = ap.parse_args()
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua and not args.wayback_only:
        print("未设置 SEC_USER_AGENT，直接走 web.archive.org（设了就优先 sec.gov）")
    with open(ROOT / "data" / "sources.toml", "rb") as fh:
        filings = tomllib.load(fh)["edgar"]
    RAW.mkdir(parents=True, exist_ok=True)
    bad = 0
    for f in filings:
        dest = RAW / f"{f['doc_id']}.htm"
        if dest.exists():
            body, src = dest.read_bytes(), "cache"
        else:
            print(f"下载 {f['doc_id']} {f['document']}")
            body, src = fetch_one(f, ua, args.wayback_only)
            dest.write_bytes(body)
        st = check_or_record("raw_sha256", dest.name, sha256(body), args.record)
        print(f"  {f['doc_id']:<20} {len(body):>9,} bytes  来源 {src:<16} sha256 {st}")
        bad += st == "MISMATCH"
    if bad:
        print(f"{bad} 个文件与 manifest 不一致：删掉 data/raw/edgar/ 里对应文件重下，或检查来源")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
