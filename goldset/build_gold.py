#!/usr/bin/env python3
"""questions.jsonl（人写的题目 + 证据串）→ goldset.jsonl（解析出的金标段落 ID）。

    python goldset/build_gold.py            # 解析并写入 goldset.jsonl；有任何一题不合格就不写、退出码 1
    python goldset/build_gold.py --dry-run --questions draft.jsonl   # 只校验草稿，不写文件
    python goldset/build_gold.py --check    # 只核对现有 goldset.jsonl 与重新解析的结果是否一致

核对内容：字段齐全、id 不重复、题型合法、每个证据串都逐字命中至少一段、
stale 题的旧证据也命中且与金标不重叠、goldset.jsonl 里的金标与重新解析结果完全一致。
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from rre.config import load_config  # noqa: E402
from rre.corpus import load_passages  # noqa: E402
from rre.gold import build_goldset, dump_jsonl, load_jsonl  # noqa: E402

QUESTIONS = ROOT / "goldset" / "questions.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--questions", default=str(QUESTIONS))
    ap.add_argument("--dry-run", action="store_true", help="只校验题目，不写 goldset.jsonl")
    args = ap.parse_args()
    cfg = load_config()
    passages = load_passages(cfg.passages_path)
    items, problems = build_goldset(load_jsonl(args.questions), passages)
    for qid, errs in problems.items():
        print(f"✗ {qid}: " + "；".join(errs))
    by_type = collections.Counter(i["type"] for i in items)
    ng = [len(i.get("gold", [])) for i in items]
    print(f"题目 {len(items)} 道，不合格 {len(problems)}；题型 {dict(by_type)}；"
          f"每题金标段数 min {min(ng)} / max {max(ng)} / 均值 {sum(ng)/len(ng):.2f}")
    if problems:
        return 1
    gold_path = cfg.path("goldset")
    if args.dry_run:
        for i in items:
            print(f"  {i['id']:<8} {len(i['gold'])} 段 {i['gold'][:3]}" + (f"  stale {i['stale']}" if i.get('stale') else ""))
        return 0
    if args.check:
        old = load_jsonl(gold_path)
        if old != items:
            diff = [a["id"] for a, b in zip(old, items) if a != b]
            print(f"✗ goldset.jsonl 与重新解析结果不一致（{len(diff) or '条数不同'}）：{diff[:10]}")
            return 1
        print("✓ goldset.jsonl 与重新解析结果一致")
        return 0
    dump_jsonl(items, gold_path)
    print(f"写入 {gold_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
