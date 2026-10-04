from rre.gold import build_goldset, resolve
from rre.htmlblocks import html_to_blocks
from rre.passages import (Passage, edgar_sections, make_passages, pack, split_long,
                          wiki_sections)

EDGAR_HTML = """
<html><body>
<ix:header><div>hidden xbrl 999999</div></ix:header>
<table><tr><td>Item 1.</td><td>Business</td><td>4</td></tr></table>
<div><span style="font-weight:700">Item 1. Business</span></div>
<div><span style="font-weight:700">Our Company</span></div>
<div><span>NVIDIA pioneered accelerated computing.</span></div>
<div style="display:none">should not appear</div>
<table><tr><td>Revenue</td><td>$</td><td>130,497</td><td>(</td><td>5</td><td>)%</td></tr></table>
<div>Table of Contents</div>
<div>12</div>
<div><span style="font-weight:700">Item 7. Management's Discussion</span></div>
<div>Revenue grew strongly.</div>
<div><span style="font-weight:700">Item 8. Financial Statements</span></div>
<div>Balance sheet text that must be excluded.</div>
</body></html>
"""


def test_html_blocks_tables_hidden_bold():
    blocks = html_to_blocks(EDGAR_HTML, lambda t, a: t == "ix:header")
    texts = [b.text for b in blocks]
    assert not any("hidden xbrl" in t for t in texts)
    assert not any("should not appear" in t for t in texts)
    assert "Revenue | $130,497 | (5)%" in texts
    assert any(b.bold and b.text == "Our Company" for b in blocks)


def test_edgar_sections_scope_and_headings():
    secs = edgar_sections(html_to_blocks(EDGAR_HTML, lambda t, a: t == "ix:header"))
    paths = [p for p, _ in secs]
    assert paths == ["Item 1. Business > Our Company", "Item 7. Management's Discussion"]
    body = " ".join(" ".join(t) for _, t in secs)
    assert "Table of Contents" not in body and "Balance sheet" not in body
    assert "130,497" in body


def test_wiki_sections_stop_at_references():
    html = ("<h2>历史</h2><p>第一段。</p><h3>早期</h3><p>第二段。</p>"
            "<h2>参考文献</h2><p>不应收录</p>")
    secs = wiki_sections(html_to_blocks(html))
    assert secs == [("历史", ["第一段。"]), ("历史 > 早期", ["第二段。"])]


def test_split_long_and_pack_no_overlap():
    s = "A" * 50 + ". " + "B" * 50 + ". " + "C" * 50 + "."
    parts = split_long(s, 60)
    assert all(len(p) <= 60 for p in parts) and "".join(parts).replace(" ", "") == s.replace(" ", "")
    packed = pack([("sec", ["x" * 30, "y" * 30, "z" * 30])], max_chars=70, min_chars=5)
    assert [b for _, b in packed] == ["x" * 30 + "\n" + "y" * 30, "z" * 30]   # 不重叠


def test_make_passages_ids_and_prefix():
    ps = make_passages("doc-a", "wiki", "zh", "维基百科：甲", [("历史", ["一" * 30]), ("", ["二" * 30])])
    assert [p.pid for p in ps] == ["doc-a#0000", "doc-a#0001"]
    assert ps[0].text == "维基百科：甲 > 历史\n" + "一" * 30
    assert ps[1].prefix == "维基百科：甲"


def _ps():
    return [Passage("n25#0", "edgar-n-fy2025", "edgar", "en", "T", "S", "we had approximately 36,000  employees"),
            Passage("n24#0", "edgar-n-fy2024", "edgar", "en", "T", "S", "we had approximately 29,600 employees"),
            Passage("z#0", "zhwiki-a", "wiki", "zh", "T", "S", "所以需要周期性刷新")]


def test_resolve_prefix_and_whitespace():
    gold, miss = resolve([{"doc": "edgar-n", "text": "approximately 36,000 employees"}], _ps())
    assert gold == ["n25#0"] and miss == []
    gold, miss = resolve([{"doc": "zhwiki", "text": "不存在"}], _ps())
    assert gold == [] and len(miss) == 1


def test_build_goldset_errors():
    qs = [
        {"id": "a", "type": "stale", "query": "q", "keywords": ["k"],
         "evidence": [{"doc": "edgar-n-fy2025", "text": "36,000"}],
         "stale_evidence": [{"doc": "edgar-n-fy2024", "text": "29,600"}]},
        {"id": "b", "type": "stale", "query": "q", "keywords": ["k"],
         "evidence": [{"doc": "edgar-n", "text": "employees"}],          # 两年都命中
         "stale_evidence": [{"doc": "edgar-n-fy2024", "text": "29,600"}]},
        {"id": "a", "type": "bogus", "query": "q", "keywords": ["k"],
         "evidence": [{"doc": "zhwiki", "text": "刷新"}]},
    ]
    items, problems = build_goldset(qs, _ps())
    assert items[0]["gold"] == ["n25#0"] and items[0]["stale"] == ["n24#0"]
    assert "a" not in problems or problems["a"] == ["未知题型 bogus", "id 重复"]
    assert problems["b"] == ["stale 与金标重叠"]
