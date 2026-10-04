import pytest

from rre.bm25 import BM25, tokenize
from rre.fusion import rrf
from rre.keyword import keyword_scores
from rre.passages import Passage


# ---------- RRF ----------

def test_rrf_scores_by_hand():
    order, sc = rrf([["a", "b", "c"], ["b", "d"]], k=60)
    assert sc["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert sc["a"] == pytest.approx(1 / 61)
    assert sc["d"] == pytest.approx(1 / 62)
    assert order[0] == "b"
    assert set(order) == {"a", "b", "c", "d"}


def test_rrf_tie_break_is_first_appearance():
    order, _ = rrf([["a"], ["b"]], k=60)          # 同分：先出现的在前
    assert order == ["a", "b"]


def test_rrf_weights():
    order, sc = rrf([["a"], ["b"]], k=60, weights=[1.0, 0.5])
    assert sc["b"] == pytest.approx(0.5 / 61)
    assert order == ["a", "b"]
    order, _ = rrf([["a"], ["b"]], k=60, weights=[0.5, 1.0])
    assert order == ["b", "a"]


def test_rrf_weights_length_mismatch():
    with pytest.raises(ValueError):
        rrf([["a"], ["b"]], weights=[1.0])


# ---------- BM25 分词 ----------

def test_tokenize_keeps_numbers_whole():
    toks = tokenize("Revenue was $130,497 million, up 114%; DRAM 2.5% 2025-01-26")
    assert "130497" in toks
    assert "114" in toks and "2.5" in toks
    assert "2025-01-26" in toks
    assert "the" not in toks


def test_tokenize_chinese_and_stopwords():
    toks = tokenize("英伟达的员工人数是多少")
    assert "的" not in toks and "多少" not in toks and "是" not in toks
    assert "员工" in toks and "人数" in toks
    # 注意：jieba 默认词典把「英伟达」切成「英 / 伟达」，这是 BM25 在公司名上偏弱的原因之一
    assert "伟达" in toks


def test_tokenize_fullwidth_normalized():
    assert tokenize("１２３") == ["123"]


# ---------- BM25 排序 ----------

def test_bm25_ranks_matching_doc_first():
    bm = BM25.from_texts(["d1", "d2", "d3"],
                         ["美光 员工 48,000 人", "英伟达 员工 36,000 人", "台积电 晶圆 代工"])
    res = bm.search("36,000 员工")
    assert res[0][0] == "d2"
    assert {r[0] for r in res} == {"d1", "d2"}       # d3 没有任何查询词
    assert bm.idf["36000"] > bm.idf["员工"]          # 稀有词 IDF 更高


def test_bm25_length_mismatch():
    with pytest.raises(ValueError):
        BM25(["a"], [["x"], ["y"]])


# ---------- 关键词加权计数基线 ----------

def _p(pid, section, body):
    return Passage(pid, "doc", "fixture", "zh", "标题", section, body)


def test_keyword_weighting():
    ps = [_p("p1", "员工", "公司共有员工\n100 人"),        # 前缀命中「员工」+10，正文 2 行各 +1，两词齐 +100
          _p("p2", "其他", "员工\n人数\n员工"),             # 正文 3 行命中 +3，两词齐 +100
          _p("p3", "其他", "无关内容")]
    res = dict(keyword_scores(["员工", "人"], ps))
    assert res["p1"] == 10 + 2 + 100
    assert res["p2"] == 3 + 100
    assert "p3" not in res


def test_keyword_one_line_counts_first_pattern_only():
    # 原工具的规则：一行只记第一个命中的关键词，所以同一行里的第二个词不算「出现过」
    ps = [_p("p", "其他", "员工 100 人")]
    assert keyword_scores(["员工", "人"], ps) == [("p", 1)]


def test_keyword_partial_match_no_bonus_and_stable_order():
    ps = [_p("a", "x", "苹果"), _p("b", "x", "苹果")]
    assert keyword_scores(["苹果", "香蕉"], ps) == [("a", 1), ("b", 1)]
    assert keyword_scores(["  "], ps) == []
