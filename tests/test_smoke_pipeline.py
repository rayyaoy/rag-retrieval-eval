"""不下载任何模型的端到端冒烟测试：小样例语料 + 假向量（字符哈希）+ Qdrant 内存模式 + 假精排。

只验证管线能跑通、输出结构正确、几个确定性的检索行为成立；假向量没有语义，不代表任何检索质量。
"""
from pathlib import Path

import pytest

from rre.config import load_config
from rre.corpus import load_passages
from rre.evaluate import TYPES, run_system, score, stale_metrics
from rre.gold import build_goldset, load_jsonl
from rre.index import encode_cached, ensure_collection, open_qdrant, texts_for
from rre.models import HashEmbedder, OverlapReranker
from rre.search import Retriever

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    work = tmp_path_factory.mktemp("work")
    passages = load_passages(FIX / "mini_passages.jsonl")
    gold, problems = build_goldset(load_jsonl(FIX / "mini_questions.jsonl"), passages)
    assert problems == {}
    emb = HashEmbedder(dim=128)
    vecs, key, _ = encode_cached(emb, texts_for(passages), work, progress=False)
    client = open_qdrant(":memory:", work)
    assert ensure_collection(client, "mini", vecs, [p.pid for p in passages], key, work) is True
    r = Retriever(passages, emb, client, "mini", OverlapReranker(),
                  {"vector_candidates": 10, "topn_each": 5, "rrf_k": 60, "rerank_top": 5})
    return r, gold, work, passages


def test_gold_resolved(setup):
    _, gold, _, _ = setup
    by = {g["id"]: g for g in gold}
    assert by["F-03"]["gold"] == ["edgar-demo-fy2025#0003"]
    assert by["F-05"]["stale"] == ["edgar-demo-fy2024#0000"]


def test_all_systems_run_and_score(setup):
    r, gold, _, passages = setup
    systems = {
        "keyword": lambda it: r.keyword(it["keywords"]),
        "bm25": lambda it: r.bm25_search(it["query"]),
        "vector": lambda it: r.vector(it["query"]),
        "rrf": lambda it: r.rrf(it["query"]),
        "hybrid": lambda it: r.hybrid(it["query"]),
        "V1": lambda it: r.vec_rerank(it["query"]),
        "V2": lambda it: r.wrrf(it["query"]),
        "V3": lambda it: r.route(it["query"]),
    }
    ids = {p.pid for p in passages}
    for name, fn in systems.items():
        ranked, lat = run_system(fn, gold, warmup=1)
        assert len(lat) == len(gold)
        for rk in ranked.values():
            assert set(rk) <= ids and len(rk) == len(set(rk)), name     # 不重复、不越界
        per_q, tables = score(ranked, gold)
        assert tables["ALL"]["n"] == len(gold)
        assert set(tables) >= {"ALL"} | {g["type"] for g in gold}
        st = stale_metrics(ranked, gold)
        assert st["n"] == 1


def test_deterministic_behaviours(setup):
    r, _, _, _ = setup
    # BM25：数字整体成词，精确数字直接命中
    assert r.bm25_search("1,234 名员工")[0] == "demo-zh#0004"
    assert r.bm25_search("38.7 billion")[0] == "edgar-demo-fy2025#0003"
    # 关键词基线：中文关键词打不中英文段落
    assert all(not p.startswith("edgar") for p in r.keyword(["代工厂"]))
    # 路由：带数字走 hybrid，不带走 vector
    assert r.route("没有数字") == r.vector("没有数字")
    assert r.route("38.7 billion") == r.hybrid("38.7 billion")


def test_vector_cache_and_collection_reuse(setup):
    r, _, work, passages = setup
    emb = HashEmbedder(dim=128)
    _, key, secs = encode_cached(emb, texts_for(passages), work, progress=False)
    assert secs == 0.0                                       # 第二次命中缓存
    assert ensure_collection(r.client, "mini", r.client and __import__("numpy").zeros((len(passages), 128)),
                             [p.pid for p in passages], key, work) is False
    _, key2, _ = encode_cached(emb, texts_for(passages, with_prefix=False), work, progress=False)
    assert key2 != key                                       # 去掉前缀 → 不同缓存


def test_config_env_override(tmp_path):
    cfg = load_config(env={"RRE_QDRANT": ":memory:", "RRE_EMBEDDER": "some/model",
                           "RRE_WORK_DIR": str(tmp_path / "w")})
    assert cfg["qdrant"]["location"] == ":memory:"
    assert cfg["models"]["embedder"] == "some/model"
    assert cfg.work_dir == tmp_path / "w" and cfg.work_dir.exists()
    assert set(TYPES) == {"crosslingual", "terminology", "number_source", "stale"}
