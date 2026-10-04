import pytest

from rre.metrics import (paired_bootstrap_ci, percentile, query_metrics, summarize,
                         switch_decision)


def test_query_metrics_basic():
    m = query_metrics(["a", "x", "b", "y", "z", "c"], ["a", "b", "c"])
    assert m["recall"] == pytest.approx(2 / 3)       # c 在第 6 名，不算进前 5
    assert m["hit"] == 1.0
    assert m["rr"] == 1.0
    assert m["first_gold_rank"] == 1
    assert m["ceiling"] == 1.0


def test_query_metrics_miss_and_mrr_cutoff():
    ranked = [f"n{i}" for i in range(10)] + ["g"]
    m = query_metrics(ranked, ["g"])
    assert m["recall"] == 0.0 and m["hit"] == 0.0
    assert m["rr"] == 0.0                              # 第 11 名，超出 MRR@10
    assert m["first_gold_rank"] == 11


def test_query_metrics_rr_second():
    assert query_metrics(["x", "g"], ["g"])["rr"] == 0.5


def test_ceiling_when_gold_more_than_five():
    gold = [f"g{i}" for i in range(8)]
    m = query_metrics(gold, gold)
    assert m["recall"] == pytest.approx(5 / 8)
    assert m["ceiling"] == pytest.approx(5 / 8)


def test_empty_gold_raises():
    with pytest.raises(ValueError):
        query_metrics(["a"], [])


def test_summarize():
    rows = [query_metrics(["a"], ["a"]), query_metrics(["b"], ["a"])]
    s = summarize(rows)
    assert s == {"n": 2, "recall@5": 0.5, "hit@5": 0.5, "mrr@10": 0.5, "recall@5_ceiling": 1.0}


def test_percentile_nearest_rank():
    xs = list(range(1, 101))
    assert percentile(xs, 0.5) == 51          # round(0.5 * 99) = 50 -> 第 51 个
    assert percentile(xs, 0.95) == 95
    assert percentile([7], 0.95) == 7


def test_bootstrap_constant_and_deterministic():
    lo, hi = paired_bootstrap_ci([0.2] * 30)
    assert lo == pytest.approx(0.2) and hi == pytest.approx(0.2)
    d = [1, 0, 0, 1, -1, 0, 1, 0, 0, 0, 1, 0]
    assert paired_bootstrap_ci(d, seed=0) == paired_bootstrap_ci(d, seed=0)
    lo, hi = paired_bootstrap_ci(d, n_boot=2000)
    mean = sum(d) / len(d)
    assert lo <= mean <= hi
    assert lo < hi


def test_bootstrap_detects_clear_effect():
    lo, _ = paired_bootstrap_ci([1.0] * 40 + [0.0] * 10, n_boot=2000)
    assert lo > 0


def test_switch_decision_all_pass():
    d = switch_decision(0.05, 0.01, -0.02, 500)
    assert d["switch"] is True and all(d["checks"].values())


@pytest.mark.parametrize("args,failed", [
    ((0.02, 0.01, 0.0, 500), "delta>=min"),
    ((0.05, 0.0, 0.0, 500), "ci_low>0"),
    ((0.05, 0.01, -0.04, 500), "no_type_drop"),
    ((0.05, 0.01, 0.0, 1000), "p95<max"),
])
def test_switch_decision_each_rule(args, failed):
    d = switch_decision(*args)
    assert d["switch"] is False
    assert [k for k, v in d["checks"].items() if not v] == [failed]


def test_switch_boundaries_inclusive():
    # 恰好 +0.03、恰好掉 0.03 都算满足（规则写的是 ≥ 与「不超过」）
    assert switch_decision(0.03, 0.001, -0.03, 999.9)["switch"] is True
