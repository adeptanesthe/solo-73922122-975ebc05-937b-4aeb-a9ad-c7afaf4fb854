"""索引层测试:计分公式、空文档统计、同分排序、WAND 与穷举一致、确有剪枝。"""
import math
import random

import pytest

from app.index import (
    B,
    K1,
    MAX_DOCS,
    CapacityError,
    IndexManager,
    bm25_idf,
    build_state,
    exhaustive_search,
    search,
)


def ref_scores(docs, terms, k):
    """独立参照:直接按公式对原始语料计分排序,不经过索引结构。"""
    uniq = list(dict.fromkeys(terms))
    n = len(docs)
    ids = sorted(docs)
    dls = {i: len(docs[i][1]) for i in ids}
    avgdl = sum(dls.values()) / n if n else 0.0
    df = {t: sum(1 for i in ids if t in docs[i][1]) for t in uniq}
    out = []
    for i in ids:
        s = 0.0
        for t in uniq:
            tf = docs[i][1].count(t)
            if tf == 0:
                continue
            idf = math.log(1.0 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * (tf * (K1 + 1.0)) / (tf + K1 * (1.0 - B + B * dls[i] / avgdl))
        if s > 0.0:
            out.append((i, s))
    out.sort(key=lambda x: (-x[1], x[0]))
    return out[:k]


def wand_pairs(hits):
    return [(i, s) for i, _, s in hits]


def test_idf_formula():
    assert bm25_idf(3, 2) == pytest.approx(math.log(1.0 + 1.5 / 2.5))
    assert bm25_idf(10, 1) > bm25_idf(10, 5) > 0.0


def test_empty_index_returns_empty():
    st = build_state({})
    hits, scored = search(st, ["a", "b"], 10)
    assert hits == [] and scored == 0
    hits, scored = exhaustive_search(st, ["a"], 10)
    assert hits == [] and scored == 0


def test_exact_scores_handcomputed():
    docs = {
        "D1": ("正文1", ("a", "b", "a")),
        "D2": ("正文2", ("a",)),
        "D3": ("空文档", ()),
    }
    st = build_state(docs)
    n, avgdl = 3, 4.0 / 3.0
    idf_a = math.log(1.0 + (n - 2 + 0.5) / 2.5)
    idf_b = math.log(1.0 + (n - 1 + 0.5) / 1.5)

    def contrib(idf, tf, dl):
        return idf * (tf * (K1 + 1.0)) / (tf + K1 * (1.0 - B + B * dl / avgdl))

    exp = {
        "D1": contrib(idf_a, 2, 3) + contrib(idf_b, 1, 3),
        "D2": contrib(idf_a, 1, 1),
    }
    hits, scored = search(st, ["a", "b"], 5)
    assert [h[0] for h in hits] == ["D1", "D2"]
    for hid, _, s in hits:
        assert s == pytest.approx(exp[hid], rel=0, abs=1e-15)
    assert all(h[0] != "D3" for h in hits)  # 空文档计入统计但不是命中


def test_empty_docs_count_in_n_and_avgdl():
    with_empty = build_state({"A": ("t", ("x",)), "B": ("t", ("x", "y")), "E": ("空", ())})
    without_empty = build_state({"A": ("t", ("x",)), "B": ("t", ("x", "y"))})
    assert with_empty.n == 3 and with_empty.avgdl == pytest.approx(3.0 / 3.0)
    assert without_empty.n == 2 and without_empty.avgdl == pytest.approx(3.0 / 2.0)
    s_with = dict((h[0], h[2]) for h in search(with_empty, ["x"], 5)[0])
    s_without = dict((h[0], h[2]) for h in search(without_empty, ["x"], 5)[0])
    assert s_with["A"] != pytest.approx(s_without["A"])  # N 与 avgdl 改变分数
    assert all(h[0] != "E" for h in search(with_empty, ["x"], 5)[0])


def test_tie_break_by_id_ascending():
    docs = {f"D{i}": ("相同正文", ("a", "b")) for i in range(10)}
    st = build_state(docs)
    hits, _ = search(st, ["a", "b"], 5)
    assert [h[0] for h in hits] == [f"D{i}" for i in range(5)]
    scores = {h[2] for h in hits}
    assert len(scores) == 1  # 完全同分


def test_query_terms_deduplicated():
    st = build_state({"A": ("t", ("x", "y")), "B": ("t", ("x",))})
    h1, s1 = search(st, ["x", "x", "y", "x"], 5)
    h2, s2 = search(st, ["x", "y"], 5)
    assert wand_pairs(h1) == wand_pairs(h2)


def test_wand_matches_exhaustive_and_reference_random():
    rng = random.Random(7)
    vocab = [f"w{i}" for i in range(40)]
    for size in (0, 1, 2, 33, 300):
        docs = {}
        for j in range(size):
            dl = rng.choice([0, 0, 1, 5, 12, 30])
            docs[f"D{j:04d}"] = (f"正文{j}", tuple(rng.choice(vocab) for _ in range(dl)))
        st = build_state(docs)
        for _ in range(120):
            terms = [rng.choice(vocab) for _ in range(rng.randint(1, 5))]
            k = rng.randint(1, 20)
            w_hits, w_scored = search(st, terms, k)
            e_hits, e_scored = exhaustive_search(st, terms, k)
            assert wand_pairs(w_hits) == wand_pairs(e_hits)
            assert wand_pairs(w_hits) == ref_scores(docs, terms, k)
            assert w_scored <= e_scored == size


def test_pruning_actually_skips_candidates():
    # 1000 篇含 common,其中前 60 篇另含 rare;k 小 => common-only 文档应被整块跳过
    docs = {}
    for j in range(1000):
        toks = ["common", "filler"] + (["rare"] if j < 60 else [])
        docs[f"D{j:04d}"] = (f"正文{j}", tuple(toks))
    st = build_state(docs)
    hits, scored = search(st, ["common", "rare"], 10)
    e_hits, e_scored = exhaustive_search(st, ["common", "rare"], 10)
    assert wand_pairs(hits) == wand_pairs(e_hits)
    assert len(hits) == 10
    assert scored < 200  # 远小于含词候选 1000 篇:确有跳过
    assert e_scored == 1000


def test_k_zero_or_negative_returns_empty():
    st = build_state({"A": ("t", ("x",))})
    assert search(st, ["x"], 0) == ([], 0)


def test_manager_batch_capacity_and_failure_keeps_state():
    mgr = IndexManager()
    mgr.apply_batch([("A", "正文", ("x",))], [])
    big = [(f"D{i}", "t", ("x",)) for i in range(MAX_DOCS)]
    with pytest.raises(CapacityError):
        mgr.apply_batch(big, [])
    st = mgr.current()
    assert st.n == 1 and "A" in st.docs  # 失败保留旧状态


def test_manager_delete_and_upsert_same_batch():
    mgr = IndexManager()
    mgr.apply_batch([("A", "旧", ("x",)), ("B", "b", ("y",))], [])
    st, deleted = mgr.apply_batch([("A", "新", ("z",))], ["B", "不存在"])
    assert deleted == 1
    assert st.n == 1 and st.docs["A"][0] == "新"
    assert search(st, ["y"], 5)[0] == []
    assert search(st, ["z"], 5)[0][0][0] == "A"


def test_manager_concurrent_updates_not_lost():
    import threading

    mgr = IndexManager()
    errors = []

    def worker(base):
        try:
            for j in range(25):
                mgr.apply_batch([(f"T{base}-{j}", "t", ("w",))], [])
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert mgr.current().n == 8 * 25  # 并发成功更正一篇不丢


def test_slow_build_does_not_block_queries_and_swaps_atomically():
    import threading
    import time

    mgr = IndexManager(build_delay=0.8)
    mgr.apply_batch([("OLD", "旧", ("oldterm",))], [])

    done = []

    def updater():
        mgr.apply_batch([("NEW", "新", ("newterm",))], [])
        done.append(True)

    t = threading.Thread(target=updater)
    t0 = time.monotonic()
    t.start()
    time.sleep(0.1)  # 构建进行中
    hits_old, _ = search(mgr.current(), ["newterm"], 5)
    dt = time.monotonic() - t0
    assert hits_old == []  # 构建期间仍是旧状态
    assert dt < 0.6  # 查询未被慢建索引挡住
    t.join()
    assert done
    hits_new, _ = search(mgr.current(), ["newterm"], 5)
    assert [h[0] for h in hits_new] == ["NEW"]  # 成功后新查询使用新状态
