"""穷举对照与剪枝效果验证。

1. 用本地语料 data/corpus.json 构建索引;
2. 手工查询 + 随机查询:BlockMax WAND 与穷举(全量计分后排序)逐位比对前 k;
3. 独立参照实现(不经过索引结构,直接按公式对原始语料计分)交叉验证;
4. 打印完整计分篇数,展示确有候选被块上界跳过(WAND scored << 穷举 N)。

用法: python3 scripts/verify.py
退出码非 0 表示不一致。
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.index import B, K1, build_state, exhaustive_search, search  # noqa: E402


def reference_hits(docs: dict, terms: list, k: int):
    """独立参照:直接从原始语料按 BM25 公式计分,不经过任何索引结构。"""
    uniq = list(dict.fromkeys(terms))
    n = len(docs)
    if n == 0 or not uniq:
        return []
    ids = sorted(docs)
    dls = {i: len(docs[i][1]) for i in ids}
    avgdl = sum(dls.values()) / n
    df = {t: sum(1 for i in ids if t in docs[i][1]) for t in uniq}
    out = []
    for i in ids:
        s = 0.0
        toks = docs[i][1]
        for t in uniq:
            tf = toks.count(t)
            if tf == 0:
                continue
            idf = math.log(1.0 + (n - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * (tf * (K1 + 1.0)) / (tf + K1 * (1.0 - B + B * dls[i] / avgdl))
        if s > 0.0:
            out.append((i, s))
    out.sort(key=lambda x: (-x[1], x[0]))
    return out[:k]


def check_equal(tag: str, a, b) -> bool:
    """a、b 为 [(id, text, score)] 或 [(id, score)],要求编号、顺序、分数完全一致。"""
    ka = [(x[0], x[-1]) for x in a]
    kb = [(x[0], x[-1]) for x in b]
    if len(ka) != len(kb) or any(x != y for x, y in zip(ka, kb)):
        print(f"[FAIL] {tag}")
        print("  wand/exhaustive:", ka[:8])
        print("  reference      :", kb[:8])
        return False
    return True


def main() -> int:
    docs_raw = json.loads((Path(__file__).resolve().parents[1] / "data" / "corpus.json").read_text(encoding="utf-8"))
    docs = {i: (d["text"], tuple(d["tokens"])) for i, d in docs_raw.items()}
    state = build_state(docs)
    n = state.n
    print(f"语料: {n} 篇, 平均长度 {state.avgdl:.3f}, 词项 {len(state.terms)} 个")

    failures = 0

    # ---- 1. 手工查询:WAND vs 穷举 vs 独立参照,并展示跳过 ----
    print("\n== 手工查询:前 k 一致性 + 剪枝效果 ==")
    crafted = [
        (["轴承过热"], 5),
        (["轴承过热", "更换轴承"], 10),
        (["报警", "停机"], 10),
        (["数控机床", "振动超标", "加急"], 20),
        (["联轴器", "安全阀"], 8),
        (["不存在的词"], 5),
        (["报警", "报警", "停机"], 10),  # 重复词项应去重
        ([], 5),
    ]
    print(f"{'查询':<28} {'k':>3} {'命中':>4} {'WAND计分':>9} {'穷举计分':>9} {'跳过':>6}")
    for terms, k in crafted:
        w_hits, w_scored = search(state, terms, k)
        e_hits, e_scored = exhaustive_search(state, terms, k)
        r_hits = reference_hits(docs, terms, k)
        ok1 = check_equal(f"wand vs exhaustive {terms}", w_hits, e_hits)
        ok2 = check_equal(f"wand vs reference {terms}", w_hits, r_hits)
        failures += 0 if (ok1 and ok2) else 1
        skipped = e_scored - w_scored
        print(f"{str(terms):<28} {k:>3} {len(w_hits):>4} {w_scored:>9} {e_scored:>9} {skipped:>6}")

    # ---- 2. 随机查询回归:WAND vs 穷举 vs 独立参照 ----
    print("\n== 随机查询回归(300 组,种子固定)==")
    rng = random.Random(42)
    vocab = list(state.terms.keys())
    total_wand_scored = 0
    for trial in range(300):
        m = rng.randint(1, 5)
        terms = [rng.choice(vocab) for _ in range(m)]
        if rng.random() < 0.1:
            terms.append("不存在的词")
        k = rng.randint(1, 20)
        w_hits, w_scored = search(state, terms, k)
        e_hits, _ = exhaustive_search(state, terms, k)
        r_hits = reference_hits(docs, terms, k)
        total_wand_scored += w_scored
        if not (check_equal(f"rand{trial} wand~exh {terms}", w_hits, e_hits)
                and check_equal(f"rand{trial} wand~ref {terms}", w_hits, r_hits)):
            failures += 1
    print(f"300 组随机查询全部比对完成,不一致 {failures} 组;WAND 平均计分 {total_wand_scored/300:.1f} 篇/次(全库 {n} 篇)")

    # ---- 3. 随机语料压力测试(含空库/单篇/空文档)----
    print("\n== 随机语料压力测试 ==")
    for size in (0, 1, 2, 37, 500):
        rng2 = random.Random(1000 + size)
        vocab2 = [f"t{i}" for i in range(25)]
        corpus = {}
        for j in range(size):
            dl = rng2.choice([0, 0, 1, 3, 8, 20])
            corpus[f"D{j:03d}"] = (f"正文{j}", tuple(rng2.choice(vocab2) for _ in range(dl)))
        st = build_state(corpus)
        for trial in range(150):
            terms = [rng2.choice(vocab2) for _ in range(rng2.randint(1, 4))]
            k = rng2.randint(1, 20)
            w_hits, _ = search(st, terms, k)
            e_hits, _ = exhaustive_search(st, terms, k)
            r_hits = reference_hits(corpus, terms, k)
            if not (check_equal(f"size{size} t{trial} wand~exh", w_hits, e_hits)
                    and check_equal(f"size{size} t{trial} wand~ref", w_hits, r_hits)):
                failures += 1
        print(f"语料规模 {size:>4}:150 组查询比对完成")

    # ---- 4. 确有跳过的示例详解 ----
    print("\n== 剪枝示例详解 ==")
    terms, k = ["报警", "停机"], 10
    w_hits, w_scored = search(state, terms, k)
    contain = sum(
        1 for i in range(n)
        if any(t in state.terms and i in set(state.terms[t].docids.tolist()) for t in dict.fromkeys(terms))
    )
    print(f"查询 {terms}, k={k}:完整计分 {w_scored} 篇;含查询词的文档 {contain} 篇;全库 {n} 篇")
    print(f"即 {contain - w_scored} 篇含词候选未经完整计分即被块上界跳过,前 {k} 名与穷举完全一致。")

    print("\n结果:", "全部通过" if failures == 0 else f"失败 {failures} 处")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
