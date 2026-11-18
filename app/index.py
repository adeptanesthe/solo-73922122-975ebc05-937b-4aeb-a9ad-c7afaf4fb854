"""BM25 精确检索核心。

设计要点:
- 语料、全库统计(N、avgdl、idf)与剪枝依据(块最大贡献)在一次构建中一起生成,
  组成不可变的 State;更正 = 构建全新 State 后原子切换,一次查询只持有并使用同一个 State。
- 正式检索使用 BlockMax WAND: 每个词项的 posting 按块(32 篇)预计算最大贡献,
  后缀最大值作为"当前位置之后所有 posting 贡献"的上界, 用于枢轴选择与整块跳过,
  绝不全量计分后排序。前 k 与穷举完全一致, 同分按编号升序。
- 穷举实现仅用于对照验证(测试与 scripts/verify.py)。
"""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Iterable

import numpy as np

K1 = 1.2
B = 0.75
BLOCK_SIZE = 32
MAX_DOCS = 5000


def bm25_idf(n: int, df: int) -> float:
    """IDF = ln(1 + (N - df + 0.5) / (df + 0.5)),恒为正。"""
    return math.log(1.0 + (n - df + 0.5) / (df + 0.5))


@dataclass(frozen=True)
class TermData:
    """单个词项的倒排与块上界。"""

    docids: np.ndarray  # int32,升序的内部文档号
    contribs: np.ndarray  # float64,每个 posting 的完整 BM25 贡献(含 idf 与长度归一)
    block_last: np.ndarray  # int32,每块最后一个文档号(用于整块跳过)
    suffix_max: np.ndarray  # float64,块最大贡献的后缀最大值:块 i 起(含)所有 posting 贡献的上界


@dataclass(frozen=True)
class State:
    """一次完整生效的不可变检索状态:语料 + 全库统计 + 剪枝依据。

    构建完成后任何字段都不再修改;查询全程持有同一引用即保证一致性。
    """

    docs: dict  # 编号 -> (正文, 分词序列 tuple);按约定不再修改
    ids: tuple  # 内部文档号 -> 编号(编号升序)
    texts: tuple  # 内部文档号 -> 正文
    n: int  # 总篇数 N(空文档计入)
    avgdl: float  # 平均文档长度(空文档按 0 计入)
    terms: dict  # 词项 -> TermData


def build_state(docs: dict) -> State:
    """由 {编号: (正文, 分词序列)} 构建完整状态。空文档计入 N 与平均长度。"""
    ids = tuple(sorted(docs.keys()))
    n = len(ids)
    texts = tuple(docs[i][0] for i in ids)
    token_lists = [docs[i][1] for i in ids]
    dls = np.array([len(t) for t in token_lists], dtype=np.float64)
    avgdl = float(dls.mean()) if n else 0.0

    postings: dict = {}
    for idx, tokens in enumerate(token_lists):
        if not tokens:
            continue
        counts: dict = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1
        for tok, tf in counts.items():
            postings.setdefault(tok, []).append((idx, tf))

    terms: dict = {}
    for tok, lst in postings.items():
        df = len(lst)
        idf = bm25_idf(n, df)
        docids = np.array([d for d, _ in lst], dtype=np.int32)
        tfs = np.array([c for _, c in lst], dtype=np.float64)
        # 分母 = tf + k1 * (1 - b + b * dl / avgdl);avgdl == 0 时全库皆空文档,无 posting,不会到此
        denom = tfs + K1 * (1.0 - B + B * dls[docids] / avgdl)
        contribs = idf * (tfs * (K1 + 1.0)) / denom
        n_blocks = (docids.shape[0] + BLOCK_SIZE - 1) // BLOCK_SIZE
        block_max = np.empty(n_blocks, dtype=np.float64)
        block_last = np.empty(n_blocks, dtype=np.int32)
        for bidx in range(n_blocks):
            lo = bidx * BLOCK_SIZE
            hi = min(lo + BLOCK_SIZE, docids.shape[0])
            block_max[bidx] = contribs[lo:hi].max()
            block_last[bidx] = docids[hi - 1]
        suffix_max = np.maximum.accumulate(block_max[::-1])[::-1].copy()
        terms[tok] = TermData(
            docids=docids, contribs=contribs, block_last=block_last, suffix_max=suffix_max
        )

    return State(docs=docs, ids=ids, texts=texts, n=n, avgdl=avgdl, terms=terms)


class _Cursor:
    """WAND 中单个词项的 posting 游标。"""

    __slots__ = ("td", "pos")

    def __init__(self, td: TermData):
        self.td = td
        self.pos = 0

    @property
    def exhausted(self) -> bool:
        return self.pos >= self.td.docids.shape[0]

    @property
    def doc(self) -> int:
        return int(self.td.docids[self.pos])

    @property
    def ub(self) -> float:
        """当前位置起(含)所有 posting 贡献的上界:块最大贡献的后缀最大。"""
        return float(self.td.suffix_max[self.pos // BLOCK_SIZE])

    def advance(self, target: int) -> None:
        """移动到第一个 docid >= target 的 posting;按块尾文档号整块跳过。"""
        td = self.td
        n = td.docids.shape[0]
        b = self.pos // BLOCK_SIZE
        n_blocks = td.block_last.shape[0]
        while b < n_blocks and int(td.block_last[b]) < target:
            b += 1
        if b >= n_blocks:
            self.pos = n
            return
        lo = max(self.pos, b * BLOCK_SIZE)
        hi = min((b + 1) * BLOCK_SIZE, n)
        self.pos = lo + int(np.searchsorted(td.docids[lo:hi], target))


def _insert_result(results: list, item: tuple, k: int) -> None:
    """把 (score, id, docidx) 按 (-score, id) 升序插入并截断到 k;k <= 20,线性即可。"""
    key = (-item[0], item[1])
    if len(results) == k and key >= (-results[-1][0], results[-1][1]):
        return  # 进不了前 k
    if len(results) == k:
        results.pop()
    i = len(results)
    while i > 0 and (-results[i - 1][0], results[i - 1][1]) > key:
        i -= 1
    results.insert(i, item)


def search(state: State, terms: Iterable[str], k: int):
    """BlockMax WAND 正式检索。

    返回 (hits, scored):hits 为 [(编号, 正文, 分数)],按分数降序、同分编号升序;
    scored 为完整计分篇数(通过枢轴检查、被完整累加分数的文档数)。
    只完整计分可能进入前 k 的候选,其余候选被块上界跳过。
    """
    uniq = list(dict.fromkeys(terms))  # 查询词项去重(保序)
    cursors = [_Cursor(state.terms[t]) for t in uniq if t in state.terms]
    if state.n == 0 or not cursors or k <= 0:
        return [], 0

    results: list = []  # (score, id, docidx),按 (-score, id) 升序,长度 <= k
    scored = 0
    threshold = -math.inf  # 当前第 k 名的分数;堆未满时为 -inf
    live = list(cursors)

    while True:
        live = [c for c in live if not c.exhausted]
        if not live:
            break
        live.sort(key=lambda c: c.doc)
        # 枢轴选择:前缀上界和首次 >= 阈值的位置。取 >= 是为了同分时按编号升序仍精确:
        # 上界和 == 阈值的文档可能凭更小编号进入前 k,不能剪。
        total = 0.0
        pivot = -1
        for i, c in enumerate(live):
            total += c.ub
            if total >= threshold:
                pivot = i
                break
        if pivot < 0:
            break  # 所有剩余文档上界和都低于阈值,结束
        pivot_doc = live[pivot].doc
        if live[0].doc == pivot_doc:
            d = pivot_doc
            s = 0.0
            for c in cursors:  # 固定词项顺序累加,与穷举逐位一致
                if not c.exhausted and c.doc == d:
                    s += float(c.td.contribs[c.pos])
                    c.pos += 1
            scored += 1
            _insert_result(results, (s, state.ids[d], d), k)
            if len(results) == k:
                threshold = results[-1][0]
        else:
            # 块上界剪枝:把最小游标按块整块跳到 pivot_doc
            live[0].advance(pivot_doc)

    hits = [(did, state.texts[d], s) for s, did, d in results]
    return hits, scored


def exhaustive_search(state: State, terms: Iterable[str], k: int):
    """对照实现:全量计分后排序。仅用于验证正式检索,不作为正式路径。

    与 search 使用同一 State、同一词项顺序累加,分数逐位一致。
    返回 (hits, scored),scored 恒为全库篇数 N。
    """
    uniq = list(dict.fromkeys(terms))
    sums: dict = {}
    for t in uniq:
        td = state.terms.get(t)
        if td is None:
            continue
        for d, c in zip(td.docids.tolist(), td.contribs.tolist()):
            sums[d] = sums.get(d, 0.0) + c
    order = sorted(sums.items(), key=lambda kv: (-kv[1], state.ids[kv[0]]))
    hits = [(state.ids[d], state.texts[d], s) for d, s in order[:k] if s > 0.0]
    return hits, state.n


class CapacityError(Exception):
    """批量更正后总篇数将超过上限。"""


class IndexManager:
    """持有当前 State;写者串行构建并原子切换,读者无锁读取同一完整状态。

    - 查询不取锁,慢建索引不挡查询;构建期间读到的是旧状态,切换后新查询用新状态。
    - 写锁串行化更正:并发成功更正逐个生效、不丢失;构建失败抛异常,旧状态保留。
    """

    def __init__(self, build_delay: float = 0.0):
        self._state = build_state({})
        self._lock = threading.Lock()
        self._build_delay = build_delay  # 模拟慢建索引(演示/测试用),秒

    def current(self) -> State:
        return self._state

    def apply_batch(self, upserts: list, deletes: list):
        """应用一批增改删。upserts: [(编号, 正文, 分词序列)];deletes: [编号]。

        同一批内先删后增改;增改中重复编号后者覆盖前者。
        成功返回 (新 State, 实际删除篇数);失败抛异常,当前状态不变。
        """
        with self._lock:
            docs = dict(self._state.docs)
            deleted = 0
            for did in deletes:
                if docs.pop(did, None) is not None:
                    deleted += 1
            for did, text, tokens in upserts:
                docs[did] = (text, tuple(tokens))
            if len(docs) > MAX_DOCS:
                raise CapacityError(f"更正后总篇数 {len(docs)} 超过上限 {MAX_DOCS},本批未生效")
            new_state = self._build(docs)
            self._state = new_state  # 原子切换:之后的新查询使用新状态
            return new_state, deleted

    def _build(self, docs: dict) -> State:
        if self._build_delay > 0:
            time.sleep(self._build_delay)  # 慢建索引:占写锁但不挡读者
        return build_state(docs)
