"""字符级 BIO 隐马尔可夫模型实体抽取。

- 两类实体:PART(故障部件)、ACTION(处理动作);字符为观测,BIO 标注。
- 训练:由人工标注区间生成字符级 BIO 序列,统计初始/转移/发射计数,全部加一平滑;
  未登录字(训练语料中未出现的字符)在解码时映射为 <UNK>,其概率即平滑后的 1/(N+V)。
- 解码:仅沿合法转移的 Viterbi(I-X 只能接在 B-X/I-X 之后,首字符不能是 I-X),
  保证输出的区间结构合法。状态顺序固定、严格大于才更新,结果确定。

纯函数实现,不持有语料状态;每次请求由调用方传入受理时同一版正文训练并预测。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

LABELS = ("PART", "ACTION")
STATES = ("O", "B-PART", "I-PART", "B-ACTION", "I-ACTION")
START_STATES = ("O", "B-PART", "B-ACTION")  # 首字符的合法状态(不能是 I-X)
UNK = "<UNK>"

# 每个状态的合法后继(合法转移):I-X 只能接在 B-X/I-X 之后
LEGAL_NEXT = {
    "O": ("O", "B-PART", "B-ACTION"),
    "B-PART": ("I-PART", "O", "B-PART", "B-ACTION"),
    "I-PART": ("I-PART", "O", "B-PART", "B-ACTION"),
    "B-ACTION": ("I-ACTION", "O", "B-PART", "B-ACTION"),
    "I-ACTION": ("I-ACTION", "O", "B-PART", "B-ACTION"),
}
# 每个状态的合法前驱(上表的反向索引)
LEGAL_PREV = {s: tuple(p for p in STATES if s in LEGAL_NEXT[p]) for s in STATES}


class SpanError(ValueError):
    """标注区间非法:越界、次序颠倒或重叠。"""


def check_spans(spans, text_len: int) -> None:
    """校验一组标注区间:0 <= start < end <= 正文长度,且两两不重叠(相邻允许)。

    spans: 具有 start/end 属性的对象序列。非法时抛 SpanError。
    """
    prev_end = 0
    for sp in sorted(spans, key=lambda s: (s.start, s.end)):
        if sp.start < 0 or sp.end <= sp.start:
            raise SpanError(f"区间 [{sp.start}, {sp.end}) 非法:需 0 <= start < end")
        if sp.end > text_len:
            raise SpanError(f"区间 [{sp.start}, {sp.end}) 越界:正文长度 {text_len}")
        if sp.start < prev_end:
            raise SpanError(f"区间 [{sp.start}, {sp.end}) 与前一区间重叠")
        prev_end = sp.end


def tags_from_spans(text: str, spans) -> list:
    """把不重叠的标注区间转换为字符级 BIO 标签序列(先调用 check_spans)。"""
    tags = ["O"] * len(text)
    for sp in spans:
        tags[sp.start] = f"B-{sp.label}"
        for i in range(sp.start + 1, sp.end):
            tags[i] = f"I-{sp.label}"
    return tags


@dataclass(frozen=True)
class BioHMM:
    """加一平滑后的字符级 BIO HMM(对数概率)。"""

    log_pi: dict  # 起始状态 -> log 概率(仅 START_STATES)
    log_trans: dict  # 源状态 -> {合法后继: log 概率}
    log_emit: dict  # 状态 -> {字符或 <UNK>: log 概率}
    vocab: frozenset  # 训练语料出现过的字符;其余字符解码时按 <UNK> 计


def train_hmm(tagged_texts) -> BioHMM:
    """由 [(正文, BIO 标签序列)] 训练 HMM。初始、转移、发射全部加一平滑。

    转移平滑只覆盖合法后继;发射平滑的词表为训练字符集加 <UNK>。
    """
    pi_counts = {s: 1 for s in START_STATES}  # 加一
    trans_counts = {s: {t: 1 for t in LEGAL_NEXT[s]} for s in STATES}  # 加一
    emit_counts = {s: {} for s in STATES}
    vocab = set()
    for text, tags in tagged_texts:
        if len(text) != len(tags):
            raise ValueError("正文长度与标签序列长度不一致")
        if not tags:
            continue
        pi_counts[tags[0]] += 1
        for ch, tag in zip(text, tags):
            counts = emit_counts[tag]
            counts[ch] = counts.get(ch, 0) + 1
            vocab.add(ch)
        for prev, nxt in zip(tags, tags[1:]):
            if nxt not in LEGAL_NEXT[prev]:
                raise ValueError(f"非法转移 {prev} -> {nxt}")
            trans_counts[prev][nxt] += 1

    total_pi = sum(pi_counts.values())
    log_pi = {s: math.log(c / total_pi) for s, c in pi_counts.items()}
    log_trans = {}
    for s in STATES:
        total = sum(trans_counts[s].values())
        log_trans[s] = {t: math.log(c / total) for t, c in trans_counts[s].items()}
    n_vocab = len(vocab) + 1  # + <UNK>
    log_emit = {}
    for s in STATES:
        counts = emit_counts[s]
        total = sum(counts.values()) + n_vocab  # 加一平滑分母
        log_emit[s] = {ch: math.log((counts.get(ch, 0) + 1) / total) for ch in vocab}
        log_emit[s][UNK] = math.log(1.0 / total)
    return BioHMM(
        log_pi=log_pi,
        log_trans=log_trans,
        log_emit=log_emit,
        vocab=frozenset(vocab),
    )


def viterbi(model: BioHMM, text: str) -> list:
    """仅沿合法转移的 Viterbi 解码,返回字符级 BIO 标签序列。

    未登录字按 <UNK> 计;空正文返回空序列。平局按固定状态顺序取先者,结果确定。
    """
    n = len(text)
    if n == 0:
        return []
    obs = [ch if ch in model.vocab else UNK for ch in text]

    neg = -math.inf
    prev_score = {s: neg for s in STATES}
    for s in START_STATES:
        prev_score[s] = model.log_pi[s] + model.log_emit[s][obs[0]]
    back = []  # 每个位置:状态 -> 最优前驱状态

    for i in range(1, n):
        cur_score = {}
        ptr = {}
        emit = model.log_emit
        for s in STATES:
            best, best_prev = neg, None
            for p in LEGAL_PREV[s]:
                cand = prev_score[p] + model.log_trans[p][s]
                if cand > best:
                    best, best_prev = cand, p
            cur_score[s] = best + emit[s][obs[i]]
            ptr[s] = best_prev
        back.append(ptr)
        prev_score = cur_score

    last = STATES[0]  # 严格大于才更新:平局取状态顺序先者,与内部循环一致
    for s in STATES[1:]:
        if prev_score[s] > prev_score[last]:
            last = s
    tags = [last]
    for ptr in reversed(back):
        last = ptr[last]
        tags.append(last)
    tags.reverse()
    return tags


def spans_from_tags(tags) -> list:
    """把 BIO 标签序列还原为 [(start, end, label)],半开区间,按出现顺序。"""
    spans = []
    i, n = 0, len(tags)
    while i < n:
        tag = tags[i]
        if tag.startswith("B-"):
            label = tag[2:]
            j = i + 1
            while j < n and tags[j] == f"I-{label}":
                j += 1
            spans.append((i, j, label))
            i = j
        else:
            i += 1
    return spans
