"""实体抽取测试:BIO HMM 单元行为 + /extract API 行为与一致性语义。"""
import pytest
from fastapi.testclient import TestClient

from app.index import IndexManager
from app.main import create_app
from app.ner import (
    LEGAL_NEXT,
    STATES,
    SpanError,
    check_spans,
    spans_from_tags,
    tags_from_spans,
    train_hmm,
    viterbi,
)


class FakeSpan:
    def __init__(self, start, end, label="PART"):
        self.start, self.end, self.label = start, end, label


def span(text, frag, label):
    """按原文片段定位区间,避免手数字符位置。"""
    s = text.index(frag)
    return {"start": s, "end": s + len(frag), "label": label}


TRAIN_TEXTS = [
    "3号车间数控机床出现传感器漂移,校准传感器后运行平稳。",
    "1号车间空压机出现轴承过热,更换轴承后恢复正常。",
    "4号车间注塑机出现温度报警,补充润滑油后故障消除。",
]
TRAIN_FRAGS = [
    ("传感器漂移", "校准传感器"),
    ("轴承过热", "更换轴承"),
    ("温度报警", "补充润滑油"),
]
TARGET_TEXTS = [
    # 目标记录的实体词均在训练样例中出现过(维修词表复用的真实场景)
    ("G1", "5号车间风机出现轴承过热,更换轴承后待复查。", ("轴承过热", "更换轴承")),
    ("G2", "2号车间机械臂出现温度报警,校准传感器后转入观察。", ("温度报警", "校准传感器")),
]


def train_docs():
    return [
        {"id": f"T{i}", "text": t, "tokens": []} for i, t in enumerate(TRAIN_TEXTS)
    ]


def train_payload():
    return [
        {
            "id": f"T{i}",
            "spans": [span(TRAIN_TEXTS[i], part, "PART"), span(TRAIN_TEXTS[i], act, "ACTION")],
        }
        for i, (part, act) in enumerate(TRAIN_FRAGS)
    ]


# ---- ner 单元测试 ----


def test_check_spans_accepts_adjacent_and_unsorted():
    check_spans([FakeSpan(5, 8), FakeSpan(0, 2), FakeSpan(2, 5), FakeSpan(8, 9)], 10)


@pytest.mark.parametrize(
    "spans",
    [
        [FakeSpan(0, 11)],                      # 越界
        [FakeSpan(9, 10), FakeSpan(8, 11)],     # 越界(后者)
        [FakeSpan(0, 5), FakeSpan(4, 6)],       # 部分重叠
        [FakeSpan(0, 8), FakeSpan(2, 3)],       # 包含
        [FakeSpan(2, 4), FakeSpan(2, 4)],       # 完全相同
        [FakeSpan(3, 3)],                       # start == end
        [FakeSpan(5, 4)],                       # start > end
        [FakeSpan(-1, 2)],                      # 负起点
    ],
)
def test_check_spans_rejects_invalid(spans):
    with pytest.raises(SpanError):
        check_spans(spans, 10)


def test_tags_and_spans_roundtrip():
    text = TRAIN_TEXTS[0]
    # 传感器漂移 PART,校准传感器 ACTION
    p0 = text.index("传感器漂移")
    p1 = text.index("校准传感器")
    spans = [FakeSpan(p0, p0 + 5, "PART"), FakeSpan(p1, p1 + 5, "ACTION")]
    tags = tags_from_spans(text, spans)
    assert tags[p0] == "B-PART" and tags[p0 + 4] == "I-PART"
    assert tags[p1] == "B-ACTION" and tags[p1 + 4] == "I-ACTION"
    assert tags[0] == "O" and tags[-1] == "O"
    back = spans_from_tags(tags)
    assert back == [(p0, p0 + 5, "PART"), (p1, p1 + 5, "ACTION")]


def trained_model():
    tagged = []
    for text, (part, act) in zip(TRAIN_TEXTS, TRAIN_FRAGS):
        spans = [FakeSpan(text.index(part), text.index(part) + len(part), "PART"),
                 FakeSpan(text.index(act), text.index(act) + len(act), "ACTION")]
        tagged.append((text, tags_from_spans(text, spans)))
    return train_hmm(tagged)


def test_hmm_decodes_expected_entities():
    model = trained_model()
    for _, text, (part, act) in TARGET_TEXTS:
        spans = spans_from_tags(viterbi(model, text))
        got = {(text[s:e], lab) for s, e, lab in spans}
        assert (part, "PART") in got and (act, "ACTION") in got


def test_viterbi_output_always_legal():
    model = trained_model()
    texts = [t for _, t, _ in TARGET_TEXTS] + TRAIN_TEXTS + ["9号车间锅炉出现油路堵塞,更换皮带后待复查。"]
    for text in texts:
        tags = viterbi(model, text)
        assert len(tags) == len(text)
        assert tags[0] != "I-PART" and tags[0] != "I-ACTION"  # 首字符不能是 I
        for prev, nxt in zip(tags, tags[1:]):
            assert nxt in LEGAL_NEXT[prev]  # 只沿合法转移


def test_viterbi_handles_unknown_and_empty():
    model = trained_model()
    tags = viterbi(model, "ABC车间🚀出现xyz,做点什么后待定。")  # 未登录字不报错
    assert len(tags) == len("ABC车间🚀出现xyz,做点什么后待定。")
    assert viterbi(model, "") == []
    assert spans_from_tags(viterbi(model, "")) == []


def test_hmm_deterministic():
    m1, m2 = trained_model(), trained_model()
    text = TARGET_TEXTS[0][1]
    assert viterbi(m1, text) == viterbi(m2, text)


# ---- /extract API 测试 ----


@pytest.fixture()
def client():
    c = TestClient(create_app(IndexManager()))
    docs = train_docs() + [{"id": tid, "text": t, "tokens": []} for tid, t, _ in TARGET_TEXTS]
    docs.append({"id": "EMPTY", "text": "", "tokens": []})
    r = c.post("/documents/batch", json={"upserts": docs, "deletes": []})
    assert r.status_code == 200
    return c


def extract(client, train, targets):
    return client.post("/extract", json={"train": train, "targets": targets})


def test_extract_happy_path(client):
    r = extract(client, train_payload(), ["G1", "G2"])
    assert r.status_code == 200
    results = {x["id"]: x for x in r.json()["results"]}
    assert set(results) == {"G1", "G2"}
    for tid, text, (part, act) in TARGET_TEXTS:
        ents = results[tid]["entities"]
        assert results[tid]["text"] == text
        got = {(e["text"], e["label"]) for e in ents}
        assert (part, "PART") in got and (act, "ACTION") in got
        for e in ents:  # 区间与原文片段一致
            assert e["text"] == text[e["start"]:e["end"]]
            assert 0 <= e["start"] < e["end"] <= len(text)


def test_extract_empty_target_returns_no_entities(client):
    r = extract(client, train_payload(), ["EMPTY"])
    assert r.status_code == 200
    assert r.json()["results"] == [{"id": "EMPTY", "text": "", "entities": []}]


def test_extract_duplicate_targets_deduplicated(client):
    r = extract(client, train_payload(), ["G1", "G1", "G2"])
    assert r.status_code == 200
    assert [x["id"] for x in r.json()["results"]] == ["G1", "G2"]


def test_extract_missing_ids_rejected(client):
    r = extract(client, train_payload() + [{"id": "NOPE", "spans": [{"start": 0, "end": 2, "label": "PART"}]}], ["G1"])
    assert r.status_code == 404 and "NOPE" in r.json()["detail"]
    r = extract(client, train_payload(), ["G1", "GHOST"])
    assert r.status_code == 404 and "GHOST" in r.json()["detail"]


def test_extract_span_out_of_bounds_rejected(client):
    bad = train_payload()
    bad[0]["spans"][0]["end"] = len(TRAIN_TEXTS[0]) + 1
    r = extract(client, bad, ["G1"])
    assert r.status_code == 422 and "越界" in r.json()["detail"]


def test_extract_overlapping_spans_rejected(client):
    bad = train_payload()
    bad[0]["spans"].append({"start": bad[0]["spans"][0]["start"] + 1, "end": bad[0]["spans"][0]["end"] + 2, "label": "ACTION"})
    r = extract(client, bad, ["G1"])
    assert r.status_code == 422 and "重叠" in r.json()["detail"]


def test_extract_request_shape_validated(client):
    ok = train_payload()
    assert extract(client, ok[:1], ["G1"]).status_code == 422          # 训练样例 < 2
    too_many = (ok * 4)[:9]
    for i, t in enumerate(too_many):                                   # 编号改不同,避开重复校验
        t["id"] = f"T{i % 3}" if i < 3 else f"T{i}"
    assert extract(client, too_many, ["G1"]).status_code == 422        # 训练样例 > 8
    assert extract(client, ok, []).status_code == 422                  # 目标 < 1
    assert extract(client, ok, ["G1"] * 6).status_code == 422          # 目标 > 5
    bad_span = train_payload()
    bad_span[0]["spans"][0]["end"] = bad_span[0]["spans"][0]["start"]  # start == end
    assert extract(client, bad_span, ["G1"]).status_code == 422
    bad_label = train_payload()
    bad_label[0]["spans"][0]["label"] = "FOO"
    assert extract(client, bad_label, ["G1"]).status_code == 422       # 非法实体类型


def test_extract_duplicate_train_ids_rejected(client):
    dup = train_payload()
    dup[1]["id"] = dup[0]["id"]
    assert extract(client, dup, ["G1"]).status_code == 422


def test_extract_uses_text_version_at_acceptance(client):
    # 成功更正后,新请求按新正文计算:旧区间越界,新区间生效
    new_text = "空压机更换轴承。"
    r = client.post("/documents/batch", json={"upserts": [{"id": "T0", "text": new_text, "tokens": []}]})
    assert r.status_code == 200
    stale = train_payload()  # 按旧正文标注的区间
    r = extract(client, stale, ["G1"])
    assert r.status_code == 422 and "越界" in r.json()["detail"]
    fresh = train_payload()
    fresh[0] = {"id": "T0", "spans": [span(new_text, "更换轴承", "ACTION")]}
    r = extract(client, fresh, ["T0"])
    assert r.status_code == 200
    ents = r.json()["results"][0]["entities"]
    assert any(e["text"] == "更换轴承" and e["label"] == "ACTION" for e in ents)


def test_failed_correction_keeps_state_for_extract(client, monkeypatch):
    before = extract(client, train_payload(), ["G1"])
    assert before.status_code == 200
    monkeypatch.setattr("app.index.MAX_DOCS", 3)  # 触发容量失败
    r = client.post("/documents/batch", json={"upserts": [{"id": "X", "text": "t", "tokens": []}]})
    assert r.status_code == 409
    after = extract(client, train_payload(), ["G1"])
    assert after.status_code == 200
    assert after.json() == before.json()  # 失败更正保留原状态,抽取结果不变


def test_extract_does_not_disturb_search(client):
    q = {"terms": ["轴承过热"], "k": 5}
    client.post("/documents/batch", json={"upserts": [
        {"id": "S1", "text": "轴承过热记录", "tokens": ["轴承过热"]},
    ]})
    before = client.post("/search", json=q).json()
    extract(client, train_payload(), ["G1", "G2"])
    after = client.post("/search", json=q).json()
    assert before == after  # 现有检索仍可用,结果不受抽取影响


def test_docs_served_and_lists_extract(client):
    assert client.get("/docs").status_code == 200
    spec = client.get("/openapi.json").json()
    assert "/extract" in spec["paths"]
