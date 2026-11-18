"""API 层测试:端点行为、参数校验、空库、批量增改删、并发一致性。"""
import threading

import pytest
from fastapi.testclient import TestClient

from app.index import MAX_DOCS, IndexManager
from app.main import create_app


@pytest.fixture()
def client():
    return TestClient(create_app(IndexManager()))


def upsert(client, docs, deletes=None):
    return client.post(
        "/documents/batch",
        json={"upserts": docs, "deletes": deletes or []},
    )


def test_empty_index_search_returns_empty(client):
    r = client.post("/search", json={"terms": ["a"], "k": 10})
    assert r.status_code == 200
    assert r.json() == {"hits": [], "scored": 0}


def test_upsert_search_hit_fields(client):
    r = upsert(client, [
        {"id": "D1", "text": "数控机床轴承过热,更换轴承后恢复正常。", "tokens": ["数控机床", "轴承过热", "更换轴承"]},
        {"id": "D2", "text": "空压机电机异响。", "tokens": ["空压机", "电机异响"]},
    ])
    assert r.status_code == 200 and r.json()["total"] == 2
    r = client.post("/search", json={"terms": ["轴承过热"], "k": 5})
    body = r.json()
    assert [h["id"] for h in body["hits"]] == ["D1"]
    hit = body["hits"][0]
    assert set(hit.keys()) == {"id", "text", "score"}
    assert hit["text"] == "数控机床轴承过热,更换轴承后恢复正常。"
    assert hit["score"] > 0
    assert body["scored"] == 1


def test_k_bounds_validated(client):
    assert client.post("/search", json={"terms": ["a"], "k": 0}).status_code == 422
    assert client.post("/search", json={"terms": ["a"], "k": 21}).status_code == 422
    assert client.post("/search", json={"terms": ["a"], "k": 1}).status_code == 200
    assert client.post("/search", json={"terms": ["a"], "k": 20}).status_code == 200


def test_invalid_payload_rejected(client):
    assert client.post("/documents/batch", json={"upserts": [{"id": "", "tokens": []}]}).status_code == 422
    assert client.post("/documents/batch", json={"upserts": [{"id": "A", "tokens": ["x", 1]}]}).status_code == 422
    assert client.post("/search", json={"terms": "notalist", "k": 5}).status_code == 422


def test_update_and_delete_take_effect_atomically(client):
    upsert(client, [{"id": "A", "text": "旧正文", "tokens": ["旧词"]}])
    upsert(client, [{"id": "A", "text": "新正文", "tokens": ["新词"]}], deletes=[])
    r = client.post("/search", json={"terms": ["旧词"], "k": 5}).json()
    assert r["hits"] == []
    r = client.post("/search", json={"terms": ["新词"], "k": 5}).json()
    assert r["hits"][0]["text"] == "新正文"
    upsert(client, [], deletes=["A"])
    assert client.post("/search", json={"terms": ["新词"], "k": 5}).json()["hits"] == []
    assert client.get("/stats").json()["docs"] == 0


def test_capacity_limit_returns_409_and_keeps_state(client):
    upsert(client, [{"id": "KEEP", "text": "保留", "tokens": ["保留词"]}])
    big = [{"id": f"D{i}", "text": "t", "tokens": ["x"]} for i in range(MAX_DOCS)]
    r = upsert(client, big)
    assert r.status_code == 409
    body = client.post("/search", json={"terms": ["保留词"], "k": 5}).json()
    assert [h["id"] for h in body["hits"]] == ["KEEP"]  # 旧状态保留
    assert client.get("/stats").json()["docs"] == 1


def test_exactly_max_docs_accepted(client):
    docs = [{"id": f"D{i:04d}", "text": "t", "tokens": ["x"]} for i in range(MAX_DOCS)]
    r = upsert(client, docs)
    assert r.status_code == 200 and r.json()["total"] == MAX_DOCS


def test_concurrent_batches_not_lost(client):
    errors = []

    def worker(base):
        try:
            for j in range(10):
                r = upsert(client, [{"id": f"W{base}-{j}", "text": "t", "tokens": ["w"]}])
                assert r.status_code == 200
        except Exception as exc:  # pragma: no cover
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert client.get("/stats").json()["docs"] == 60


def test_exhaustive_endpoint_matches_formal(client):
    # 15 篇含 rare 的文档填满前 10,common-only 文档上界低于阈值 => 被块上界跳过
    upsert(client, [
        {"id": f"D{i:02d}", "text": f"正文{i}", "tokens": ["common"] + (["rare", "extra"] if i < 15 else [])}
        for i in range(50)
    ])
    req = {"terms": ["common", "rare"], "k": 10}
    formal = client.post("/search", json=req).json()
    exh = client.post("/search/exhaustive", json=req).json()
    assert [(h["id"], h["score"]) for h in formal["hits"]] == [
        (h["id"], h["score"]) for h in exh["hits"]
    ]
    assert formal["scored"] < exh["scored"] == 50  # 正式检索确有跳过


def test_stats_and_healthz(client):
    assert client.get("/healthz").json()["status"] == "ok"
    upsert(client, [
        {"id": "A", "text": "t", "tokens": ["x", "y"]},
        {"id": "E", "text": "", "tokens": []},
    ])
    s = client.get("/stats").json()
    assert s["docs"] == 2 and s["avg_len"] == 1.0 and s["terms"] == 2
    assert s["k1"] == 1.2 and s["b"] == 0.75
