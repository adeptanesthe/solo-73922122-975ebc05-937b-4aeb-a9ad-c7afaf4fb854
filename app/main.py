"""FastAPI 入口:设备维修记录 BM25 精确检索 + 字符级 BIO HMM 实体抽取。

- POST /documents/batch  批量增改删(语料、全库统计、剪枝依据一起生效)
- POST /search           正式检索(BlockMax WAND 块上界剪枝)
- POST /search/exhaustive 对照接口(全量计分后排序,仅用于验证一致性)
- POST /extract          用人工标注样例训练加平滑 BIO HMM,对指定记录做实体抽取
- GET  /stats            当前状态统计
- GET  /healthz          健康检查
- GET  /docs             Swagger UI
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from .index import B, K1, MAX_DOCS, CapacityError, IndexManager, exhaustive_search, search
from .ner import SpanError, check_spans, spans_from_tags, tags_from_spans, train_hmm, viterbi
from .schemas import (
    BatchRequest,
    BatchResponse,
    EntityOut,
    ExtractRequest,
    ExtractResponse,
    ExtractResult,
    Hit,
    SearchRequest,
    SearchResponse,
)


def create_app(manager: IndexManager | None = None) -> FastAPI:
    if manager is None:
        delay = float(os.environ.get("BM25_BUILD_DELAY", "0") or 0)
        manager = IndexManager(build_delay=delay)

    app = FastAPI(
        title="设备维修记录 BM25 检索",
        version="1.1.0",
        description=(
            "精确 BM25(k1=1.2, b=0.75, IDF=ln(1+(N-df+0.5)/(df+0.5)))。"
            "正式检索使用 BlockMax WAND 块上界剪枝,前 k 与穷举一致,同分按编号升序。"
            "实体抽取:提交 2~8 条人工标注样例训练加一平滑的字符级 BIO HMM,"
            "经合法转移的 Viterbi 解码,对 1~5 条记录预测故障部件/处理动作区间。"
        ),
    )
    app.state.manager = manager

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/docs")

    @app.get("/healthz", tags=["meta"])
    def healthz():
        return {"status": "ok"}

    @app.get("/stats", tags=["meta"])
    def stats():
        s = manager.current()
        return {
            "docs": s.n,
            "avg_len": s.avgdl,
            "terms": len(s.terms),
            "k1": K1,
            "b": B,
            "max_docs": MAX_DOCS,
        }

    @app.post("/documents/batch", response_model=BatchResponse, tags=["corpus"])
    def batch_update(req: BatchRequest):
        """批量增改删。语料、全库统计与剪枝依据一起生效;失败保留旧状态。"""
        try:
            new_state, deleted = manager.apply_batch(
                [(d.id, d.text, d.tokens) for d in req.upserts],
                list(req.deletes),
            )
        except CapacityError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return BatchResponse(total=new_state.n, upserted=len(req.upserts), deleted=deleted)

    @app.post("/search", response_model=SearchResponse, tags=["search"])
    def search_ep(req: SearchRequest):
        """正式检索:块上界剪枝,返回命中编号、正文、分数及完整计分篇数。"""
        state = manager.current()  # 一次查询只用同一完整状态
        hits, scored = search(state, req.terms, req.k)
        return SearchResponse(
            hits=[Hit(id=i, text=t, score=sc) for i, t, sc in hits], scored=scored
        )

    @app.post("/search/exhaustive", response_model=SearchResponse, tags=["debug"])
    def search_exhaustive_ep(req: SearchRequest):
        """对照接口:全量计分后排序,仅用于验证正式检索的一致性,非正式路径。"""
        state = manager.current()
        hits, scored = exhaustive_search(state, req.terms, req.k)
        return SearchResponse(
            hits=[Hit(id=i, text=t, score=sc) for i, t, sc in hits], scored=scored
        )

    @app.post("/extract", response_model=ExtractResponse, tags=["extract"])
    def extract_ep(req: ExtractRequest):
        """实体抽取:用人工标注样例训练加一平滑的字符级 BIO HMM,Viterbi 解码预测区间。

        训练与预测只使用受理时的同一版正文(同一 State 快照);更正成功后的新请求
        自动使用新正文。编号缺失(404)、区间非法/越界/重叠(422)时整次拒绝。
        """
        state = manager.current()  # 受理时快照:本次训练与预测只用这一版正文

        train_ids = [e.id for e in req.train]
        if len(set(train_ids)) != len(train_ids):
            raise HTTPException(status_code=422, detail="训练样例中记录编号重复")
        missing = [i for i in dict.fromkeys(train_ids + list(req.targets)) if i not in state.docs]
        if missing:
            raise HTTPException(status_code=404, detail=f"记录编号不存在: {missing}")

        tagged = []
        for ex in req.train:
            text = state.docs[ex.id][0]
            try:
                check_spans(ex.spans, len(text))
            except SpanError as exc:
                raise HTTPException(status_code=422, detail=f"记录 {ex.id}: {exc}") from exc
            tagged.append((text, tags_from_spans(text, ex.spans)))

        model = train_hmm(tagged)
        results = []
        for tid in dict.fromkeys(req.targets):  # 重复编号按首次出现去重
            text = state.docs[tid][0]
            entities = [
                EntityOut(start=s, end=e, label=lab, text=text[s:e])
                for s, e, lab in spans_from_tags(viterbi(model, text))
            ]
            results.append(ExtractResult(id=tid, text=text, entities=entities))
        return ExtractResponse(results=results)

    return app


app = create_app()
