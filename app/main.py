"""FastAPI 入口:设备维修记录 BM25 精确检索。

- POST /documents/batch  批量增改删(语料、全库统计、剪枝依据一起生效)
- POST /search           正式检索(BlockMax WAND 块上界剪枝)
- POST /search/exhaustive 对照接口(全量计分后排序,仅用于验证一致性)
- GET  /stats            当前状态统计
- GET  /healthz          健康检查
- GET  /docs             Swagger UI
"""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from .index import B, K1, MAX_DOCS, CapacityError, IndexManager, exhaustive_search, search
from .schemas import BatchRequest, BatchResponse, Hit, SearchRequest, SearchResponse


def create_app(manager: IndexManager | None = None) -> FastAPI:
    if manager is None:
        delay = float(os.environ.get("BM25_BUILD_DELAY", "0") or 0)
        manager = IndexManager(build_delay=delay)

    app = FastAPI(
        title="设备维修记录 BM25 检索",
        version="1.0.0",
        description=(
            "精确 BM25(k1=1.2, b=0.75, IDF=ln(1+(N-df+0.5)/(df+0.5)))。"
            "正式检索使用 BlockMax WAND 块上界剪枝,前 k 与穷举一致,同分按编号升序。"
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

    return app


app = create_app()
