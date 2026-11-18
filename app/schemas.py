"""请求/响应模型。"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class DocUpsert(BaseModel):
    """单篇记录的增改:编号、正文、分词序列。"""

    id: str = Field(min_length=1, max_length=128, description="记录编号")
    text: str = Field(default="", description="记录正文")
    tokens: list[str] = Field(default_factory=list, description="分词序列(可为空,空文档计入统计)")


class BatchRequest(BaseModel):
    """批量增改删:同一批内先删后增改。"""

    upserts: list[DocUpsert] = Field(default_factory=list, description="新增/修改的记录")
    deletes: list[str] = Field(default_factory=list, description="删除的编号")


class BatchResponse(BaseModel):
    status: str = "ok"
    total: int = Field(description="生效后的总篇数")
    upserted: int = Field(description="本批提交的增改条数")
    deleted: int = Field(description="本批实际删除的篇数")


class SearchRequest(BaseModel):
    terms: list[str] = Field(default_factory=list, description="查询词项(服务端再去重)")
    k: int = Field(ge=1, le=20, description="返回篇数,1~20")


class Hit(BaseModel):
    id: str
    text: str
    score: float


class SearchResponse(BaseModel):
    hits: list[Hit]
    scored: int = Field(description="完整计分篇数")


# ---- 实体抽取(/extract)----

EntityLabel = Literal["PART", "ACTION"]  # PART=故障部件,ACTION=处理动作


class SpanIn(BaseModel):
    """一条人工标注区间:半开区间 [start, end),按 Unicode 字符(码位)位置。"""

    start: int = Field(ge=0, description="起始字符位置(含)")
    end: int = Field(ge=1, description="结束字符位置(不含)")
    label: EntityLabel = Field(description="实体类型:PART=故障部件,ACTION=处理动作")

    @model_validator(mode="after")
    def _check_order(self):
        if self.end <= self.start:
            raise ValueError("end 必须大于 start")
        return self


class TrainExample(BaseModel):
    """一条训练样例:现存记录编号 + 正文中的人工标注区间(不得重叠)。"""

    id: str = Field(min_length=1, max_length=128, description="现存记录编号")
    spans: list[SpanIn] = Field(min_length=1, description="人工标注区间,至少 1 条,不得重叠")


class ExtractRequest(BaseModel):
    """实体抽取请求:2~8 条人工标注样例训练,1~5 条记录待抽取。"""

    train: list[TrainExample] = Field(min_length=2, max_length=8, description="人工标注样例,2~8 条")
    targets: list[str] = Field(min_length=1, max_length=5, description="待抽取记录编号,1~5 条;重复编号按首次出现去重")


class EntityOut(BaseModel):
    start: int = Field(description="起始字符位置(含)")
    end: int = Field(description="结束字符位置(不含)")
    label: EntityLabel
    text: str = Field(description="原文片段")


class ExtractResult(BaseModel):
    id: str = Field(description="记录编号")
    text: str = Field(description="受理时该记录的正文")
    entities: list[EntityOut] = Field(description="预测实体,按出现顺序")


class ExtractResponse(BaseModel):
    results: list[ExtractResult]
