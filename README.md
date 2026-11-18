# 设备维修记录 BM25 精确检索

面向设备维护工程师的故障处理办法检索服务:FastAPI + NumPy 实现精确 BM25,
正式检索使用 BlockMax WAND 块上界剪枝(不全量计分后排序),前 k 与穷举完全一致。

## 快速启动

```bash
docker compose up --build
```

- 交互式 API 文档: <http://localhost:8000/docs>
- 健康检查: <http://localhost:8000/healthz>

载入本地示例语料(1500 篇维修记录,含空文档):

```bash
python3 scripts/load_corpus.py            # 默认 http://localhost:8000
```

本地开发(不用 Docker):

```bash
pip install -r requirements-dev.txt
uvicorn app.main:app --port 8000
```

## 检索模型

- 固定 `k1 = 1.2`、`b = 0.75`;`IDF = ln(1 + (N − df + 0.5) / (df + 0.5))`(恒为正)。
- 文档得分 = Σ(去重查询词项) `idf · tf·(k1+1) / (tf + k1·(1 − b + b·dl/avgdl))`。
- 空文档(分词序列为空)计入 N 与平均长度 avgdl,不产生任何 posting,永远不会成为命中。
- 空库检索返回空结果(`{"hits": [], "scored": 0}`)。
- 同分按编号升序;查询词项服务端去重;`k` 限 1~20。

## 接口

### `POST /documents/batch` — 批量增改删

```json
{
  "upserts": [{"id": "R0001", "text": "数控机床轴承过热,更换轴承后恢复正常。", "tokens": ["数控机床", "轴承过热", "更换轴承"]}],
  "deletes": ["R0002"]
}
```

- 同一批内先删后增改;增改中重复编号后者覆盖前者。
- 语料、全库统计(N、avgdl、idf)与剪枝依据(块最大贡献)在一批中一起构建、原子生效。
- 更正后总篇数超过 5000 返回 `409`,旧状态完整保留;参数非法返回 `422`,旧状态不变。
- 写者串行:并发成功更正逐个生效、不丢失;读者不加锁:慢建索引期间查询仍用旧状态即时返回,
  构建成功后的新查询自动使用新状态(可用环境变量 `BM25_BUILD_DELAY=秒` 演示)。

### `POST /search` — 正式检索(块上界剪枝)

```json
{"terms": ["轴承过热", "更换轴承"], "k": 10}
```

响应:

```json
{
  "hits": [{"id": "R0001", "text": "...", "score": 3.42}],
  "scored": 88
}
```

- `hits`:命中记录的编号、正文、分数,按分数降序、同分编号升序,至多 k 篇。
- `scored`:完整计分篇数(通过枢轴检查、被完整累加分数的文档数)。
  块上界剪枝下 `scored` 通常远小于含词候选数与全库篇数——被跳过的候选未经完整计分。

### `POST /search/exhaustive` — 穷举对照(仅验证用)

全量计分后排序的对照接口,响应结构相同,`scored` 恒为全库篇数 N。
用于核对正式检索的前 k 一致性,不是正式检索路径。

### 其他

- `GET /stats`:篇数、平均长度、词项数、BM25 参数。
- `GET /healthz`:健康检查。

## 剪枝正确性说明

每个词项的 posting 按 32 篇分块,预计算块内最大贡献;块最大贡献的后缀最大值即
"当前位置之后所有 posting 贡献"的上界。WAND 每轮按当前文档号排序游标,以前缀上界和
选择枢轴:上界和严格小于当前第 k 名分数的文档不可能进入前 k,按块整块跳过;
只有与枢轴文档对齐的候选才被完整计分。枢轴判断取 `>=` 阈值,同分文档仍可凭编号升序
进入前 k,保证与穷举逐位一致(同一 State、同一词项顺序累加,浮点结果完全相同)。

## 验证

```bash
python3 -m pytest tests/ -q     # 23 项单元/API/并发测试
python3 scripts/verify.py       # 穷举对照 + 随机回归 + 剪枝示例
```

`scripts/verify.py` 用本地语料 `data/corpus.json` 与随机语料,将 BlockMax WAND 与
穷举实现、独立参照实现(直接按公式对原始语料计分)逐位比对,并打印完整计分篇数,
展示确有候选被跳过,例如:

```
查询 ['报警', '停机'], k=10:完整计分 88 篇;含查询词的文档 449 篇;全库 1500 篇
即 361 篇含词候选未经完整计分即被块上界跳过,前 10 名与穷举完全一致。
```

## 目录

```
app/index.py      BM25 状态构建、BlockMax WAND、穷举对照、原子状态管理
app/main.py       FastAPI 应用与端点
app/schemas.py    请求/响应模型
data/corpus.json  本地示例语料(scripts/gen_corpus.py 生成,确定性种子)
scripts/verify.py 穷举对照与剪枝效果验证
scripts/load_corpus.py 向运行中的服务批量写入语料
tests/            单元、API 与并发一致性测试
```
