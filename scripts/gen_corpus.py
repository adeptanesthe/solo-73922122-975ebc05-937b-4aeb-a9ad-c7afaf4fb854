"""生成本地示例语料 data/corpus.json(确定性种子)。

模拟设备维修记录:编号、正文、分词序列。词频呈长尾分布,
少量常见词(报警/停机/巡检等)便于演示块上界剪枝的跳过效果。
"""
from __future__ import annotations

import json
import random
from pathlib import Path

MACHINES = ["数控机床", "注塑机", "空压机", "输送带", "机械臂", "锅炉", "离心泵", "液压站", "风机", "起重机", "切割机", "冷却塔"]
FAULTS = ["轴承过热", "液压泄漏", "电机异响", "传感器漂移", "皮带打滑", "振动超标", "温度报警", "压力不足", "油路堵塞", "控制失灵", "刀具磨损", "冷却不足"]
ACTIONS = ["更换轴承", "紧固螺栓", "清洗油路", "校准传感器", "更换密封圈", "补充润滑油", "调整间隙", "更换皮带", "复位控制器", "更换刀具"]
RESULTS = ["恢复正常", "运行平稳", "故障消除", "待复查", "转入观察"]
COMMON = ["报警", "停机", "巡检", "工单", "备件", "夜班", "白班", "加急", "复查", "保养", "检修", "记录"]
RARE = ["联轴器", "变频器", "减速箱", "安全阀", "流量计", "编码器", "接触器", "继电器"]

N_DOCS = 1500
EMPTY_RATIO = 0.04  # 空文档比例(计入 N 与平均长度)


def gen(n_docs: int = N_DOCS, seed: int = 20261117) -> dict:
    rng = random.Random(seed)
    docs = {}
    for i in range(n_docs):
        did = f"R{i:04d}"
        if rng.random() < EMPTY_RATIO:
            docs[did] = {"text": "", "tokens": []}
            continue
        machine = rng.choice(MACHINES)
        fault = rng.choice(FAULTS)
        action = rng.choice(ACTIONS)
        result = rng.choice(RESULTS)
        loc = f"{rng.randint(1, 6)}号车间"
        text = f"{loc}{machine}出现{fault},{action}后{result}。"
        tokens = [loc, machine, fault, action, result]
        # 常见词:每篇以较高概率追加若干
        for w in COMMON:
            if rng.random() < 0.18:
                tokens.append(w)
        # 罕见词:低概率
        for w in RARE:
            if rng.random() < 0.03:
                tokens.append(w)
        # 少量重复词,制造 tf>1
        if rng.random() < 0.25:
            tokens.append(rng.choice([machine, fault, action]))
        docs[did] = {"text": text, "tokens": tokens}
    return docs


def main() -> None:
    out = Path(__file__).resolve().parents[1] / "data" / "corpus.json"
    docs = gen()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(docs, ensure_ascii=False, indent=1), encoding="utf-8")
    n_empty = sum(1 for d in docs.values() if not d["tokens"])
    print(f"已生成 {len(docs)} 篇(含空文档 {n_empty} 篇)-> {out}")


if __name__ == "__main__":
    main()
