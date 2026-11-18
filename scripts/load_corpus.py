"""把本地语料 data/corpus.json 批量写入运行中的服务。

用法: python3 scripts/load_corpus.py [--url http://localhost:8000] [--batch 500]
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--batch", type=int, default=500)
    ap.add_argument(
        "--corpus",
        default=str(Path(__file__).resolve().parents[1] / "data" / "corpus.json"),
    )
    args = ap.parse_args()

    docs = json.loads(Path(args.corpus).read_text(encoding="utf-8"))
    items = [
        {"id": did, "text": d["text"], "tokens": d["tokens"]} for did, d in docs.items()
    ]
    total = 0
    for i in range(0, len(items), args.batch):
        chunk = items[i : i + args.batch]
        payload = json.dumps({"upserts": chunk, "deletes": []}).encode("utf-8")
        req = urllib.request.Request(
            f"{args.url}/documents/batch", data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            body = json.loads(resp.read())
        total = body["total"]
        print(f"已写入 {i + len(chunk)}/{len(items)},库内总数 {total}")
    print(f"完成,库内共 {total} 篇")
    return 0


if __name__ == "__main__":
    sys.exit(main())
