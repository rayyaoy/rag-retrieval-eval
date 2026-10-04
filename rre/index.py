"""向量索引：编码（带磁盘缓存）→ Qdrant（本地文件 / 内存 / 远程服务，由配置决定）。

向量缓存：{work_dir}/emb/{key}.npy，key = sha256(模型名, 窗口, 是否带前缀, 全部文本)。
文本或模型一变 key 就变，不会用错缓存。Qdrant 侧在 {work_dir}/qdrant_state.json 记下
每个 collection 对应的 key，key 不变就复用，变了就重建。
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from .passages import Passage


def texts_for(passages: Sequence[Passage], with_prefix: bool = True) -> list[str]:
    return [p.text if with_prefix else p.body for p in passages]


def cache_key(model_name: str, max_seq: int | str, texts: Sequence[str]) -> str:
    h = hashlib.sha256()
    h.update(f"{model_name}|{max_seq}|{len(texts)}".encode())
    for t in texts:
        h.update(b"\x00")
        h.update(t.encode())
    return h.hexdigest()[:24]


def encode_cached(embedder, texts: Sequence[str], work_dir: Path, max_seq: int | str = "",
                  progress: bool = True) -> tuple[np.ndarray, str, float]:
    """返回 (向量, key, 本次编码耗时秒；命中缓存为 0)。"""
    key = cache_key(embedder.name, max_seq, texts)
    path = Path(work_dir) / "emb" / f"{key}.npy"
    if path.exists():
        return np.load(path), key, 0.0
    t = time.time()
    vecs = embedder.encode(texts, progress=progress)
    secs = time.time() - t
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vecs)
    (path.with_suffix(".json")).write_text(json.dumps(
        {"model": embedder.name, "n": len(texts), "dim": int(vecs.shape[1]),
         "encode_seconds": round(secs, 1), "dtype": getattr(embedder, "dtype", "?"),
         "device": getattr(embedder, "device", "?"), "built_at": time.strftime("%Y-%m-%d %H:%M:%S")},
        ensure_ascii=False, indent=1))
    return vecs, key, secs


def open_qdrant(location: str, work_dir: Path):
    from qdrant_client import QdrantClient
    if location == ":memory:":
        return QdrantClient(":memory:")
    if location.startswith(("http://", "https://")):
        return QdrantClient(url=location)
    if location == "local":
        location = str(Path(work_dir) / "qdrant")
    return QdrantClient(path=location)


def ensure_collection(client, name: str, vecs: np.ndarray, pids: Sequence[str], key: str,
                      work_dir: Path) -> bool:
    """collection 不存在或 key 变了就（重）建。返回是否重建。"""
    from qdrant_client import models
    state_path = Path(work_dir) / "qdrant_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if client.collection_exists(name) and state.get(name) == key \
            and client.count(name).count == len(pids):
        return False
    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(name, vectors_config=models.VectorParams(
        size=int(vecs.shape[1]), distance=models.Distance.COSINE))
    B = 256
    for i in range(0, len(pids), B):
        client.upsert(name, [models.PointStruct(id=j, vector=vecs[j].tolist(), payload={"pid": pids[j]})
                             for j in range(i, min(i + B, len(pids)))])
    state[name] = key
    state_path.write_text(json.dumps(state, indent=1))
    return True


def vector_search(client, name: str, qvec: np.ndarray, limit: int) -> list[tuple[str, float]]:
    res = client.query_points(name, query=qvec.astype("float32").tolist(), limit=limit,
                              with_payload=["pid"]).points
    return [(p.payload["pid"], float(p.score)) for p in res]
