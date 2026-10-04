"""配置加载：config.toml + 环境变量覆盖。

不依赖任何本机路径。相对路径一律以仓库根目录（本文件上两级）为基准。
"""
from __future__ import annotations

import copy
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# 环境变量 -> (节, 键)
ENV_OVERRIDES = {
    "RRE_DATA_DIR": ("paths", "data_dir"),
    "RRE_WORK_DIR": ("paths", "work_dir"),
    "RRE_GOLDSET": ("paths", "goldset"),
    "RRE_RESULTS_DIR": ("paths", "results_dir"),
    "RRE_EMBEDDER": ("models", "embedder"),
    "RRE_RERANKER": ("models", "reranker"),
    "RRE_DEVICE": ("models", "device"),
    "RRE_QDRANT": ("qdrant", "location"),
    "RRE_COLLECTION": ("qdrant", "collection"),
}

DEFAULTS = {
    "paths": {"data_dir": "data", "work_dir": ".work", "goldset": "goldset/goldset.jsonl",
              "results_dir": "results/public"},
    "models": {"embedder": "BAAI/bge-m3", "reranker": "BAAI/bge-reranker-v2-m3",
               "max_seq_length": 1024, "reranker_max_length": 512, "device": "auto",
               "offline": True, "batch_size": 16},
    "qdrant": {"location": "local", "collection": "rre_bge_m3"},
    "chunk": {"max_chars": 900, "min_chars": 20},
    "retrieval": {"vector_candidates": 100, "topn_each": 20, "rrf_k": 60, "rerank_top": 20},
}


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


@dataclass
class Config:
    raw: dict

    def __getitem__(self, section: str) -> dict:
        return self.raw[section]

    def path(self, key: str) -> Path:
        p = Path(self.raw["paths"][key])
        return p if p.is_absolute() else ROOT / p

    @property
    def data_dir(self) -> Path:
        return self.path("data_dir")

    @property
    def work_dir(self) -> Path:
        d = self.path("work_dir")
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def passages_path(self) -> Path:
        return self.data_dir / "passages.jsonl"


def load_config(path: str | os.PathLike | None = None, env: dict | None = None) -> Config:
    env = os.environ if env is None else env
    path = path or env.get("RRE_CONFIG") or ROOT / "config.toml"
    raw = DEFAULTS
    p = Path(path)
    if p.exists():
        with open(p, "rb") as f:
            raw = _merge(DEFAULTS, tomllib.load(f))
    raw = copy.deepcopy(raw)
    for var, (sec, key) in ENV_OVERRIDES.items():
        if env.get(var):
            raw[sec][key] = env[var]
    if raw["models"].get("offline"):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    return Config(raw)
