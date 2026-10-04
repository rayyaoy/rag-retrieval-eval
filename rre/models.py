"""模型加载：向量模型（BGE-M3）与精排模型（bge-reranker-v2-m3）。名称都来自配置。

另有 HashEmbedder：不下载任何模型的确定性假向量（字符 n-gram 哈希），只用于单元测试与冒烟测试，
不代表任何检索质量。
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Sequence

import numpy as np


def pick_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class STEmbedder:
    """sentence-transformers 封装。fp16 只在 GPU（cuda/mps）上开。"""

    def __init__(self, name: str, max_seq_length: int = 1024, device: str = "auto",
                 batch_size: int = 16):
        import torch
        from sentence_transformers import SentenceTransformer
        self.device = pick_device(device)
        kw = {"torch_dtype": torch.float16} if self.device in ("mps", "cuda") else {}
        self.model = SentenceTransformer(name, device=self.device, model_kwargs=kw)
        self.model.max_seq_length = max_seq_length
        self.name, self.batch_size = name, batch_size
        get_dim = getattr(self.model, "get_embedding_dimension", None) or \
            self.model.get_sentence_embedding_dimension
        self.dim = get_dim()
        self.dtype = "float16" if kw else "float32"

    def encode(self, texts: Sequence[str], progress: bool = False) -> np.ndarray:
        v = self.model.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True,
                              show_progress_bar=progress, convert_to_numpy=True)
        return v.astype("float32")


class CrossEncoderReranker:
    def __init__(self, name: str, max_length: int = 512, device: str = "auto"):
        import torch
        from sentence_transformers import CrossEncoder
        self.device = pick_device(device)
        kw = {"torch_dtype": torch.float16} if self.device in ("mps", "cuda") else {}
        self.model = CrossEncoder(name, device=self.device, max_length=max_length, model_kwargs=kw)
        self.name = name

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        s = self.model.predict([(query, t) for t in texts], batch_size=16, show_progress_bar=False)
        return [float(x) for x in s]


class HashEmbedder:
    """测试用：字符 1–2 gram 哈希到 dim 维再归一化。确定、无依赖、无语义。"""

    name = "hash-ngram-test"
    dtype = "float32"

    def __init__(self, dim: int = 256):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype="float32")
        t = re.sub(r"\s+", " ", text.lower())
        for n in (1, 2):
            for i in range(len(t) - n + 1):
                h = int.from_bytes(hashlib.md5(t[i:i + n].encode()).digest()[:4], "little")
                v[h % self.dim] += 1.0
        norm = math.sqrt(float((v * v).sum())) or 1.0
        return v / norm

    def encode(self, texts: Sequence[str], progress: bool = False) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), "float32")


class OverlapReranker:
    """测试用：按查询与段落的字符重合度打分。"""

    name = "overlap-test"

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        q = set(query.lower())
        return [len(q & set(t.lower())) / (len(q) or 1) for t in texts]
