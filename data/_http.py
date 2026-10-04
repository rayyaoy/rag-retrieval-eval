"""下载小工具：标准库实现，处理 gzip、重试和 sha256。"""
from __future__ import annotations

import gzip
import http.client
import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "manifest.json"


def get(url: str, headers: dict, timeout: int = 120, retries: int = 6) -> bytes:
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip", **headers})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
                if r.headers.get("Content-Encoding") == "gzip" or body[:2] == b"\x1f\x8b":
                    body = gzip.decompress(body)
                return body
        except urllib.error.HTTPError as e:      # 429 限流退避重试；其他 4xx 不重试
            if 400 <= e.code < 500 and e.code != 429:
                raise
            last = e
            if e.code == 429:
                time.sleep(10 * (i + 1))
        except (urllib.error.URLError, TimeoutError, ConnectionError, http.client.IncompleteRead) as e:
            last = e
        time.sleep(2 * (i + 1))
    raise last  # type: ignore[misc]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def save_manifest(m: dict) -> None:
    MANIFEST.write_text(json.dumps(m, ensure_ascii=False, indent=1, sort_keys=True) + "\n")


def check_or_record(kind: str, key: str, digest: str, record: bool) -> str:
    """有记录就比对，没有记录且 record=True 就写入。返回 ok / recorded / MISMATCH / unknown。"""
    m = load_manifest()
    sec = m.setdefault(kind, {})
    if key in sec:
        return "ok" if sec[key] == digest else "MISMATCH"
    if record:
        sec[key] = digest
        save_manifest(m)
        return "recorded"
    return "unknown"
