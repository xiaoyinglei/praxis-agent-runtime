from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator

from rag.schema.runtime import CacheEntry
from rag.storage.repositories.redis_cache_repo import RedisCacheRepo
from rag.storage.repositories.sqlite_metadata_repo import SQLiteMetadataRepo


def _runtime_payload() -> dict[str, str | int | bool]:
    return {
        "embedding_model_name": "local-embed",
        "chunk_token_size": 512,
        "rerank_enabled": False,
    }


def _assert_json_primitives(value: object) -> None:
    if value is None or type(value) in {bool, int, float, str}:
        return
    if isinstance(value, list):
        for item in value:
            _assert_json_primitives(item)
        return
    if isinstance(value, dict):
        assert all(isinstance(key, str) for key in value)
        for item in value.values():
            _assert_json_primitives(item)
        return
    raise AssertionError(f"non-primitive persisted value: {value!r}")


def _assert_safe_json(raw: str) -> None:
    _assert_json_primitives(json.loads(raw))
    assert "agent_runtime.modeling.config" not in raw
    assert "agent_runtime.modeling.contracts" not in raw
    assert "__module__" not in raw


class _FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def get(self, key: str) -> str | None:
        return self.values.get(key)

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        del ex
        self.values[key] = value

    def delete(self, key: str) -> int:
        return int(self.values.pop(key, None) is not None)

    def scan_iter(self, *, match: str) -> Iterator[str]:
        del match
        return iter(self.values)


def test_sqlite_cache_entry_persists_primitive_runtime_contract(tmp_path) -> None:
    entry = CacheEntry(namespace="runtime", cache_key="bundle", payload=_runtime_payload())
    repo = SQLiteMetadataRepo(tmp_path / "metadata.sqlite")

    dumped = repo._dump(entry)
    assert repo._load(CacheEntry, dumped) == entry
    repo.save_cache_entry(entry)
    assert repo.get_cache_entry("bundle", namespace="runtime").payload == _runtime_payload()
    with sqlite3.connect(tmp_path / "metadata.sqlite") as connection:
        raw = connection.execute(
            "SELECT payload FROM cache_entries WHERE namespace = ? AND cache_key = ?",
            ("runtime", "bundle"),
        ).fetchone()[0]
    _assert_safe_json(raw)


def test_redis_cache_entry_persists_primitive_runtime_contract() -> None:
    fake = _FakeRedis()
    entry = CacheEntry(namespace="runtime", cache_key="bundle", payload=_runtime_payload())
    repo = RedisCacheRepo("redis://unused", client=fake)

    repo.save_cache_entry(entry)
    assert repo.get_cache_entry("bundle", namespace="runtime").payload == _runtime_payload()
    _assert_safe_json(fake.values["rag-cache:runtime:bundle"])
