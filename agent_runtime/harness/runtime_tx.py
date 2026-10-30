"""Restricted persistence capabilities available to transaction commands.

This view exposes detached projections and fact/blob writes, never a SQLite
connection, cursor, owner lifecycle, provider or tool execution capability.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

from agent_runtime.harness.rollout import RolloutStore


class RuntimeTx:
    __slots__ = ("__persistence",)

    def __init__(self, persistence: RolloutStore) -> None:
        self.__persistence = persistence

    @property
    def path(self) -> Path:
        return self.__persistence.path

    def transaction(self) -> AbstractContextManager[None]:
        return self.__persistence.transaction()

    def read_snapshot(self) -> AbstractContextManager[None]:
        return self.__persistence.read_snapshot()

    def first(self, *args: Any, **kwargs: Any) -> Mapping[str, Any] | None:
        return self.__persistence.first(*args, **kwargs)

    def rows(self, *args: Any, **kwargs: Any) -> tuple[Mapping[str, Any], ...]:
        return self.__persistence.rows(*args, **kwargs)

    def count(self, *args: Any, **kwargs: Any) -> int:
        return self.__persistence.count(*args, **kwargs)

    def maximum(self, *args: Any, **kwargs: Any) -> int:
        return self.__persistence.maximum(*args, **kwargs)

    def turn_with_workspace(self, *args: Any, **kwargs: Any) -> Mapping[str, Any] | None:
        return self.__persistence.turn_with_workspace(*args, **kwargs)

    def metadata(self, key: str) -> str | None:
        return self.__persistence.metadata(key)

    def put_metadata(self, *args: Any, **kwargs: Any) -> None:
        self.__persistence.put_metadata(*args, **kwargs)

    def delete_metadata(self, key: str) -> None:
        self.__persistence.delete_metadata(key)

    def append_record(self, **fact: Any) -> None:
        self.__persistence.append_record(**fact)

    def assert_writable(self) -> None:
        self.__persistence.assert_writable()

    def install_blob(self, *args: Any, **kwargs: Any) -> tuple[str, int]:
        return self.__persistence.install_blob(*args, **kwargs)

    def artifact_integrity_errors(self, *args: Any, **kwargs: Any) -> list[str]:
        return self.__persistence.artifact_integrity_errors(*args, **kwargs)
