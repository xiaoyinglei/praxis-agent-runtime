"""Praxis 测试公共 fixture(Task 4 起):workspace 目录与独立 db 的 Store。

make_turn 等 turn 工厂助手留待 Task 7 补充,此处只建最小公共面。
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from praxis.store.log import Store


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    return root


@pytest.fixture
def store(workspace: Path) -> Iterator[Store]:
    s = Store(workspace / "store.db")
    try:
        yield s
    finally:
        s.close()
