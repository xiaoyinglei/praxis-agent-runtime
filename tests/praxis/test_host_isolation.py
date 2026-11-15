"""N-6:默认套件不得触碰真实宿主目录,但运维显式环境优先。

``tests/praxis/conftest.py`` 的 autouse ``_isolate_host_state`` 调用
``host_isolation_updates`` 决定是否重定向 ``PRAXIS_CONFIG_DIR`` / ``TMPDIR``。
本文件把该纯函数契约与 fixture 的实际决策都变成可断言行为。
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from tests.praxis.conftest import _ISOLATION_DECISIONS, host_isolation_updates


def test_updates_honor_operator_config_dir(tmp_path: Path) -> None:
    """运维显式 PRAXIS_CONFIG_DIR(授权 live 冒烟)时不做任何重定向。"""
    assert host_isolation_updates(tmp_path / "base", {"PRAXIS_CONFIG_DIR": "/ext/reg"}) == {}


def test_updates_isolate_by_default(tmp_path: Path) -> None:
    base = tmp_path / "base"
    updates = host_isolation_updates(base, {"TMPDIR": "/host/tmp"})
    host_root = base.with_name(f"{base.name}-host-state")
    assert updates["PRAXIS_CONFIG_DIR"] == str(host_root / "config")
    assert updates["TMPDIR"] == str(host_root / "tmp")
    assert updates["TEMP"] == updates["TMP"] == updates["TMPDIR"]


def test_fixture_decision_matches_operator_environment() -> None:
    """fixture 实际决策:外部已设 → 原样保留;未设 → per-test tmp。"""
    operator_config = _ISOLATION_DECISIONS["operator_config_dir"]
    reused = _ISOLATION_DECISIONS["reused_operator_env"]
    updates = _ISOLATION_DECISIONS["updates"]
    assert reused is bool(operator_config)

    if operator_config:
        assert updates == {}
        assert os.environ["PRAXIS_CONFIG_DIR"] == operator_config
    else:
        assert os.environ["PRAXIS_CONFIG_DIR"] == updates["PRAXIS_CONFIG_DIR"]
        assert "-host-state/config" in os.environ["PRAXIS_CONFIG_DIR"]
        assert tempfile.gettempdir() == updates["TMPDIR"]
        assert "-host-state/tmp" in tempfile.gettempdir()
        # 真实宿主注册库绝不在隔离路径内(零新增的路径级证据)。
        real_registry = Path.home().resolve() / ".config" / "praxis"
        isolated = Path(os.environ["PRAXIS_CONFIG_DIR"]).resolve()
        assert isolated != real_registry
        assert not isolated.is_relative_to(real_registry)


def test_default_model_config_store_stays_isolated() -> None:
    """默认套件里 ``ModelConfigStore()`` 的 sqlite 只落在 per-test 目录。"""
    if _ISOLATION_DECISIONS["operator_config_dir"]:
        pytest.skip("operator run: the real registry path is intentional")
    from praxis.adapters.config.store import ModelConfigStore

    store = ModelConfigStore()
    try:
        db = Path(os.environ["PRAXIS_CONFIG_DIR"]) / "config.sqlite"
        assert db.exists()
        assert str(db).startswith(os.environ["PRAXIS_CONFIG_DIR"])
    finally:
        store.close()
