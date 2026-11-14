"""全局模型配置库 ModelConfigStore(设计 §5.15;D4/D5;B26/B28)。

覆盖:
- 连接/模型 upsert 往返、默认模型、跨实例("跨进程")可见性;
- CAS:B26 的外部写不再造成永久冲突,连续冲突给出干净的 observed_revision;
- 凭据卫生:表结构只有 api_key_env,绝无 token/key/secret 值列;
- B28:有界 LRU 缓存 + 锁;同实例写后不脏读;
- 并发写不丢行、不死循环(CAS 有限重试);
- 安全面:decide 第 4 步对 <host_config_dir()>/config.sqlite 的读写一律地板 deny,
  以及 darwin seatbelt 内的真实读写强制(内容不变)。
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from praxis.adapters.config.store import (
    MODEL_CONFIG_SCHEMA_VERSION,
    ConfigVersionConflict,
    ConnectionEntry,
    ModelConfigStore,
    ModelEntry,
)
from praxis.adapters.sandbox.base import probe_backend
from praxis.adapters.sandbox.compile import compile_sandbox, finalize_sandbox
from praxis.core import permission
from praxis.protocol.floor import host_config_dir
from praxis.protocol.models import CostMicros, ModelSpec
from praxis.protocol.permission import PermissionContext
from praxis.protocol.policy import PathRule, PermissionPolicy
from praxis.protocol.tools import ResolvedUse, ToolDefinition, ToolEffect, ToolTarget
from tests.praxis.conftest import make_minimal_tool

_AVAILABLE = probe_backend()

_CONNECTION = ConnectionEntry(
    "deepseek",
    "openai-compatible",
    "https://api.deepseek.com/v1",
    "DEEPSEEK_API_KEY",
    "cloud",
)


def _model(model_id: str, connection: ConnectionEntry | None = None, **spec_fields: Any) -> ModelEntry:
    return ModelEntry(
        ModelSpec(model_id=model_id, **spec_fields),
        connection if connection is not None else _CONNECTION,
        revision=0,
    )


@pytest.fixture
def store(tmp_path: Path) -> Iterator[ModelConfigStore]:
    instance = ModelConfigStore(tmp_path / "config.sqlite")
    instance.upsert_connection(_CONNECTION)
    try:
        yield instance
    finally:
        instance.close()


# ---------------------------------------------------------------------------
# 往返与默认模型
# ---------------------------------------------------------------------------


def test_upsert_roundtrip_and_default(tmp_path: Path) -> None:
    s = ModelConfigStore(tmp_path / "config.sqlite")
    try:
        s.upsert_connection(_CONNECTION)
        rev = s.upsert_model(_model("deepseek-flash", remote_model_id="deepseek-chat"))
        entry = s.get("deepseek-flash")
        assert entry is not None and entry.spec.wire_model_id == "deepseek-chat"
        # connection 是唯一连接来源:base_url/api_key_env 由连接行填充。
        assert entry.spec.base_url == _CONNECTION.base_url
        assert entry.spec.api_key_env == _CONNECTION.api_key_env
        assert entry.connection == _CONNECTION
        s.set_default_model("deepseek-flash", expected_revision=rev)
        # 新实例模拟跨进程:默认模型必须可见。
        fresh = ModelConfigStore(tmp_path / "config.sqlite")
        try:
            assert fresh.resolve_spec(None).model_id == "deepseek-flash"
        finally:
            fresh.close()
    finally:
        s.close()


def test_full_spec_and_cost_roundtrip(store: ModelConfigStore) -> None:
    spec_fields = dict(
        remote_model_id="wire-name",
        context_window_tokens=32768,
        max_output_tokens=2048,
        supports_tools=False,
        tokenizer_model="cl100k_base",
        timeout_seconds=33.5,
        auto_compact_token_limit=20000,
        provider_options={"top_p": 0.9, "seed": 7},
    )
    store.upsert_model(_model("configured", cost=CostMicros(1_000_000, 2_000_000, 300, 400), **spec_fields))
    entry = store.get("configured")
    assert entry is not None
    spec = entry.spec
    assert spec.remote_model_id == "wire-name"
    assert spec.context_window_tokens == 32768
    assert spec.max_output_tokens == 2048
    assert spec.supports_tools is False
    assert spec.tokenizer_model == "cl100k_base"
    assert spec.timeout_seconds == 33.5
    assert spec.auto_compact_token_limit == 20000
    assert dict(spec.provider_options) == {"top_p": 0.9, "seed": 7}
    assert spec.cost == CostMicros(1_000_000, 2_000_000, 300, 400)


def test_list_models_sorted_and_remove_clears_default(store: ModelConfigStore) -> None:
    store.upsert_model(_model("b"))
    store.upsert_model(_model("a"))
    assert [entry.spec.model_id for entry in store.list_models()] == ["a", "b"]
    rev = store.revision
    store.set_default_model("a", expected_revision=rev)
    rev = store.revision
    store.remove_model("a", expected_revision=rev)
    assert store.get("a") is None
    assert store.default_model() is None
    with pytest.raises(KeyError, match="not registered"):
        store.remove_model("a", expected_revision=store.revision)


def test_resolve_spec_missing_is_actionable(tmp_path: Path) -> None:
    s = ModelConfigStore(tmp_path / "config.sqlite")
    try:
        with pytest.raises(KeyError, match="no models registered; run: praxis models add"):
            s.resolve_spec(None)
        s.upsert_connection(_CONNECTION)
        s.upsert_model(_model("only"))
        with pytest.raises(KeyError, match="no default model; run: praxis models set-default"):
            s.resolve_spec(None)
        with pytest.raises(KeyError, match="not registered"):
            s.resolve_spec("missing")
    finally:
        s.close()


# ---------------------------------------------------------------------------
# B26:CAS 不再永久冲突
# ---------------------------------------------------------------------------


def test_external_write_then_switch_does_not_stay_conflicted(tmp_path: Path) -> None:
    db = tmp_path / "config.sqlite"
    a = ModelConfigStore(db)
    b = ModelConfigStore(db)
    try:
        a.upsert_connection(_CONNECTION)
        ra = a.upsert_model(_model("alpha"), expected_revision=a.revision)
        # b 的本地视角陈旧,但写路径在同一次 IMMEDIATE 事务内 re-read,
        # 因此按 DB 真值校验,不会误报"永久冲突"。
        rb = b.upsert_model(_model("beta"), expected_revision=ra)
        assert rb > ra
        assert a.revision == b.revision == rb
        # 连续外部写之后:陈旧 expected_revision 得到干净报错并带 observed。
        with pytest.raises(ConfigVersionConflict) as excinfo:
            a.upsert_model(_model("gamma"), expected_revision=0)
        assert excinfo.value.observed_revision == rb
        # 不永久冲突:用 observed 重试即可推进(内存/数据库视角已刷新)。
        replay = a.upsert_model(_model("gamma"), expected_revision=excinfo.value.observed_revision)
        assert replay == rb + 1
    finally:
        a.close()
        b.close()


def test_remove_and_set_default_honour_cas(store: ModelConfigStore) -> None:
    store.upsert_model(_model("a"))
    stale = store.revision - 1  # 已知陈旧
    with pytest.raises(ConfigVersionConflict) as excinfo:
        store.remove_model("a", expected_revision=stale)
    assert excinfo.value.observed_revision == store.revision
    with pytest.raises(ConfigVersionConflict) as excinfo:
        store.set_default_model("a", expected_revision=stale)
    assert excinfo.value.observed_revision == store.revision
    assert store.get("a") is not None
    # 失败事务保留了 observable 状态,后续用最新 revision 可成功。
    store.set_default_model("a", expected_revision=store.revision)
    assert store.default_model() == "a"


# ---------------------------------------------------------------------------
# 凭据卫生:表结构绝无凭据值列
# ---------------------------------------------------------------------------


def test_no_credential_columns(tmp_path: Path) -> None:
    assert MODEL_CONFIG_SCHEMA_VERSION == "1"
    s = ModelConfigStore(tmp_path / "config.sqlite")
    try:
        rows = s._conn.execute("SELECT sql FROM sqlite_master WHERE type = 'table'").fetchall()
    finally:
        s.close()
    schema = " ".join(str(row[0]).lower() for row in rows)
    assert schema  # 非空:确实读到了表结构
    assert "token" not in schema
    assert "secret" not in schema
    assert "api_key" not in schema.replace("api_key_env", "")
    assert "api_key_env" in schema


# ---------------------------------------------------------------------------
# B28:有界缓存 + 同实例写后不脏读
# ---------------------------------------------------------------------------


def test_cache_bounded_and_not_dirty_after_write(store: ModelConfigStore) -> None:
    first = store.get("probe")
    assert first is None
    store.upsert_model(_model("probe", max_output_tokens=4096))
    assert store.get("probe") is not None  # type: ignore[union-attr]
    # 同实例写成功即清缓存:改值后立刻读到新值,不脏读。
    store.upsert_model(_model("probe", max_output_tokens=1024))
    updated = store.get("probe")
    assert updated is not None and updated.spec.max_output_tokens == 1024
    # 200 个模型,缓存上限 64(LRU 淘汰),公开读取仍完整。
    for index in range(200):
        store.upsert_model(_model(f"m{index:03d}"))
    assert len(store.list_models()) == 201
    assert len(store._cache) <= 64
    assert store.get("m199") is not None
    assert "m199" in store._cache


# ---------------------------------------------------------------------------
# 并发:有界线程数,不丢行、CAS 不死循环
# ---------------------------------------------------------------------------


def _join_all(threads: list[threading.Thread], errors: list[BaseException]) -> None:
    for thread in threads:
        thread.join(timeout=30)
    assert not any(thread.is_alive() for thread in threads), "worker thread did not finish"
    assert errors == []


@pytest.mark.parametrize(("n_threads", "n_ops"), [(4, 8), (6, 4)])
def test_concurrent_writers_serialize(tmp_path: Path, n_threads: int, n_ops: int) -> None:
    db = tmp_path / "config.sqlite"
    seed = ModelConfigStore(db)
    seed.upsert_connection(_CONNECTION)
    seed.close()

    barrier = threading.Barrier(n_threads)
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        instance = ModelConfigStore(db)
        try:
            barrier.wait(timeout=10)
            for op in range(n_ops):
                instance.upsert_model(_model(f"t{index}-{op}"))
        except BaseException as exc:  # noqa: BLE001 — 收集到主线程断言
            errors.append(exc)
        finally:
            instance.close()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for thread in threads:
        thread.start()
    _join_all(threads, errors)

    final = ModelConfigStore(db)
    try:
        assert len(final.list_models()) == n_threads * n_ops
        assert final.revision == 1 + n_threads * n_ops  # 1 次连接写 + N 次模型写
    finally:
        final.close()


def test_concurrent_cas_writers_do_not_livelock(tmp_path: Path) -> None:
    db = tmp_path / "config.sqlite"
    seed = ModelConfigStore(db)
    seed.upsert_connection(_CONNECTION)
    seed.close()

    n_threads, per_thread = 4, 3
    barrier = threading.Barrier(n_threads)
    errors: list[BaseException] = []

    def worker(index: int) -> None:
        instance = ModelConfigStore(db)
        try:
            barrier.wait(timeout=10)
            for op in range(per_thread):
                model_id = f"cas-{index}-{op}"
                for _attempt in range(100):
                    try:
                        instance.upsert_model(_model(model_id), expected_revision=instance.revision)
                        break
                    except ConfigVersionConflict:
                        continue
                else:
                    raise AssertionError(f"CAS livelock for {model_id}")
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            instance.close()

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for thread in threads:
        thread.start()
    _join_all(threads, errors)

    final = ModelConfigStore(db)
    try:
        assert len(final.list_models()) == n_threads * per_thread
    finally:
        final.close()


# ---------------------------------------------------------------------------
# 安全面 1:decide 第 4 步对配置库的读写一律地板 deny
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("effect", "tool_name"),
    [
        (ToolEffect.READ_WORKSPACE, "read"),
        (ToolEffect.WRITE_WORKSPACE, "write"),
    ],
)
def test_config_db_denied_by_floor_via_decide(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, effect: ToolEffect, tool_name: str
) -> None:
    monkeypatch.setenv("PRAXIS_CONFIG_DIR", str(tmp_path / "praxis-config"))
    workspace = tmp_path / "ws"
    workspace.mkdir()
    target = host_config_dir() / "config.sqlite"
    effects = frozenset({effect})
    tool = make_minimal_tool(
        definition=ToolDefinition(
            name=tool_name, description="floor probe", input_schema={"type": "object"}
        ),
        static_effects=effects,
        resolve_use=lambda arguments: ResolvedUse(
            effects=effects, targets=(ToolTarget(kind="path", value=str(target)),)
        ),
    )
    use = ResolvedUse(effects=effects, targets=(ToolTarget(kind="path", value=str(target)),))
    # 显式授权配置目录,让第 1 步放行;此时地板必须是唯一且不可绕过的拒绝者。
    policy = PermissionPolicy(filesystem=(PathRule(path=str(host_config_dir()), access="write"),))
    ctx = PermissionContext(workspace=str(workspace), policy=policy, mode="bypass", sandbox=_AVAILABLE)
    decision = permission.decide(tool, use, ctx)
    assert decision.is_deny, decision
    assert decision.reason.startswith("floor:"), decision.reason


# ---------------------------------------------------------------------------
# 安全面 2:真实 seatbelt 内读/写配置库都失败且内容不变
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    sys.platform != "darwin" or not _AVAILABLE.available or _AVAILABLE.backend != "seatbelt",
    reason="real bash enforcement needs the darwin seatbelt backend",
)
def test_real_sandbox_denies_config_store_read_and_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config_dir = tmp_path / "praxis-config"
    config_dir.mkdir()
    db = config_dir / "config.sqlite"
    db.write_text("orig\n", encoding="utf-8")
    monkeypatch.setenv("PRAXIS_CONFIG_DIR", str(config_dir))
    workspace = tmp_path / "ws"
    workspace.mkdir()
    ctx = PermissionContext(workspace=str(workspace), policy=PermissionPolicy(), mode="on-request")
    use = ResolvedUse(
        effects=frozenset({ToolEffect.EXECUTE_PROCESS}),
        targets=(ToolTarget(kind="path", value=str(workspace)),),
    )
    spec = compile_sandbox(ctx, use, workspace)
    try:
        write = subprocess.run(
            [*spec.argv_prefix, "/bin/bash", "-lc", f"echo hacked > {db}"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=30,
        )
        read = subprocess.run(
            [*spec.argv_prefix, "/bin/bash", "-lc", f"cat {db}"],
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=30,
        )
    finally:
        finalize_sandbox(spec, workspace)
    assert write.returncode != 0, write
    assert read.returncode != 0, read
    assert db.read_text(encoding="utf-8") == "orig\n"


def test_store_path_defaults_to_host_config_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PRAXIS_CONFIG_DIR", str(tmp_path / "host-config"))
    instance = ModelConfigStore()
    try:
        assert instance._path == host_config_dir() / "config.sqlite"
        assert instance._path.exists()
        assert instance._path.parent.stat().st_mode & 0o777 == 0o700
    finally:
        instance.close()


def test_cost_json_is_plain_integers(store: ModelConfigStore) -> None:
    store.upsert_model(_model("priced", cost=CostMicros(11, 22, 33, 44)))
    entry = store.get("priced")
    assert entry is not None
    raw = json.loads(
        store._conn.execute("SELECT cost FROM models WHERE model_id = 'priced'").fetchone()[0]
    )
    assert raw == {
        "input_per_1m": 11,
        "output_per_1m": 22,
        "cache_read_per_1m": 33,
        "cache_write_per_1m": 44,
    }


def test_schema_tables_are_the_documented_three(tmp_path: Path) -> None:
    instance = ModelConfigStore(tmp_path / "config.sqlite")
    try:
        tables = {
            str(row[0])
            for row in instance._conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
    finally:
        instance.close()
    assert {"model_config", "connections", "models"} <= tables


def test_sqlite_integrity_check(store: ModelConfigStore) -> None:
    assert store._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert sqlite3.sqlite_version_info >= (3, 24)