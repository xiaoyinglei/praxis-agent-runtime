"""Task 8:内容寻址 artifact 存储 + 工具结果写时外部化 + read_artifact(设计 §5.7;D10)。

覆盖:put/get 完整性(GET 后重算 sha256)、0600、非法 digest、离线 GC、
referenced_by fail-open、写时外部化的三条纪律(指针进 item / 全文在 artifact /
全文不进 item)、无 artifacts 注入的降级路径、read_artifact 的分页与错误码。
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, cast

import pytest

from praxis.core.events import EventBus
from praxis.core.pipeline import ExecCtx, execute
from praxis.protocol.items import Item
from praxis.protocol.json import thaw_json
from praxis.protocol.permission import PermissionContext
from praxis.protocol.policy import PermissionPolicy
from praxis.protocol.sandbox import ToolExecContext
from praxis.protocol.tools import (
    CancellationMode,
    InterruptBehavior,
    ResolvedUse,
    ResultStatus,
    Tool,
    ToolCall,
    ToolContentBlock,
    ToolDefinition,
    ToolEffect,
    ToolResult,
    json_schema_input,
)
from praxis.store.artifacts import ArtifactIntegrityError, ArtifactStore
from praxis.store.log import Store
from praxis.store.turns import create_turn
from praxis.tools.builtins.artifacts_tool import create_read_artifact_tool
from praxis.tools.builtins.filesystem import create_read_tool

DIGEST_A = "a" * 64


# ---------------------------------------------------------------------------
# 测试辅助
# ---------------------------------------------------------------------------


def _ctx(
    store: Store,
    workspace: Path,
    tools: dict[str, Tool],
    artifacts: ArtifactStore | None = None,
) -> ExecCtx:
    policy = PermissionPolicy()
    turn = create_turn(store, turn_id="turn-art", thread_id="thread-art", binding={})
    return ExecCtx(
        store=store,
        bus=EventBus(),
        workspace=workspace,
        tools=tools,
        permissions=PermissionContext(workspace=str(workspace), policy=policy, mode="on-request"),
        turn_id=turn.turn_id,
        thread_id="thread-art",
        policy_revision=policy.revision,
        artifacts=artifacts,
    )


def _text_tool(text: str, max_bytes: int) -> Tool:
    schema: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}
    validate = json_schema_input(schema)

    async def run(arguments: dict, exec_ctx: ToolExecContext) -> ToolResult:
        return ToolResult(
            tool_call_id="pending",
            tool_name="big",
            content=(ToolContentBlock(type="text", data={"text": text}),),
            status=ResultStatus.OK,
        )

    return Tool(
        definition=ToolDefinition(name="big", description="big output toy", input_schema=schema),
        validate_input=validate,
        run=run,
        static_effects=frozenset(),
        resolve_use=lambda arguments: ResolvedUse(effects=frozenset(), targets=()),
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=30,
        max_model_output_bytes=max_bytes,
    )


def _result_item(payload: dict[str, Any], kind: str = "tool_result") -> Item:
    return Item(
        item_id="i",
        thread_id="t",
        turn_id="turn",
        seq=1,
        kind=kind,
        payload=payload,
        created_at=0.0,
    )


def _stored_text(store: Store, turn_id: str) -> str:
    item = store.read_turn_items(turn_id)[-1]
    return cast("str", item.payload["content"][0]["data"]["text"])  # type: ignore[index]


# ---------------------------------------------------------------------------
# ArtifactStore:roundtrip / 完整性 / 0600 / 非法 digest
# ---------------------------------------------------------------------------


def test_put_get_roundtrip_and_mode(tmp_path: Path) -> None:
    a = ArtifactStore(tmp_path)
    d = a.put(b"x" * 5000)
    assert len(d) == 64 and a.get(d) == b"x" * 5000
    assert (tmp_path / d[:2] / d).stat().st_mode & 0o077 == 0  # 0600:group/other 无权限


def test_put_is_content_addressed(tmp_path: Path) -> None:
    a = ArtifactStore(tmp_path)
    d1 = a.put(b"same")
    path = tmp_path / d1[:2] / d1
    old = 1_000_000.0
    os.utime(path, (old, old))
    d2 = a.put(b"same")
    assert d1 == d2
    assert list(a.iter_digests()) == [d1]
    assert a.size(d1) == 4
    assert path.stat().st_mtime == old  # 已存在即返回,不重写(去重)


def test_missing_artifact_returns_none(tmp_path: Path) -> None:
    a = ArtifactStore(tmp_path)
    assert a.get(DIGEST_A) is None
    assert a.size(DIGEST_A) is None


@pytest.mark.parametrize("bad", ["../etc/passwd", "ABC", "a" * 63, "a" * 65, "", "g" * 64, "A" * 64])
def test_invalid_digest_raises_value_error(tmp_path: Path, bad: str) -> None:
    a = ArtifactStore(tmp_path)
    with pytest.raises(ValueError, match="invalid artifact digest"):
        a.get(bad)
    with pytest.raises(ValueError, match="invalid artifact digest"):
        a.size(bad)


def test_get_and_size_map_oserror_to_missing(tmp_path: Path) -> None:
    """目录替代文件/不可读 → None(由工具映射 artifact_not_found),不炸穿。"""
    a = ArtifactStore(tmp_path)
    d = a.put(b"x")
    path = tmp_path / d[:2] / d
    path.unlink()
    path.mkdir()
    assert a.get(d) is None
    assert a.size(d) is None


def test_integrity_mismatch_raises(tmp_path: Path) -> None:
    a = ArtifactStore(tmp_path)
    d = a.put(b"good")
    (tmp_path / d[:2] / d).write_bytes(b"tampered")
    with pytest.raises(ArtifactIntegrityError):
        a.get(d)


# ---------------------------------------------------------------------------
# 离线 GC 与 referenced_by(写路径永不调用)
# ---------------------------------------------------------------------------


def test_gc_keeps_referenced_and_fresh(tmp_path: Path) -> None:
    a = ArtifactStore(tmp_path)
    referenced = a.put(b"keep referenced")
    stale = a.put(b"stale unreferenced")
    fresh = a.put(b"fresh unreferenced")
    old = time.time() - 7200
    for digest in (referenced, stale):
        os.utime(tmp_path / digest[:2] / digest, (old, old))

    removed = a.gc({referenced}, now=time.time(), grace_seconds=3600)

    assert removed == 1
    assert a.get(referenced) is not None
    assert a.get(stale) is None
    assert a.get(fresh) is not None  # 未引用但新鲜 → 宽限期内保留


def test_symlinked_shard_and_file_are_ignored(tmp_path: Path) -> None:
    """GC 离线运行:存储内指向外部的符号链接不能被枚举/删除。"""
    root = tmp_path / "artifacts"
    root.mkdir()
    a = ArtifactStore(root)
    outside = tmp_path / "outside"
    outside.mkdir()
    shard_digest = "ab" + "c" * 62
    (outside / shard_digest).write_bytes(b"outside file")
    old = time.time() - 7200
    os.utime(outside / shard_digest, (old, old))
    (root / "ab").symlink_to(outside, target_is_directory=True)

    file_digest = "cd" + "e" * 62
    link_target = tmp_path / "link-target"
    link_target.write_bytes(b"linked")
    (root / "cd").mkdir()
    (root / "cd" / file_digest).symlink_to(link_target)

    assert list(a.iter_digests()) == []
    assert a.gc(set(), now=time.time()) == 0  # 不删任何东西
    assert (outside / shard_digest).exists()
    assert link_target.exists()


def test_referenced_by_fail_open() -> None:
    other = "b" * 64
    items = [
        _result_item({"metadata": {"artifact_digest": DIGEST_A}}),
        _result_item({"metadata": {"artifact_digest": other}}, kind="model_response"),  # 未知 kind
        _result_item({"metadata": "not-a-mapping"}),
        _result_item({"metadata": {"artifact_digest": 123}}),
        _result_item({}),
        _result_item({"metadata": None}),
    ]
    assert ArtifactStore.referenced_by(items) == {DIGEST_A}


# ---------------------------------------------------------------------------
# 管道写时外部化(B22:全文只存一份)
# ---------------------------------------------------------------------------


def test_pipeline_externalizes_oversized_result(store: Store, workspace: Path, tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    artifacts = ArtifactStore(root)
    full = "z" * 100_000
    ctx = _ctx(store, workspace, {"big": _text_tool(full, 4096)}, artifacts=artifacts)

    result = asyncio.run(execute(ToolCall("c1", "big", {}), ctx))

    assert result.truncated and result.metadata["artifact_bytes"] == 100_000
    digest = cast("str", result.metadata["artifact_digest"])
    assert len(digest) == 64
    stored = ctx.store.read_turn_items(ctx.turn_id)[-1]
    assert stored.kind == "tool_result"
    # 1) 摘要进 item;2) 全文在 artifact;3) 全文不进 item(B22 纪律)
    assert stored.payload["metadata"]["artifact_digest"] == digest
    assert ArtifactStore(root).get(digest) == full.encode("utf-8")
    assert full not in json.dumps(thaw_json(stored.payload))
    # item 的 content 正是指针 + 头部片段,不是全文
    assert stored.payload["content"][0]["data"]["text"] == result.text
    assert len(result.text.encode("utf-8")) < 4096
    assert f"artifact://{digest}" in result.text
    assert result.text.startswith("z" * 2048)


def test_externalized_head_never_splits_multibyte(store: Store, workspace: Path, tmp_path: Path) -> None:
    artifacts = ArtifactStore(tmp_path / "artifacts")
    full = "汉字🙂" * 5000  # 每字符组 10 字节;limit=1001 会切在第 100 个字符之后
    ctx = _ctx(store, workspace, {"big": _text_tool(full, 1001)}, artifacts=artifacts)

    result = asyncio.run(execute(ToolCall("c1", "big", {}), ctx))

    digest = cast("str", result.metadata["artifact_digest"])
    head = result.text.split("\n[full output stored:")[0]
    assert "�" not in result.text  # 不产出替换字符
    assert full.startswith(head)  # 头部是完整字符前缀
    assert artifacts.get(digest) == full.encode("utf-8")


def test_put_oserror_degrades_to_truncation(
    store: Store, workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """磁盘满/权限错:如实降级为截断并标 artifact_unavailable,不假装存好了。"""

    def _disk_full(self: ArtifactStore, data: bytes) -> str:
        raise OSError("disk full")

    monkeypatch.setattr(ArtifactStore, "put", _disk_full)
    artifacts = ArtifactStore(tmp_path / "artifacts")
    ctx = _ctx(store, workspace, {"big": _text_tool("w" * 10_000, 256)}, artifacts=artifacts)

    result = asyncio.run(execute(ToolCall("c1", "big", {}), ctx))

    assert result.truncated
    assert result.metadata["artifact_unavailable"] is True
    assert "artifact_digest" not in result.metadata
    assert len(result.text.encode("utf-8")) <= 256
    stored = ctx.store.read_turn_items(ctx.turn_id)[-1]
    assert stored.payload["metadata"]["artifact_unavailable"] is True


def test_pipeline_without_artifacts_keeps_byte_truncation(
    store: Store, workspace: Path, tmp_path: Path
) -> None:
    """降级路径:未注入 artifacts(单测/裸管道)→ 旧字节截断语义原样保留。"""
    ctx = _ctx(store, workspace, {"big": _text_tool("x" * 100, 10)}, artifacts=None)

    result = asyncio.run(execute(ToolCall("c1", "big", {}), ctx))

    assert result.truncated and len(result.text.encode("utf-8")) <= 10
    assert result.metadata.get("artifact_digest") is None
    assert not (tmp_path / "artifacts").exists()  # 没有任何落盘
    stored = ctx.store.read_turn_items(ctx.turn_id)[-1]
    assert stored.payload["content"][0]["data"]["text"] == "x" * 10


def test_write_path_does_not_run_gc_or_add_tables(
    store: Store, workspace: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GC 只能离线:execute 的写路径除那次 put 外不碰 artifact 事务,也不加表。"""

    def _no_gc(*args: object, **kwargs: object) -> int:
        raise AssertionError("ArtifactStore.gc must never run in the write path")

    monkeypatch.setattr(ArtifactStore, "gc", _no_gc)
    db = workspace / "store.db"

    def tables() -> set[str]:
        with sqlite3.connect(db) as conn:
            return {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}

    before = tables()
    artifacts = ArtifactStore(tmp_path / "artifacts")
    ctx = _ctx(store, workspace, {"big": _text_tool("q" * 10_000, 128)}, artifacts=artifacts)
    result = asyncio.run(execute(ToolCall("c1", "big", {}), ctx))

    assert result.truncated
    assert tables() == before  # D10:artifact 不进 SQLite,写路径不新增表
    assert len(list(artifacts.iter_digests())) == 1  # 只有 put 落的那一个文件


# ---------------------------------------------------------------------------
# read_artifact 工具
# ---------------------------------------------------------------------------


def test_read_artifact_pages(tmp_path: Path) -> None:
    a = ArtifactStore(tmp_path)
    digest = a.put(b"0123456789")
    tool = create_read_artifact_tool(a)

    first = asyncio.run(tool.run({"digest": digest, "offset": 2, "max_bytes": 3}, ToolExecContext()))
    assert first.status is ResultStatus.OK
    assert first.text == "234"
    assert first.truncated and first.metadata["next_offset"] == 5
    assert first.metadata["size_bytes"] == 10

    tail = asyncio.run(tool.run({"digest": digest, "offset": 8}, ToolExecContext()))
    assert tail.text == "89"
    assert tail.truncated is False and tail.metadata["next_offset"] is None


def test_read_artifact_full_page_is_body_not_pointer(
    store: Store, workspace: Path, tmp_path: Path
) -> None:
    """单页上限必须严格小于工具自身输出上限:满页不能再被外部化。"""
    artifacts = ArtifactStore(tmp_path / "artifacts")
    payload = "q" * 196_608
    digest = artifacts.put(payload.encode("utf-8"))
    ctx = _ctx(
        store, workspace, {"read_artifact": create_read_artifact_tool(artifacts)}, artifacts=artifacts
    )

    result = asyncio.run(
        execute(ToolCall("c1", "read_artifact", {"digest": digest, "max_bytes": 196_608}), ctx)
    )

    assert result.truncated is False
    assert "artifact_digest" not in result.metadata  # 未再外部化
    assert result.text == payload  # 返回的是正文,不是又一张指针


def test_read_artifact_resolve_use_has_artifact_target(tmp_path: Path) -> None:
    tool = create_read_artifact_tool(ArtifactStore(tmp_path))
    use = tool.resolve_use({"digest": DIGEST_A})
    assert use.effects == frozenset({ToolEffect.READ_WORKSPACE})
    assert use.targets[0].kind == "artifact"  # 非 "path":不参与地板/路径裁决
    assert use.targets[0].value == DIGEST_A
    assert tool.static_effects == frozenset({ToolEffect.READ_WORKSPACE})
    assert tool.idempotent and tool.concurrency_safe


def test_read_artifact_not_found(tmp_path: Path) -> None:
    tool = create_read_artifact_tool(ArtifactStore(tmp_path))
    r = asyncio.run(tool.run({"digest": DIGEST_A}, ToolExecContext()))
    assert r.status is ResultStatus.ERROR and r.error_code == "artifact_not_found"
    assert r.is_error


def test_read_artifact_integrity_error(tmp_path: Path) -> None:
    a = ArtifactStore(tmp_path)
    digest = a.put(b"good")
    (tmp_path / digest[:2] / digest).write_bytes(b"tampered")
    tool = create_read_artifact_tool(a)
    r = asyncio.run(tool.run({"digest": digest}, ToolExecContext()))
    assert r.status is ResultStatus.ERROR and r.error_code == "artifact_integrity"


@pytest.mark.parametrize("bad", ["../etc/passwd", "ABC", "a" * 63, "", "a" * 65, 123])
def test_read_artifact_rejects_malformed_digest(
    store: Store, workspace: Path, tmp_path: Path, bad: object
) -> None:
    """Review Focus 4:一切畸形输入都是结果,不炸穿管道(无路径成分)。"""
    tool = create_read_artifact_tool(ArtifactStore(tmp_path / "artifacts"))
    ctx = _ctx(store, workspace, {"read_artifact": tool})

    result = asyncio.run(execute(ToolCall("c1", "read_artifact", {"digest": cast("Any", bad)}), ctx))

    assert result.status is ResultStatus.ERROR
    assert result.error_code == "invalid_input"
    assert result.is_error
    # 直接调用 run(不经管道)同样是结果而不是异常
    direct = asyncio.run(tool.run({"digest": cast("Any", bad)}, ToolExecContext()))
    assert direct.status is ResultStatus.ERROR and direct.error_code == "invalid_input"


def test_read_artifact_missing_digest_is_result_not_exception(tmp_path: Path) -> None:
    tool = create_read_artifact_tool(ArtifactStore(tmp_path))
    r = asyncio.run(tool.run({}, ToolExecContext()))
    assert r.status is ResultStatus.ERROR and r.error_code == "invalid_input"


def test_composition_registers_read_artifact(workspace: Path) -> None:
    """默认工具表包含 read_artifact(组合根装配),tools 参数仍可整体替换。"""
    from praxis.adapters.models.scripted import ScriptedAdapter
    from praxis.composition import build_session

    session = build_session(workspace, ScriptedAdapter([]))
    try:
        assert "read_artifact" in session._tools
        assert {"read", "write", "bash"}.issubset(session._tools)
        assert session._artifacts is not None
    finally:
        session.close()

    # 工具表被整体替换且没有 read_artifact → 不注入 artifacts(退回旧截断)。
    custom = build_session(workspace, ScriptedAdapter([]), tools=(create_read_tool(workspace),))
    try:
        assert custom._artifacts is None
    finally:
        custom.close()
