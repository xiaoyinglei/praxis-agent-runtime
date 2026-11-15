"""T9:``materialize_skill_asset`` 与激活态投影(设计 §5.14;Q8)。

钉死的行为面:

1. 工具形状:name=``materialize_skill_asset``、schema
   ``{skill_id: str(1..500), relative_path: str(1..4096)}``(extra=forbid)、
   ``static_effects={READ_WORKSPACE, WRITE_WORKSPACE}``、``resolve_use`` 目标
   ``(skill:<skill_id>, path:<dest>)``、``idempotent=True``、
   ``concurrency_safe=False``、COOPERATIVE/CANCEL、``timeout_seconds=5.0``、
   ``max_model_output_bytes=50_000``、source=BUILTIN;
2. 唯一权限耦合:``resolve_use`` 照常交 ``decide``;``run`` 检查
   ``exec_ctx.active_skill_ids``——未激活 ⇒ ``ERROR / skill_not_active``。
   目标自身携带 ``skill`` 不构成激活证明(激活只来自本回合**已提交**的
   ``invoke_skill`` 成功结果);
3. 激活后 roundtrip:字节逐字一致、``source_fingerprint`` = sha256(字节)、
   ``size_bytes`` 正确;
4. 路径校验:绝对、``..``、首段不在 ``{scripts, references}``、只有一段 ⇒
   ``$.relative_path`` 的 invalid_input;源越界(含符号链接逃逸)与目标越界
   (``checked_path`` 双候选)分别拒绝;
5. Q8 落点 ``<workspace>/.scratch/skills/<safe_id>/<rel>``(不是 ``.praxis``
   ——那是地板),``.scratch`` 不是 FLOOR_ENTRIES 成员且 read 工具可读;
   工具描述明确 gitignore 负担;
6. ``latest_active_skill_ids`` 只收集成功 ``invoke_skill`` 的字符串
   ``metadata["skill_id"]``,畸形输入 fail-open;item 是 turn 局部的,新 turn
   从空集开始(旧 ``_context_for_turn`` 语义)。

仓库无 pytest-asyncio:asyncio.run 包装,同 test_skills_invoke.py 风格。
"""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import Any

import pytest

from praxis.core.events import EventBus
from praxis.core.pipeline import ExecCtx, execute
from praxis.core.skill_state import latest_active_skill_ids
from praxis.protocol.floor import FLOOR_ENTRIES, floor_reason
from praxis.protocol.items import Item, ItemKind
from praxis.protocol.permission import PermissionContext
from praxis.protocol.policy import PermissionPolicy
from praxis.protocol.sandbox import ToolExecContext
from praxis.protocol.tools import (
    CancellationMode,
    InterruptBehavior,
    ResultStatus,
    Tool,
    ToolCall,
    ToolEffect,
    ToolSource,
    ToolTarget,
    ToolValidationError,
)
from praxis.store.log import Store
from praxis.store.turns import create_turn
from praxis.tools.builtins.filesystem import create_read_tool
from praxis.tools.builtins.skills import (
    create_invoke_skill_tool,
    create_materialize_skill_asset_tool,
    load_skill_catalog,
)

_MATERIALIZE = "materialize_skill_asset"


# ---------------------------------------------------------------------------
# 脚手架
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_external_skill_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """目录只来自 workspace 的 ``.agents/skills``;外部根不参与本文件断言。"""
    monkeypatch.delenv("SKILL_PATH", raising=False)


def _write_skill(
    workspace: Path,
    name: str,
    *,
    body: str = "\nbody\n",
    namespace: str | None = None,
) -> Path:
    """写一个真实 loader 形状的 skill,返回 skill 目录(资产放它下面)。"""
    (workspace / ".git").mkdir(exist_ok=True)  # 锚定 repo root,PROJECT walk 只扫本 workspace
    skills_root = workspace / ".agents" / "skills"
    skill_dir = skills_root / namespace / name if namespace is not None else skills_root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: test skill\n---\n{body}",
        encoding="utf-8",
    )
    return skill_dir


def _write_asset(skill_dir: Path, relative: str, data: bytes) -> Path:
    path = skill_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _exec_ctx(
    store: Store,
    workspace: Path,
    tools: tuple[Tool, ...],
    *,
    turn_id: str = "turn-materialize",
) -> ExecCtx:
    policy = PermissionPolicy()
    turn = create_turn(store, turn_id=turn_id, thread_id="thread-materialize", binding={})
    return ExecCtx(
        store=store,
        bus=EventBus(),
        workspace=workspace,
        tools={tool.definition.name: tool for tool in tools},
        permissions=PermissionContext(
            workspace=str(workspace.resolve()),
            policy=policy,
            mode="on-request",
        ),
        turn_id=turn.turn_id,
        thread_id="thread-materialize",
        policy_revision=policy.revision,
    )


def _run(ctx: ExecCtx, tool_name: str, arguments: dict[str, Any], call_id: str = "c1") -> Any:
    return asyncio.run(execute(ToolCall(call_id, tool_name, arguments), ctx))


def _skill_tools(workspace: Path) -> tuple[Tool, Tool]:
    return (
        create_invoke_skill_tool(load_skill_catalog(workspace)),
        create_materialize_skill_asset_tool(workspace),
    )


def _result_payload(store: Store, turn_id: str) -> dict[str, Any]:
    items = store.read_turn_items(turn_id)
    assert items[-1].kind == ItemKind.TOOL_RESULT
    return dict(items[-1].payload)


# ---------------------------------------------------------------------------
# 工具形状
# ---------------------------------------------------------------------------


def test_tool_shape_matches_legacy_numbers(workspace: Path) -> None:
    _write_skill(workspace, "shape")
    tool = create_materialize_skill_asset_tool(workspace)

    assert tool.definition.name == _MATERIALIZE
    assert tool.source is ToolSource.BUILTIN
    assert tool.static_effects == frozenset(
        {ToolEffect.READ_WORKSPACE, ToolEffect.WRITE_WORKSPACE}
    )
    assert tool.idempotent is True
    assert tool.concurrency_safe is False
    assert tool.cancellation_mode is CancellationMode.COOPERATIVE
    assert tool.interrupt_behavior is InterruptBehavior.CANCEL
    # 旧 integrations/skills.py:209-210 逐字
    assert tool.timeout_seconds == 5.0
    assert tool.max_model_output_bytes == 50_000

    schema = tool.definition.input_schema
    assert schema["additionalProperties"] is False
    assert tuple(schema["required"]) == ("skill_id", "relative_path")
    properties = schema["properties"]
    assert properties["skill_id"]["minLength"] == 1
    assert properties["skill_id"]["maxLength"] == 500  # 旧 file:73-77
    assert properties["relative_path"]["minLength"] == 1
    assert properties["relative_path"]["maxLength"] == 4096  # 旧 file:78-82

    use = tool.resolve_use({"skill_id": "shape", "relative_path": "scripts/run.sh"})
    assert use.effects == frozenset(
        {ToolEffect.READ_WORKSPACE, ToolEffect.WRITE_WORKSPACE}
    )
    expected = (
        workspace.resolve() / ".scratch" / "skills" / "shape" / "scripts" / "run.sh"
    )
    assert use.targets == (
        ToolTarget(kind="skill", value="shape"),
        ToolTarget(kind="path", value=str(expected)),
    )

    # Q8:落点与 gitignore 负担必须写进模型可见描述。
    description = tool.definition.description
    assert ".scratch/skills/" in description
    assert "gitignore" in description.casefold()
    assert ".praxis" in description


def test_safe_id_formula_and_gitignore_note(workspace: Path) -> None:
    """命名空间 skill 的 ``ns:name`` ⇒ ``ns_name``;safe_id 公式 Q8 逐字。"""
    _write_skill(workspace, "round", namespace="ns")
    tool = create_materialize_skill_asset_tool(workspace)
    use = tool.resolve_use({"skill_id": "ns:round", "relative_path": "scripts/a.sh"})
    assert "ns_round" in use.targets[1].value
    assert ":" not in use.targets[1].value


# ---------------------------------------------------------------------------
# 激活守卫(唯一权限耦合)
# ---------------------------------------------------------------------------


def test_inactive_skill_is_skill_not_active_direct(workspace: Path) -> None:
    _write_skill(workspace, "ghost")
    tool = create_materialize_skill_asset_tool(workspace)

    result = asyncio.run(
        tool.run(
            {"skill_id": "ghost", "relative_path": "scripts/a.sh"},
            ToolExecContext(),  # 无激活:默认空集
        )
    )

    assert result.status is ResultStatus.ERROR
    assert result.is_error is True
    assert result.error_code == "skill_not_active"
    assert "ghost" in (result.error_message or "")


def test_forged_skill_target_does_not_grant_activation_via_pipeline(
    workspace: Path, store: Store
) -> None:
    """resolve_use 的 ``skill:<id>`` 目标不是激活证明:run 仍返回 skill_not_active。"""
    _write_skill(workspace, "ghost")
    tool = create_materialize_skill_asset_tool(workspace)
    ctx = _exec_ctx(store, workspace, (tool,))

    result = _run(
        ctx, _MATERIALIZE, {"skill_id": "ghost", "relative_path": "scripts/a.sh"}
    )

    assert result.status is ResultStatus.ERROR
    assert result.error_code == "skill_not_active"

    items = store.read_turn_items(ctx.turn_id)
    # 调用项已落库(decide 放行了无激活也无权裁定的工具),目标里确有 skill:ghost。
    assert [item.kind for item in items] == [ItemKind.TOOL_CALL, ItemKind.TOOL_RESULT]
    call_targets = [dict(target) for target in items[0].payload["targets"]]
    assert {"kind": "skill", "value": "ghost"} in call_targets

    payload = _result_payload(store, ctx.turn_id)
    assert payload["status"] == "error"
    assert payload["is_error"] is True
    assert payload["error_code"] == "skill_not_active"


def test_activated_id_outside_catalog_is_skill_not_found(workspace: Path) -> None:
    """激活事实存在但目录快照里没有该 id(异常装配路径)⇒ 明确的错误,不是 tool_error。"""
    tool = create_materialize_skill_asset_tool(workspace)  # 空目录:无任何 skill

    result = asyncio.run(
        tool.run(
            {"skill_id": "ghost", "relative_path": "scripts/a.sh"},
            ToolExecContext(active_skill_ids=frozenset({"ghost"})),
        )
    )

    assert result.status is ResultStatus.ERROR
    assert result.error_code == "skill_not_found"


def test_missing_asset_is_asset_not_found(workspace: Path) -> None:
    _write_skill(workspace, "empty")
    tool = create_materialize_skill_asset_tool(workspace)

    result = asyncio.run(
        tool.run(
            {"skill_id": "empty", "relative_path": "scripts/missing.sh"},
            ToolExecContext(active_skill_ids=frozenset({"empty"})),
        )
    )

    assert result.status is ResultStatus.ERROR
    assert result.error_code == "asset_not_found"
    assert not (workspace / ".scratch").exists()


# ---------------------------------------------------------------------------
# 激活后 roundtrip(Q8 落点 / 字节 / 指纹)
# ---------------------------------------------------------------------------


def test_activated_roundtrip_bytes_metadata_and_scratch_landing(
    workspace: Path, store: Store
) -> None:
    skill_dir = _write_skill(workspace, "round")
    data = b"\x00\x01binary payload \xff\n"
    _write_asset(skill_dir, "references/blob.bin", data)
    invoke, materialize = _skill_tools(workspace)
    ctx = _exec_ctx(store, workspace, (invoke, materialize))

    activated = _run(ctx, "invoke_skill", {"name": "round"}, call_id="c-invoke")
    assert activated.status is ResultStatus.OK

    result = _run(
        ctx,
        _MATERIALIZE,
        {"skill_id": "round", "relative_path": "references/blob.bin"},
        call_id="c-materialize",
    )

    assert result.status is ResultStatus.OK and result.is_error is False
    expected_path = ".scratch/skills/round/references/blob.bin"
    assert dict(result.metadata) == {
        "workspace_path": expected_path,
        "source_fingerprint": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
    }
    destination = workspace / ".scratch" / "skills" / "round" / "references" / "blob.bin"
    assert destination.read_bytes() == data

    # 事实经管道落库后仍可投影(T9 的唯一事实源)。
    payload = _result_payload(store, ctx.turn_id)
    assert payload["status"] == "ok"
    assert payload["metadata"]["workspace_path"] == expected_path
    assert payload["metadata"]["source_fingerprint"] == hashlib.sha256(data).hexdigest()
    assert payload["metadata"]["size_bytes"] == len(data)


def test_scratch_is_not_floor_and_read_tool_can_consume_it(
    workspace: Path, store: Store
) -> None:
    skill_dir = _write_skill(workspace, "scratchy")
    _write_asset(skill_dir, "references/notes.md", b"hello from skill asset\n")
    invoke, materialize = _skill_tools(workspace)
    ctx = _exec_ctx(store, workspace, (invoke, materialize))

    assert _run(ctx, "invoke_skill", {"name": "scratchy"}).status is ResultStatus.OK
    result = _run(
        ctx,
        _MATERIALIZE,
        {"skill_id": "scratchy", "relative_path": "references/notes.md"},
        call_id="c-materialize",
    )
    assert result.status is ResultStatus.OK

    destination = (
        workspace / ".scratch" / "skills" / "scratchy" / "references" / "notes.md"
    )
    assert ".scratch" not in {entry.name for entry in FLOOR_ENTRIES}
    assert floor_reason(destination, write=True, private_paths=()) == ""

    # 工作区内的普通目录:read 工具(同一 checked_path 围栏)可以直接消费。
    read = asyncio.run(
        create_read_tool(workspace).run(
            {"path": ".scratch/skills/scratchy/references/notes.md"}, ToolExecContext()
        )
    )
    assert read.status is ResultStatus.OK
    assert "hello from skill asset" in read.text


# ---------------------------------------------------------------------------
# 路径校验
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_path",
    [
        "scripts/../etc/passwd",  # ...
        "/etc/passwd",  # 绝对
        "notes/readme.md",  # 首段不在 {scripts, references}
        "scripts",  # 只有一段(无文件名)
        "references",  # 只有一段
        ".",
        "",
    ],
)
def test_invalid_relative_paths_are_rejected(workspace: Path, bad_path: str) -> None:
    tool = create_materialize_skill_asset_tool(workspace)
    with pytest.raises(ToolValidationError) as excinfo:
        tool.validate_input({"skill_id": "any", "relative_path": bad_path})
    assert excinfo.value.path == "$.relative_path"


def test_invalid_relative_path_via_pipeline_is_invalid_input(
    workspace: Path, store: Store
) -> None:
    tool = create_materialize_skill_asset_tool(workspace)
    ctx = _exec_ctx(store, workspace, (tool,))

    result = _run(
        ctx, _MATERIALIZE, {"skill_id": "any", "relative_path": "notes/readme.md"}
    )

    assert result.status is ResultStatus.ERROR
    assert result.error_code == "invalid_input"
    payload = _result_payload(store, ctx.turn_id)
    assert payload["error_code"] == "invalid_input"
    assert "$.relative_path" in (payload["error_message"] or "")


# ---------------------------------------------------------------------------
# 符号链接:目标按 checked_path 双候选;源逃逸一律拒绝
# ---------------------------------------------------------------------------


def test_destination_symlink_inside_workspace_is_followed(
    workspace: Path, store: Store
) -> None:
    """``.scratch/skills`` 指向工作区内的真实目录 ⇒ checked_path 解析后仍在内,允许。"""
    skill_dir = _write_skill(workspace, "linked")
    data = b"#!/bin/sh\necho linked\n"
    _write_asset(skill_dir, "scripts/tool.sh", data)

    real = workspace / "real-scratch"
    real.mkdir()
    (workspace / ".scratch").mkdir()
    (workspace / ".scratch" / "skills").symlink_to(real, target_is_directory=True)

    invoke, materialize = _skill_tools(workspace)
    ctx = _exec_ctx(store, workspace, (invoke, materialize))
    assert _run(ctx, "invoke_skill", {"name": "linked"}).status is ResultStatus.OK

    result = _run(
        ctx,
        _MATERIALIZE,
        {"skill_id": "linked", "relative_path": "scripts/tool.sh"},
        call_id="c-materialize",
    )

    assert result.status is ResultStatus.OK
    assert (
        real / "linked" / "scripts" / "tool.sh"
    ).read_bytes() == data
    # 返回给模型的是词法落点(经同一符号链接可读回同一字节)。
    assert result.metadata["workspace_path"] == ".scratch/skills/linked/scripts/tool.sh"


def test_destination_symlink_escaping_workspace_is_invalid_input(
    workspace: Path, store: Store, tmp_path: Path
) -> None:
    _write_skill(workspace, "escape")
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / ".scratch").symlink_to(outside, target_is_directory=True)

    tool = create_materialize_skill_asset_tool(workspace)
    ctx = _exec_ctx(store, workspace, (tool,))

    result = _run(
        ctx, _MATERIALIZE, {"skill_id": "escape", "relative_path": "scripts/a.sh"}
    )

    assert result.status is ResultStatus.ERROR
    assert result.error_code == "invalid_input"
    assert not (outside / "skills").exists()


def test_source_symlink_outside_skill_root_is_rejected(
    workspace: Path, store: Store, tmp_path: Path
) -> None:
    skill_dir = _write_skill(workspace, "leaky")
    secret = tmp_path / "secret.txt"
    secret.write_bytes(b"top secret\n")
    (skill_dir / "scripts").mkdir(parents=True, exist_ok=True)
    (skill_dir / "scripts" / "leak.sh").symlink_to(secret)

    invoke, materialize = _skill_tools(workspace)
    ctx = _exec_ctx(store, workspace, (invoke, materialize))
    assert _run(ctx, "invoke_skill", {"name": "leaky"}).status is ResultStatus.OK

    result = _run(
        ctx,
        _MATERIALIZE,
        {"skill_id": "leaky", "relative_path": "scripts/leak.sh"},
        call_id="c-materialize",
    )

    # 源围栏 + O_NOFOLLOW:不读技能根之外的字节,也不落任何目标。
    assert result.status is ResultStatus.ERROR
    assert result.error_code == "asset_not_found"
    assert not (workspace / ".scratch").exists()


# ---------------------------------------------------------------------------
# 激活态投影:turn 局部、fail-open
# ---------------------------------------------------------------------------


def _result_item(
    seq: int,
    *,
    tool_name: str = "invoke_skill",
    status: str = "ok",
    is_error: bool = False,
    metadata: object = None,
) -> Item:
    payload: dict[str, Any] = {
        "tool_name": tool_name,
        "status": status,
        "is_error": is_error,
    }
    if metadata is not None:
        payload["metadata"] = metadata
    return Item(
        item_id=f"item-{seq}",
        thread_id="thread",
        turn_id="turn",
        seq=seq,
        kind="tool_result",
        payload=payload,
        created_at=0.0,
    )


def test_latest_active_skill_ids_only_collects_successful_invoke_metadata() -> None:
    items = [
        _result_item(1, metadata={"skill_id": "alpha"}),
        _result_item(2, status="error", is_error=True, metadata={"skill_id": "beta"}),
        _result_item(3, metadata={"skill_id": 7}),  # 非字符串
        _result_item(4, metadata="not-a-mapping"),  # 畸形 fail-open
        _result_item(5, tool_name=_MATERIALIZE, metadata={"skill_id": "gamma"}),
        _result_item(6),  # 无 metadata
        _result_item(7, metadata={"skill_id": ""}),  # 空串
        Item(
            item_id="item-8",
            thread_id="thread",
            turn_id="turn",
            seq=8,
            kind="model_response",
            payload={"metadata": {"skill_id": "delta"}},
            created_at=0.0,
        ),
    ]

    assert latest_active_skill_ids(items) == frozenset({"alpha"})
    assert latest_active_skill_ids([]) == frozenset()


def test_activation_does_not_leak_across_turns(workspace: Path, store: Store) -> None:
    """item 是 turn 局部的:上一回合的激活不复活(旧 ``_context_for_turn`` 语义)。"""
    skill_dir = _write_skill(workspace, "leaky")
    _write_asset(skill_dir, "scripts/a.sh", b"asset\n")
    invoke, materialize = _skill_tools(workspace)

    ctx_a = _exec_ctx(store, workspace, (invoke, materialize), turn_id="turn-a")
    assert _run(ctx_a, "invoke_skill", {"name": "leaky"}).status is ResultStatus.OK
    assert latest_active_skill_ids(store.read_turn_items(ctx_a.turn_id)) == frozenset(
        {"leaky"}
    )

    ctx_b = _exec_ctx(store, workspace, (invoke, materialize), turn_id="turn-b")
    assert latest_active_skill_ids(store.read_turn_items(ctx_b.turn_id)) == frozenset()

    result = _run(
        ctx_b,
        _MATERIALIZE,
        {"skill_id": "leaky", "relative_path": "scripts/a.sh"},
        call_id="c-materialize",
    )
    assert result.status is ResultStatus.ERROR
    assert result.error_code == "skill_not_active"
    assert not (workspace / ".scratch").exists()