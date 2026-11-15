"""T8:``invoke_skill`` 普通工具(设计 §5.14;数值面逐条对照旧实现)。

钉死的行为面:

1. 工具形状:name=``invoke_skill``、schema ``{name: str(1..500), args: str|null(≤4000)}``
   (extra=forbid)、**无 enum**(目录已进上下文,刻意不复制旧 ``available_skill_ids``
   的 enum)、``static_effects=frozenset()``、``resolve_use`` target
   ``(skill, name)``、``idempotent=True``、``concurrency_safe=False``、
   COOPERATIVE/CANCEL、``timeout_seconds=5.0``、``max_model_output_bytes=300_000``、
   source=BUILTIN;
2. 四个错误码:``skill_not_found``(未知名/激活时文件消失)、``skill_disabled``、
   ``skill_content_changed``(重读重算指纹漂移)、
   ``unsupported_skill_constraints``(激活时 frontmatter 出现非空约束——loader
   已把它排除出目录,故只能经漂移到达,这正是 R-T8-2);
3. 指纹重算 procedure(R-T8-1):重读 ``manifest.skill_file``、用 loader 的
   parse helpers 解析、置 ``frontmatter["name"] = manifest.name``、再调
   ``skill_fingerprint``;命名空间 skill(``<ns>:<name>``)必须 round-trip 相等,
   否则会假报 ``skill_content_changed``;
4. 替换语义(旧 ``skills/runtime.py:90-93`` 逐字):``${SKILL_DIR}`` 字面
   replace、``$SKILL_DIR`` 正则 word-boundary + lambda、``args is not None``
   才替换 ``$ARGUMENTS``;args 里的反斜杠/反引用序列绝不被解释;
5. 正文 > ``MAX_SKILL_INSTRUCTIONS_CHARS=40_000`` 截断并标 ``truncated``
   (content text 仍以 ``skill activated: <name>`` 开头);
6. ``metadata`` 恰为 ``{skill_id, source, fingerprint, truncated}``,经真实
   ``pipeline.execute`` 落库后 tool_result item payload 仍可投影 ``skill_id``
   (T9 的激活态判定的唯一事实源)。

仓库无 pytest-asyncio:asyncio.run 包装,同 test_tools_planning.py 风格。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from praxis.core.events import EventBus
from praxis.core.pipeline import ExecCtx, execute
from praxis.protocol.items import ItemKind
from praxis.protocol.permission import PermissionContext
from praxis.protocol.policy import PermissionPolicy
from praxis.protocol.sandbox import ToolExecContext
from praxis.protocol.skills import (
    MAX_INVOKE_ARGS_CHARS,
    MAX_SKILL_INSTRUCTIONS_CHARS,
    MAX_SKILL_NAME_CHARS,
    SkillCatalog,
)
from praxis.protocol.tools import (
    CancellationMode,
    InterruptBehavior,
    ResultStatus,
    Tool,
    ToolCall,
    ToolSource,
    ToolTarget,
)
from praxis.store.log import Store
from praxis.store.turns import create_turn
from praxis.tools.builtins.skills import create_invoke_skill_tool, load_skill_catalog

_HEADER = "skill activated: "


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
    body: str,
    extra_frontmatter: str = "",
    namespace: str | None = None,
) -> Path:
    """在真实 loader 目录形状写一个 SKILL.md;返回文件路径(便于漂移测试改写)。"""
    (workspace / ".git").mkdir(exist_ok=True)  # 锚定 repo root,PROJECT walk 只扫本 workspace
    skills_root = workspace / ".agents" / "skills"
    skill_dir = skills_root / namespace / name if namespace is not None else skills_root / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        f"---\nname: {name}\ndescription: test skill\n{extra_frontmatter}---\n{body}",
        encoding="utf-8",
    )
    return skill_file


def _catalog(workspace: Path) -> SkillCatalog:
    return load_skill_catalog(workspace)


def _invoke(tool: Tool, arguments: dict[str, Any]) -> Any:
    """裸 run 路径(不经权限/管道)的投影。"""
    return asyncio.run(tool.run(arguments, ToolExecContext()))


def _exec_ctx(store: Store, workspace: Path, tool: Tool) -> ExecCtx:
    """裸 ExecCtx(同 test_tools_planning.py):验证 metadata 真的落进 item。"""
    policy = PermissionPolicy()
    turn = create_turn(store, turn_id="turn-skills", thread_id="thread-skills", binding={})
    return ExecCtx(
        store=store,
        bus=EventBus(),
        workspace=workspace,
        tools={tool.definition.name: tool},
        permissions=PermissionContext(
            workspace=str(workspace.resolve()),
            policy=policy,
            mode="on-request",
        ),
        turn_id=turn.turn_id,
        thread_id="thread-skills",
        policy_revision=policy.revision,
    )


# ---------------------------------------------------------------------------
# 工具形状
# ---------------------------------------------------------------------------


def test_tool_shape_matches_legacy_numbers(workspace: Path) -> None:
    _write_skill(workspace, "shape", body="\nbody\n")
    tool = create_invoke_skill_tool(_catalog(workspace))

    assert tool.definition.name == "invoke_skill"
    assert tool.source is ToolSource.BUILTIN
    assert tool.static_effects == frozenset()
    assert tool.idempotent is True
    assert tool.concurrency_safe is False
    assert tool.cancellation_mode is CancellationMode.COOPERATIVE
    assert tool.interrupt_behavior is InterruptBehavior.CANCEL
    assert tool.timeout_seconds == 5.0
    # 旧 integrations/skills.py:155 max_model_output_bytes=300_000
    assert tool.max_model_output_bytes == 300_000

    schema = tool.definition.input_schema
    assert schema["additionalProperties"] is False
    assert tuple(schema["required"]) == ("name",)  # 冻结 JSON:tuple 化
    properties = schema["properties"]
    assert properties["name"]["minLength"] == 1
    assert properties["name"]["maxLength"] == MAX_SKILL_NAME_CHARS  # 500(旧 file:44-46)
    assert properties["args"]["maxLength"] == MAX_INVOKE_ARGS_CHARS  # 4000(旧 file:47-51)
    assert set(properties["args"]["type"]) == {"string", "null"}

    use = tool.resolve_use({"name": "shape"})
    assert use.effects == frozenset()
    assert use.targets == (ToolTarget(kind="skill", value="shape"),)


def test_schema_has_no_enum_for_name(workspace: Path) -> None:
    """有意差异:目录已进上下文;旧 available_skill_ids enum 不复制。"""
    _write_skill(workspace, "noenum", body="\nbody\n")
    tool = create_invoke_skill_tool(_catalog(workspace))
    assert "enum" not in tool.definition.input_schema["properties"]["name"]


# ---------------------------------------------------------------------------
# 正常激活:元数据与首行
# ---------------------------------------------------------------------------


def test_success_metadata_exact_keys_and_header(workspace: Path) -> None:
    _write_skill(workspace, "hello", body="\n# Hello\n\nDo the thing.\n")
    catalog = _catalog(workspace)
    manifest = catalog.get("hello")
    assert manifest is not None

    result = _invoke(create_invoke_skill_tool(catalog), {"name": "hello"})

    assert result.status is ResultStatus.OK and result.is_error is False
    assert result.text == f"{_HEADER}hello\n\n# Hello\n\nDo the thing.\n"
    # 恰好这四个键(JSON-safe 字符串/布尔);T9 从 skill_id 投影激活态。
    assert dict(result.metadata) == {
        "skill_id": "hello",
        "source": manifest.source.value,
        "fingerprint": manifest.fingerprint,
        "truncated": False,
    }
    assert result.truncated is False


def test_metadata_lands_in_tool_result_item_payload(workspace: Path, store: Store) -> None:
    _write_skill(workspace, "piped", body="\ninstructions\n")
    catalog = _catalog(workspace)
    tool = create_invoke_skill_tool(catalog)
    ctx = _exec_ctx(store, workspace, tool)

    result = asyncio.run(execute(ToolCall("c1", "invoke_skill", {"name": "piped"}), ctx))

    assert result.status is ResultStatus.OK
    assert dict(result.metadata) == {
        "skill_id": "piped",
        "source": "project",
        "fingerprint": catalog.get("piped").fingerprint,  # type: ignore[union-attr]
        "truncated": False,
    }
    items = store.read_turn_items(ctx.turn_id)
    assert [item.kind for item in items] == [ItemKind.TOOL_CALL, ItemKind.TOOL_RESULT]
    payload = items[-1].payload
    assert payload["status"] == "ok"
    assert payload["metadata"]["skill_id"] == "piped"
    assert payload["metadata"]["fingerprint"] == result.metadata["fingerprint"]


# ---------------------------------------------------------------------------
# 替换语义(旧 skills/runtime.py:90-93 逐字)
# ---------------------------------------------------------------------------


def test_substitutes_both_skill_dir_forms_and_arguments(workspace: Path) -> None:
    body = "\nDir literal ${SKILL_DIR}; dir regex $SKILL_DIR; args: $ARGUMENTS\n"
    _write_skill(workspace, "subst", body=body)
    catalog = _catalog(workspace)
    manifest = catalog.get("subst")
    assert manifest is not None

    result = _invoke(create_invoke_skill_tool(catalog), {"name": "subst", "args": "alpha beta"})

    expected_body = (
        f"\nDir literal {manifest.root_dir}; dir regex {manifest.root_dir}; args: alpha beta\n"
    )
    assert result.text == f"{_HEADER}subst\n{expected_body}"
    # 两种写法各替换一次,绝无二次解释(根目录只出现两次)。
    assert result.text.count(manifest.root_dir) == 2


def test_arguments_regex_special_chars_are_literal(workspace: Path) -> None:
    """lambda 替换 ⇒ 反斜杠/反引用序列按字面数据落地(旧 file:87-88 注释)。"""
    _write_skill(workspace, "dots", body="\nargs=[$ARGUMENTS]\n")
    catalog = _catalog(workspace)
    args = r"C:\new\1 \g<0> $1 \1"

    result = _invoke(create_invoke_skill_tool(catalog), {"name": "dots", "args": args})

    assert result.text == f"{_HEADER}dots\n\nargs=[{args}]\n"


def test_arguments_placeholder_untouched_when_args_absent(workspace: Path) -> None:
    """旧实现只在 ``args is not None`` 时替换;None 保持字面占位符。"""
    _write_skill(workspace, "noargs", body="\nargs=[$ARGUMENTS]\n")
    catalog = _catalog(workspace)

    result = _invoke(create_invoke_skill_tool(catalog), {"name": "noargs"})

    assert result.text == f"{_HEADER}noargs\n\nargs=[$ARGUMENTS]\n"


# ---------------------------------------------------------------------------
# 截断
# ---------------------------------------------------------------------------


def test_long_body_is_truncated_at_40k(workspace: Path) -> None:
    body = "A" * (MAX_SKILL_INSTRUCTIONS_CHARS + 500)
    _write_skill(workspace, "long", body=body)
    catalog = _catalog(workspace)

    result = _invoke(create_invoke_skill_tool(catalog), {"name": "long"})

    assert result.status is ResultStatus.OK
    assert result.truncated is True
    assert result.metadata["truncated"] is True
    prefix = f"{_HEADER}long\n"
    assert result.text.startswith(prefix)
    instructions = result.text[len(prefix):]
    assert len(instructions) == MAX_SKILL_INSTRUCTIONS_CHARS  # 40_000
    assert instructions == body[:MAX_SKILL_INSTRUCTIONS_CHARS]


# ---------------------------------------------------------------------------
# 错误码
# ---------------------------------------------------------------------------


def test_unknown_name_is_skill_not_found(workspace: Path) -> None:
    _write_skill(workspace, "known", body="\nbody\n")
    result = _invoke(create_invoke_skill_tool(_catalog(workspace)), {"name": "nope"})

    assert result.status is ResultStatus.ERROR
    assert result.is_error is True
    assert result.error_code == "skill_not_found"
    assert "nope" not in (result.error_message or "")  # 固定有界消息,不回显任意名字


def test_disabled_skill_is_skill_disabled(workspace: Path) -> None:
    _write_skill(
        workspace,
        "manual",
        body="\nbody\n",
        extra_frontmatter="disable_model_invocation: true\n",
    )
    catalog = _catalog(workspace)
    assert catalog.get("manual") is not None  # loader 保留,invoke 才拒绝

    result = _invoke(create_invoke_skill_tool(catalog), {"name": "manual"})

    assert result.error_code == "skill_disabled"
    assert result.status is ResultStatus.ERROR


def test_fingerprint_drift_is_skill_content_changed(workspace: Path) -> None:
    skill_file = _write_skill(workspace, "drift", body="\noriginal body\n")
    tool = create_invoke_skill_tool(_catalog(workspace))
    skill_file.write_text(
        skill_file.read_text(encoding="utf-8") + "\nedited after load\n",
        encoding="utf-8",
    )

    result = _invoke(tool, {"name": "drift"})

    assert result.error_code == "skill_content_changed"
    assert result.status is ResultStatus.ERROR


def test_drift_to_constraints_is_unsupported(workspace: Path) -> None:
    """R-T8-2:loader 已排除带约束的 skill;该错误码只能经激活时漂移到达。"""
    skill_file = _write_skill(workspace, "constrain", body="\nbody\n")
    tool = create_invoke_skill_tool(_catalog(workspace))
    skill_file.write_text(
        "---\nname: constrain\ndescription: test skill\n"
        "allowed_tools:\n  - bash\n---\n\nbody\n",
        encoding="utf-8",
    )

    result = _invoke(tool, {"name": "constrain"})

    assert result.error_code == "unsupported_skill_constraints"
    assert result.status is ResultStatus.ERROR


def test_constrained_skill_is_invisible_in_catalog(workspace: Path) -> None:
    """loader 一处 skip ⇒ 目录里没有;正常调用是 ``skill_not_found``(R-T8-2)。"""
    _write_skill(
        workspace,
        "hidden",
        body="\nbody\n",
        extra_frontmatter="paths:\n  - src/**\n",
    )
    catalog = _catalog(workspace)
    assert catalog.get("hidden") is None

    result = _invoke(create_invoke_skill_tool(catalog), {"name": "hidden"})

    assert result.error_code == "skill_not_found"


def test_missing_file_at_activation_is_skill_not_found(workspace: Path) -> None:
    skill_file = _write_skill(workspace, "gone", body="\nbody\n")
    tool = create_invoke_skill_tool(_catalog(workspace))
    skill_file.unlink()

    result = _invoke(tool, {"name": "gone"})

    assert result.error_code == "skill_not_found"
    assert result.status is ResultStatus.ERROR


def test_malformed_rewrite_is_skill_content_changed(workspace: Path) -> None:
    skill_file = _write_skill(workspace, "syntax", body="\nbody\n")
    tool = create_invoke_skill_tool(_catalog(workspace))
    skill_file.write_text("no frontmatter at all\n", encoding="utf-8")

    result = _invoke(tool, {"name": "syntax"})

    assert result.error_code == "skill_content_changed"


def test_oversized_name_via_pipeline_is_invalid_input(workspace: Path, store: Store) -> None:
    _write_skill(workspace, "short", body="\nbody\n")
    tool = create_invoke_skill_tool(_catalog(workspace))
    ctx = _exec_ctx(store, workspace, tool)

    result = asyncio.run(execute(ToolCall("c1", "invoke_skill", {"name": "x" * 2000}), ctx))

    assert result.status is ResultStatus.ERROR
    assert result.error_code == "invalid_input"
    assert result.is_error is True


# ---------------------------------------------------------------------------
# R-T8-1:命名空间 skill 的指纹 round-trip
# ---------------------------------------------------------------------------


def test_namespaced_skill_recompute_roundtrips(workspace: Path) -> None:
    """manifest.name 已含 ``<ns>:`` 前缀;重算必须置回该名,不可用原始解析。"""
    _write_skill(workspace, "nested", body="\nbranch workflow\n", namespace="team")
    catalog = _catalog(workspace)
    manifest = catalog.get("team:nested")
    assert manifest is not None
    assert manifest.name == "team:nested"

    result = _invoke(create_invoke_skill_tool(catalog), {"name": "team:nested"})

    assert result.status is ResultStatus.OK
    assert result.text == f"{_HEADER}team:nested\n\nbranch workflow\n"
    assert result.metadata["skill_id"] == "team:nested"
    assert result.metadata["fingerprint"] == manifest.fingerprint