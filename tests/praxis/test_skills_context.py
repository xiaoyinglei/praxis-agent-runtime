"""T7:skill 目录进上下文构建的纯函数步骤(设计 §5.14;R-T7-1 唯一注入点)。

钉死:
1. ``append_skill_catalog`` 纯函数、幂等、永不抛;空目录 = 恒等(逐条同对象);
2. 插入位置:第一条 system 消息**之后**(无 system 时插到最前);
3. ``disable_model_invocation=True`` 不进列表(全禁用 = 无可列 → 恒等);
4. description 超 ``MAX_SKILL_DESCRIPTION_CHARS`` → 截断并带 ``…`` 标记;
5. 总预算 ``MAX_CATALOG_LISTING_CHARS``:只裁**整条**尾部条目,并以
   ``[… more skills elided]`` 收尾;
6. 端到端:目录消息真的进入 ``build_request`` 的 wire body;
7. 压缩判定看到**同一份** enriched messages(阈值无盲区):目录把请求推过
   阈值即触发压缩,若 measure 只看未注入的投影则不会触发。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from praxis.adapters.models.scripted import ScriptedAdapter
from praxis.adapters.models.tokenization import measure_request
from praxis.core.context import append_skill_catalog, build_context, to_messages
from praxis.core.events import EventBus
from praxis.core.statemachine import TurnStatus
from praxis.core.turn import TurnDeps, TurnState, run_turn
from praxis.protocol.items import ItemKind
from praxis.protocol.models import (
    ModelMessage,
    ModelResponse,
    ModelSettings,
    Usage,
    WireRequest,
)
from praxis.protocol.permission import PermissionContext
from praxis.protocol.policy import PermissionPolicy
from praxis.protocol.skills import (
    MAX_CATALOG_LISTING_CHARS,
    MAX_SKILL_DESCRIPTION_CHARS,
    SkillCatalog,
    SkillManifest,
    SkillSource,
)
from praxis.store.log import ItemEntry, Store
from praxis.store.turns import create_turn

THREAD = "thread-1"
TURN = "turn-1"
SYSTEM_PROMPT = "test-agent system prompt"
_HEADER = "Available skills (invoke with invoke_skill, name exactly as listed):"
_ELISION = "[… more skills elided]"


# ---------------------------------------------------------------------------
# 脚手架
# ---------------------------------------------------------------------------


def _manifest(
    name: str,
    description: str = "does something useful",
    *,
    disabled: bool = False,
    source: SkillSource = SkillSource.PROJECT,
) -> SkillManifest:
    root = f"/praxis-test-skills/{name}"
    return SkillManifest(
        name=name,
        description=description,
        when_to_use=None,
        version=None,
        skill_file=f"{root}/SKILL.md",
        root_dir=root,
        source=source,
        fingerprint="f" * 64,
        disable_model_invocation=disabled,
    )


def _catalog(*manifests: SkillManifest) -> SkillCatalog:
    return SkillCatalog(tuple(manifests))


def _messages() -> tuple[ModelMessage, ...]:
    return (
        ModelMessage(role="system", content="SYS"),
        ModelMessage(role="user", content="hi"),
    )


def _measure(req: WireRequest) -> int:
    return measure_request(req).count


def _response(text: str) -> ModelResponse:
    return ModelResponse(
        content=text,
        tool_calls=(),
        usage=Usage(input_tokens=3, output_tokens=5),
        finish_reason="stop",
    )


def _start_turn(store: Store) -> None:
    store.ensure_thread(THREAD)
    create_turn(store, turn_id=TURN, thread_id=THREAD, binding={})


def _turn_deps(
    store: Store,
    bus: EventBus,
    workspace: Path,
    adapter: ScriptedAdapter,
    *,
    skills_catalog: SkillCatalog | None = None,
    compact_limit_tokens: int | None = None,
) -> TurnDeps:
    return TurnDeps(
        store=store,
        bus=bus,
        adapter=adapter,
        tools={},
        permissions=PermissionContext(
            workspace=str(workspace), policy=PermissionPolicy(), mode="never"
        ),
        workspace=workspace,
        system_prompt=SYSTEM_PROMPT,
        compact_limit_tokens=compact_limit_tokens,
        measure_tokens=_measure,
        skills_catalog=skills_catalog,
    )


# ---------------------------------------------------------------------------
# 纯函数语义
# ---------------------------------------------------------------------------


def test_empty_catalog_is_identity() -> None:
    """空目录 ⇒ 逐条同对象、同顺序,零注入。"""
    messages = _messages()
    out = append_skill_catalog(messages, SkillCatalog(()))
    assert out == messages
    assert out[0] is messages[0] and out[1] is messages[1]


def test_pure_idempotent_and_inserted_after_first_system_message() -> None:
    """纯(不动输入)、幂等(二次应用不变)、恰好插在第一条 system 之后(index 1)。"""
    source = list(_messages())
    catalog = _catalog(_manifest("alpha", "does alpha things"))

    once = append_skill_catalog(source, catalog)
    twice = append_skill_catalog(once, catalog)

    assert source == list(_messages())  # 输入未被修改
    assert twice == once  # 幂等:目录消息不重复
    assert [message.role for message in once] == ["system", "system", "user"]
    assert once[0] is source[0] and once[2] is source[1]
    assert once[1].content == f"{_HEADER}\n- alpha: does alpha things"


def test_no_system_message_inserts_at_front() -> None:
    """无 system 消息时目录消息插到最前,且仍幂等。"""
    messages = (ModelMessage(role="user", content="hi"),)
    catalog = _catalog(_manifest("alpha"))
    out = append_skill_catalog(messages, catalog)

    assert out[0].role == "system" and out[0].content.startswith(_HEADER)
    assert out[1] is messages[0]
    assert append_skill_catalog(out, catalog) == out


def test_disabled_skills_are_not_listed() -> None:
    """disable_model_invocation=True 的 skill 不进列表;全禁用 = 无可列 = 恒等。"""
    messages = _messages()
    catalog = _catalog(
        _manifest("alpha", "visible skill"),
        _manifest("manual", "manual only", disabled=True),
    )
    out = append_skill_catalog(messages, catalog)

    assert "- alpha: visible skill" in out[1].content
    assert "manual" not in out[1].content

    only_disabled = _catalog(_manifest("manual", "manual only", disabled=True))
    assert append_skill_catalog(messages, only_disabled) == messages


def test_long_description_is_truncated_with_ellipsis_marker() -> None:
    """description 超过 MAX_SKILL_DESCRIPTION_CHARS ⇒ 截断 + ``…`` 标记。"""
    long_description = "x" * (MAX_SKILL_DESCRIPTION_CHARS + 50)
    catalog = _catalog(_manifest("alpha", long_description))

    content = append_skill_catalog(_messages(), catalog)[1].content

    expected = f"{_HEADER}\n- alpha: {long_description[:MAX_SKILL_DESCRIPTION_CHARS]}…"
    assert content == expected
    assert long_description not in content


def test_budget_trims_whole_trailing_entries_and_marks_elision() -> None:
    """总预算:保留前缀整条目,裁掉尾部整条,以固定 marker 收尾。"""
    description = "d" * 100
    manifests = [_manifest(f"skill{index:03d}", description) for index in range(60)]
    content = append_skill_catalog(_messages(), _catalog(*manifests))[1].content

    assert len(content) <= MAX_CATALOG_LISTING_CHARS
    lines = content.split("\n")
    assert lines[0] == _HEADER and lines[-1] == _ELISION
    kept = lines[1:-1]
    assert 0 < len(kept) < len(manifests)  # 确实裁了尾部,且至少保留一条
    # 保留的都是完整的、按目录顺序的前缀条目(没有半截行)。
    assert kept == [
        f"- skill{index:03d}: {description}" for index in range(len(kept))
    ]


# ---------------------------------------------------------------------------
# turn 接线:目录消息进出站请求 + 压缩判定同口径
# ---------------------------------------------------------------------------


def test_catalog_message_reaches_build_request_wire_body(
    store: Store, bus: EventBus, workspace: Path
) -> None:
    """端到端:目录消息进入 adapter.build_request 定型的 wire body。"""
    _start_turn(store)
    store.append_items(
        THREAD, TURN, [ItemEntry(kind=ItemKind.USER_MESSAGE, payload={"text": "hi"})]
    )
    adapter = ScriptedAdapter([_response("done")])
    catalog = _catalog(_manifest("alpha", "does alpha things"))
    deps = _turn_deps(store, bus, workspace, adapter, skills_catalog=catalog)

    result = asyncio.run(run_turn(deps, TurnState(TURN, THREAD)))

    assert result.status is TurnStatus.COMPLETED and result.output == "done"
    request = adapter.last_request
    assert request is not None and request.serialized_json is not None
    assert _HEADER in request.serialized_json
    assert "- alpha: does alpha things" in request.serialized_json
    assert [message.role for message in request.messages[:2]] == ["system", "system"]


def test_compaction_measure_sees_the_same_enriched_messages(
    store: Store, bus: EventBus, workspace: Path
) -> None:
    """R-T7-1:压缩判定测量的是注入目录后的真实出站体,不是未注入的投影。

    阈值刻意取"未注入目录的请求"的精确 token 数:只有 measure 看到同一份
    enriched messages 才会严格超阈值并触发压缩。若目录在 build_request 之后
    才拼上(阈值盲区),measured == limit → 不压缩,本测试失败。
    """
    _start_turn(store)
    store.append_items(
        THREAD,
        TURN,
        [
            ItemEntry(kind=ItemKind.USER_MESSAGE, payload={"text": f"message {i} " + "z" * 40})
            for i in range(12)
        ],
    )
    catalog = _catalog(_manifest("alpha", "does alpha things"))
    adapter = ScriptedAdapter([_response("SUMMARY-TEXT"), _response("done")])

    baseline_messages = [
        ModelMessage(role="system", content=SYSTEM_PROMPT),
        *to_messages(build_context(store.read_items(THREAD))),
    ]
    baseline_tokens = _measure(
        adapter.build_request(baseline_messages, (), ModelSettings())
    )
    deps = _turn_deps(
        store,
        bus,
        workspace,
        adapter,
        skills_catalog=catalog,
        compact_limit_tokens=baseline_tokens,
    )

    result = asyncio.run(run_turn(deps, TurnState(TURN, THREAD)))

    assert result.status is TurnStatus.COMPLETED and result.output == "done"
    kinds = [item.kind for item in store.read_items(THREAD)]
    assert kinds.count(ItemKind.SUMMARY) == 1
    assert kinds.count(ItemKind.COMPACTION_BOUNDARY) == 1


# ---------------------------------------------------------------------------
# composition:显式参数绕过 fs 扫描;缺省从 workspace 双根装载
# ---------------------------------------------------------------------------


def test_composition_explicit_catalog_bypasses_filesystem_scan(workspace: Path) -> None:
    from praxis.composition import build_session

    adapter = ScriptedAdapter([_response("done")])
    catalog = _catalog(_manifest("explicit", "passed by the test"))
    session = build_session(workspace, adapter, skills_catalog=catalog)
    try:
        assert session._skills_catalog is catalog
        result = asyncio.run(session.submit("hi"))
        assert result.status is TurnStatus.COMPLETED
        request = adapter.last_request
        assert request is not None and request.serialized_json is not None
        assert "- explicit: passed by the test" in request.serialized_json
    finally:
        session.close()


def test_composition_loads_catalog_from_workspace(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """缺省参数 = 从 workspace 的 .agents/skills/ 装载(空目录自然恒等)。"""
    monkeypatch.delenv("SKILL_PATH", raising=False)
    skill_dir = workspace / ".agents" / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo\ndescription: demo skill\n---\n\nbody\n", encoding="utf-8"
    )
    from praxis.composition import build_session

    adapter = ScriptedAdapter([_response("done")])
    session = build_session(workspace, adapter)
    try:
        result = asyncio.run(session.submit("hi"))
        assert result.status is TurnStatus.COMPLETED
        request = adapter.last_request
        assert request is not None and request.serialized_json is not None
        assert "- demo: demo skill" in request.serialized_json
    finally:
        session.close()