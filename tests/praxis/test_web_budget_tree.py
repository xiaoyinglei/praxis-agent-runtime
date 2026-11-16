"""阶段4 T15:web turn 计数进预算树——dispatch gate、事实折叠口径与 Q8 默认兑现。

控制器裁决在此钉死:
- web 第 33 次 fetch(默认 turn_requests=32)→ ``web_request_budget_exceeded``
  tool_error(旧错误码 parity,非炸 turn);拒绝结果**不占配额**(pre-dispatch,
  与 denied/not_executed 同族——否则 --extend-web-requests 1 会被拒绝本身吃掉);
- deny 的调用不计数;started 无结果保守计 1(旧 reserve-before-IO parity);
- browser_requests 独立维(browser=True 同时消耗 web+browser 两维);
- 跨 resume 计数累计(树折叠天然成立:新 tracker 从事实重建);
- Q8:composition 把 WebLimits.turn_requests/browser_requests(默认 32/64)
  折进 root Limits 的 None 维;宿主 web.json 可调,显式 budget 维优先;
- T3 minor⑩:--extend-web-requests/--extend-browser-requests 出路旗标补齐,
  _echo_budget_hint 对 web/browser 维点名对应旗标;
- browser_render 旗标:渲染结果 metadata["browser_render"]=True(T2 fold 消费)。

有意差异(T17 对照表钉):旧 reserve_resource_unit SQLite 原子预扣 → 推导事实
+进程内 gate(跨进程原子归阶段 5);browser per-call 字节/时限不动(调用内预算
非账本)。仓库无 pytest-asyncio:与 test_turn_budget.py 同风格,asyncio.run 包装。
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from praxis.adapters.models.scripted import ScriptedAdapter
from praxis.adapters.web.browser import RenderedPage
from praxis.adapters.web.client import PublicHttpClient
from praxis.adapters.web.limits import WebLimits
from praxis.adapters.web.tools import create_web_fetch_tools
from praxis.core.budget import BudgetTracker, fold_turn_facts
from praxis.core.events import EventBus
from praxis.core.pipeline import ExecCtx, execute
from praxis.core.statemachine import TurnStatus
from praxis.interfaces.cli import app
from praxis.interfaces.cli import main as cli_main
from praxis.interfaces.sdk import open_session
from praxis.protocol.budget import Limits
from praxis.protocol.floor import host_config_dir
from praxis.protocol.items import ItemKind
from praxis.protocol.json import JsonValue
from praxis.protocol.models import ModelResponse, ModelToolCall, Usage
from praxis.protocol.permission import PermissionContext, SandboxAvailability
from praxis.protocol.policy import NetworkPolicy, PermissionPolicy, ToolRule
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
from praxis.store.log import ItemEntry, Store
from praxis.store.turns import create_turn

THREAD = "thread-1"
TURN = "turn-1"
_EMPTY_SCHEMA: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}


# ---------------------------------------------------------------------------
# 脚手架(与 test_turn_budget.py 同风格)
# ---------------------------------------------------------------------------


def _resp(content: str, calls: tuple[tuple[str, str], ...] = ()) -> ModelResponse:
    return ModelResponse(
        content=content,
        tool_calls=tuple(ModelToolCall(call_id=c, name=n, arguments={}) for c, n in calls),
        usage=Usage(0, 0),
        finish_reason="tool_calls" if calls else "stop",
    )


def _web_toy(
    name: str = "web_fetch",
    *,
    effects: frozenset[ToolEffect] = frozenset(),
    metadata: dict[str, JsonValue] | None = None,
) -> Tool:
    """空 effects 玩具经 decide 自动放行(与 test_turn_budget 同口径);gate 按名字判 web 维。

    ``metadata`` 模拟真实 web 工具的结果旗标(如 render 面的 browser_render)。
    """

    async def run(arguments: Mapping[str, JsonValue], exec_ctx: ToolExecContext) -> ToolResult:
        return ToolResult(
            tool_call_id="pending",
            tool_name=name,
            content=(ToolContentBlock(type="text", data={"text": "fetched"}),),
            status=ResultStatus.OK,
            metadata=metadata or {},
        )

    return Tool(
        definition=ToolDefinition(name=name, description=f"{name} toy", input_schema=_EMPTY_SCHEMA),
        validate_input=json_schema_input(_EMPTY_SCHEMA),
        run=run,
        static_effects=effects,
        resolve_use=lambda arguments: ResolvedUse(effects=effects, targets=()),
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=30,
        max_model_output_bytes=64_000,
    )


def _turn(store: Store, *, turn_id: str = TURN) -> None:
    store.ensure_thread(THREAD)
    create_turn(store, turn_id=turn_id, thread_id=THREAD, binding={})


def _tracker(store: Store, limits: Limits, *, turn_id: str = TURN) -> BudgetTracker:
    return BudgetTracker(
        store=store, thread_id=THREAD, turn_id=turn_id, root_limits=limits, unit_costs=None
    )


def _result_item(
    call_id: str,
    *,
    status: str = "ok",
    tool_name: str = "web_fetch",
    metadata: dict[str, JsonValue] | None = None,
    error_code: str | None = None,
) -> ItemEntry:
    payload: dict[str, JsonValue] = {
        "call_id": call_id,
        "tool_name": tool_name,
        "status": status,
        "content": (),
        "is_error": status in ("error", "timeout", "cancelled"),
        "error_code": error_code,
        "error_message": None,
        "truncated": False,
        "metadata": metadata or {},
    }
    return ItemEntry(kind=ItemKind.TOOL_RESULT, payload=payload)


def _call_item(call_id: str, *, tool_name: str = "web_fetch") -> ItemEntry:
    return ItemEntry(
        kind=ItemKind.TOOL_CALL,
        payload={"call_id": call_id, "tool_name": tool_name, "arguments": {}},
    )


def _results(session: Any) -> dict[str, Mapping[str, JsonValue]]:
    return {
        str(item.payload.get("call_id")): item.payload
        for item in session.items()
        if item.kind == ItemKind.TOOL_RESULT
    }


# ---------------------------------------------------------------------------
# brief ④:第 33 次 fetch → web_request_budget_exceeded(tool_error,非炸 turn)
# ---------------------------------------------------------------------------


def test_web_33rd_fetch_exceeded_tool_error_not_turn_failure(workspace: Path) -> None:
    calls = tuple((f"c{i}", "web_fetch") for i in range(1, 34))
    script = [_resp("", (call,)) for call in calls] + [_resp("done")]
    session = open_session(
        workspace, ScriptedAdapter(script), tools=[_web_toy()], budget=Limits(web_requests=32)
    )
    try:
        result = asyncio.run(session.submit("go"))
        assert result.status is TurnStatus.COMPLETED  # 非炸 turn:模型拿到错误后自行收尾
        results = _results(session)
        for i in range(1, 33):
            assert results[f"c{i}"]["status"] == "ok"
        assert results["c33"]["status"] == "error"
        assert results["c33"]["error_code"] == "web_request_budget_exceeded"
        assert results["c33"]["is_error"] is True
        # 拒绝发生在 dispatch 之前:不写 tool_call(未执行语义同 invalid_input)。
        calls_items = [i for i in session.items() if i.kind == ItemKind.TOOL_CALL]
        assert len(calls_items) == 32
        # 拒绝结果不占配额(fold 例外):used 恰 32,不是 33。
        facts = fold_turn_facts(session.items(), unit_costs=None)
        assert facts.web_requests == 32
    finally:
        session.close()


def test_budget_exceeded_result_not_billable_in_fold(store: Store) -> None:
    """pre-dispatch 预算拒绝与 denied/not_executed 同族:不消耗配额。

    否则 resume --extend-web-requests 1 的增量会被拒绝结果本身吃掉,永远续不上。
    """
    _turn(store)
    store.append_items(
        THREAD,
        TURN,
        [
            _call_item("z1"),
            _result_item("z1", status="error", error_code="web_request_budget_exceeded"),
        ],
    )
    facts = fold_turn_facts(store.read_turn_items(TURN), unit_costs=None)
    assert facts.web_requests == 0
    tracker = _tracker(store, Limits(web_requests=1))
    assert tracker.admit_web_request("web_fetch", browser=False).allowed


def test_web_deny_does_not_count(workspace: Path) -> None:
    script = [_resp("", (("c1", "web_fetch"),)), _resp("", (("c2", "web_fetch"),)), _resp("done")]
    session = open_session(
        workspace,
        ScriptedAdapter(script),
        tools=[_web_toy()],
        budget=Limits(web_requests=1),
        policy=PermissionPolicy(tools=(ToolRule(pattern="web_fetch", decision="deny"),)),
    )
    try:
        result = asyncio.run(session.submit("go"))
        assert result.status is TurnStatus.COMPLETED
        results = _results(session)
        # 两条都是权限 deny(不是预算拒绝):deny 不计数,第二条仍有机会执行。
        assert results["c1"]["status"] == "denied"
        assert results["c2"]["status"] == "denied"
        assert results["c1"]["error_code"] == "permission_denied"
        assert results["c2"]["error_code"] == "permission_denied"
        assert fold_turn_facts(session.items(), unit_costs=None).web_requests == 0
    finally:
        session.close()


# ---------------------------------------------------------------------------
# started 无结果保守计 1 / browser 独立维(tracker 级)
# ---------------------------------------------------------------------------


def test_started_web_call_without_result_counts_one(store: Store) -> None:
    _turn(store)
    store.append_items(THREAD, TURN, [_call_item("x1")])
    facts = fold_turn_facts(store.read_turn_items(TURN), unit_costs=None)
    assert facts.web_requests == 1
    tracker = _tracker(store, Limits(web_requests=1))
    admission = tracker.admit_web_request("web_fetch", browser=False)
    assert not admission.allowed
    assert admission.dimension == "web_requests"


def test_browser_dimension_independent_and_dual_consumption(store: Store) -> None:
    _turn(store)
    tracker = _tracker(store, Limits(web_requests=10, browser_requests=1))
    assert tracker.admit_web_request("web_fetch", browser=True).allowed
    # 结果落库后才 note(管道序纪律);browser_render 旗标进 browser 维。
    store.append_items(
        THREAD, TURN, [_result_item("b1", metadata={"browser_render": True})]
    )
    tracker.note_result("web_fetch", browser=True)
    facts = tracker.refresh().used
    assert (facts.web_requests, facts.browser_requests) == (1, 1)
    # browser 维耗尽:browser 请求拒,纯 web 请求仍放行(web 维有余量)。
    denied = tracker.admit_web_request("web_fetch", browser=True)
    assert not denied.allowed and denied.dimension == "browser_requests"
    assert tracker.admit_web_request("web_fetch", browser=False).allowed
    # 无 browser_render 旗标的结果只计 web 维。
    store.append_items(THREAD, TURN, [_result_item("w1")])
    facts = tracker.refresh().used
    assert (facts.web_requests, facts.browser_requests) == (2, 1)


def test_non_web_tool_name_unconstrained(store: Store) -> None:
    _turn(store)
    tracker = _tracker(store, Limits(web_requests=0, browser_requests=0))
    assert tracker.admit_web_request("read", browser=False).allowed
    tracker.note_result("read", browser=False)  # no-op,不负债


# ---------------------------------------------------------------------------
# pipeline gate:in-flight 释放 / browser 旗标来自 effects / budget=None 零扰动
# ---------------------------------------------------------------------------


def _ctx(
    store: Store,
    workspace: Path,
    tools: dict[str, Tool],
    *,
    budget: BudgetTracker | None,
    mode: str = "bypass",
    turn_id: str = TURN,
) -> ExecCtx:
    _turn(store, turn_id=turn_id)
    # network allow:EXECUTE_PROCESS 的命令网络姿态门控放行(D9 默认 ask 会
    # 把 render 面玩具拖进审批,与本测试的预算判据无关)。
    policy = PermissionPolicy(network=NetworkPolicy(decision="allow"))
    return ExecCtx(
        store=store,
        bus=EventBus(),
        workspace=workspace,
        tools=tools,
        permissions=PermissionContext(
            workspace=str(workspace),
            policy=policy,
            mode=mode,  # type: ignore[arg-type]
            sandbox=SandboxAvailability(backend="seatbelt", available=True),
        ),
        turn_id=turn_id,
        thread_id=THREAD,
        policy_revision=policy.revision,
        budget=budget,
    )


def test_pipeline_inflight_released_after_result(store: Store, workspace: Path) -> None:
    """note_result 在结果落库后释放 in-flight:限额 2 恰放行两次,第三次拒。

    若 in-flight 泄漏(不 note),第二次就会被 fold(1)+in-flight(1)=2 误拒。
    """
    tracker = _tracker(store, Limits(web_requests=2))
    ctx = _ctx(store, workspace, {"web_fetch": _web_toy()}, budget=tracker)
    for call_id in ("c1", "c2"):
        result = asyncio.run(execute(ToolCall(tool_call_id=call_id, tool_name="web_fetch", arguments={}), ctx))
        assert result.status is ResultStatus.OK
    third = asyncio.run(execute(ToolCall(tool_call_id="c3", tool_name="web_fetch", arguments={}), ctx))
    assert third.status is ResultStatus.ERROR
    assert third.error_code == "web_request_budget_exceeded"
    assert "web_requests" in (third.error_message or "")


def test_pipeline_browser_flag_from_execute_process_effect(store: Store, workspace: Path) -> None:
    """render 面(effects 含 EXECUTE_PROCESS)按 browser=True 准入,独立维耗尽点名。"""
    render_effects = frozenset({ToolEffect.NETWORK, ToolEffect.EXECUTE_PROCESS})
    tracker = _tracker(store, Limits(web_requests=5, browser_requests=1))
    # 结果带 browser_render 旗标(真实 render 工具的形状):fold 才计 browser 维。
    render_toy = _web_toy(effects=render_effects, metadata={"browser_render": True})
    ctx = _ctx(store, workspace, {"web_fetch": render_toy}, budget=tracker)
    first = asyncio.run(execute(ToolCall(tool_call_id="c1", tool_name="web_fetch", arguments={}), ctx))
    assert first.status is ResultStatus.OK
    second = asyncio.run(execute(ToolCall(tool_call_id="c2", tool_name="web_fetch", arguments={}), ctx))
    assert second.status is ResultStatus.ERROR
    assert second.error_code == "web_request_budget_exceeded"
    assert "browser_requests" in (second.error_message or "")
    assert tracker.refresh().used.browser_requests == 1  # 拒绝不占 browser 配额


def test_pipeline_budget_none_zero_disturbance(store: Store, workspace: Path) -> None:
    ctx = _ctx(store, workspace, {"web_fetch": _web_toy()}, budget=None)
    for call_id in ("c1", "c2", "c3"):
        result = asyncio.run(execute(ToolCall(tool_call_id=call_id, tool_name="web_fetch", arguments={}), ctx))
        assert result.status is ResultStatus.OK


# ---------------------------------------------------------------------------
# 跨 resume 累计(树折叠天然成立)
# ---------------------------------------------------------------------------


def test_web_counts_accumulate_across_resume(workspace: Path) -> None:
    # c1 fetch(用掉唯一配额)→ c2 probe ask 暂停;resume 批准后 c3 fetch 必须
    # 看到跨 resume 的累计 used=1 → 预算拒绝(新 tracker 从事实重建)。
    script = [
        _resp("", (("c1", "web_fetch"),)),
        _resp("", (("c2", "probe"),)),
        _resp("", (("c3", "web_fetch"),)),
        _resp("done"),
    ]
    probe = _web_toy("probe", effects=frozenset({ToolEffect.NETWORK}))
    session = open_session(
        workspace, ScriptedAdapter(script), tools=[_web_toy(), probe], budget=Limits(web_requests=1)
    )
    try:
        paused = asyncio.run(session.submit("go"))
        assert paused.status is TurnStatus.PAUSED and paused.reason == "approval_required"
        assert [a.call_id for a in paused.pending_approvals] == ["c2"]
        result = asyncio.run(session.resume(paused.turn_id, approval={"c2": "allow"}))
        assert result.status is TurnStatus.COMPLETED
        results = _results(session)
        assert results["c1"]["status"] == "ok"
        assert results["c3"]["status"] == "error"
        assert results["c3"]["error_code"] == "web_request_budget_exceeded"
        assert fold_turn_facts(session.items(), unit_costs=None).web_requests == 1
    finally:
        session.close()


# ---------------------------------------------------------------------------
# Q8:WebLimits 默认折进 root Limits;宿主配置可调;显式维优先
# ---------------------------------------------------------------------------


def test_q8_default_web_limits_folded_into_binding(workspace: Path) -> None:
    session = open_session(workspace, ScriptedAdapter([_resp("hi")]), tools=[_web_toy()])
    try:
        result = asyncio.run(session.submit("go"))
        assert result.status is TurnStatus.COMPLETED
        assert session.status(result.turn_id).binding["budget"] == {
            "web_requests": 32,
            "browser_requests": 64,
        }
    finally:
        session.close()


def test_q8_host_config_overrides_default_and_enforced(workspace: Path) -> None:
    config_dir = host_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "web.json").write_text(
        json.dumps({"limits": {"turn_requests": 4}}), encoding="utf-8"
    )
    calls = tuple((f"c{i}", "web_fetch") for i in range(1, 6))
    script = [_resp("", (call,)) for call in calls] + [_resp("done")]
    session = open_session(workspace, ScriptedAdapter(script), tools=[_web_toy()])
    try:
        result = asyncio.run(session.submit("go"))
        assert result.status is TurnStatus.COMPLETED
        assert session.status(result.turn_id).binding["budget"]["web_requests"] == 4
        results = _results(session)
        assert results["c4"]["status"] == "ok"
        assert results["c5"]["error_code"] == "web_request_budget_exceeded"
    finally:
        session.close()


def test_q8_explicit_budget_dimension_wins_over_config(workspace: Path) -> None:
    config_dir = host_config_dir()
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "web.json").write_text(
        json.dumps({"limits": {"turn_requests": 4}}), encoding="utf-8"
    )
    session = open_session(
        workspace, ScriptedAdapter([_resp("hi")]), tools=[_web_toy()], budget=Limits(web_requests=7)
    )
    try:
        result = asyncio.run(session.submit("go"))
        binding = session.status(result.turn_id).binding["budget"]
        assert binding == {"web_requests": 7, "browser_requests": 64}
    finally:
        session.close()


def test_web_limits_turn_requests_field_and_range() -> None:
    assert WebLimits().turn_requests == 32  # 旧 web_limits.py:9 默认 parity
    assert WebLimits(turn_requests=1).turn_requests == 1
    assert WebLimits(turn_requests=256).turn_requests == 256
    for kwargs in ({"turn_requests": 0}, {"turn_requests": 257}, {"turn_requests": 1.5}, {"turn_requests": True}):
        with pytest.raises(ValueError):
            WebLimits(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# browser_render 旗标:渲染结果写 metadata(T2 fold 消费的形状)
# ---------------------------------------------------------------------------


class _FakeRenderer:
    async def render(self, url: str, *, initial: Any = None) -> RenderedPage:
        return RenderedPage(
            text="line one\nline two",
            final_url=url,
            requests_used=2,
            bytes_used=100,
            dom_bytes=200,
            truncated=False,
            blocked_ask=0,
            blocked_deny=0,
            blocked_other=0,
            budget_aborted=0,
            budget_stopped=False,
            failed_resources=0,
            redirected_subresources=0,
            load_timeouts=(),
        )


def test_render_result_carries_browser_render_flag(store: Store) -> None:
    client = PublicHttpClient(authorize=lambda url: None, limits=WebLimits())
    tool = create_web_fetch_tools(client, limits=WebLimits(), renderer_factory=lambda: _FakeRenderer())[0]
    result = asyncio.run(
        tool.run({"url": "https://example.com/", "render": True}, ToolExecContext())
    )
    assert result.status is ResultStatus.OK
    assert result.metadata["rendered"] is True
    assert result.metadata["browser_render"] is True  # T2 fold browser 维判据
    # 旗标进 fold 即 browser 维计 1(纯文本结果无旗标只计 web,见 tracker 级测试)。
    _turn(store)
    store.append_items(THREAD, TURN, [_result_item("r1", metadata=dict(result.metadata))])
    facts = fold_turn_facts(store.read_turn_items(TURN), unit_costs=None)
    assert (facts.web_requests, facts.browser_requests) == (1, 1)


# ---------------------------------------------------------------------------
# T3 minor⑩:CLI extend 旗标补齐 + hint 点名
# ---------------------------------------------------------------------------


def test_budget_extend_flags_cover_web_dimensions() -> None:
    assert cli_main._BUDGET_EXTEND_FLAGS["web_requests"] == "--extend-web-requests"
    assert cli_main._BUDGET_EXTEND_FLAGS["browser_requests"] == "--extend-browser-requests"


@pytest.mark.parametrize(
    ("dimension", "flag"),
    [("web_requests", "--extend-web-requests"), ("browser_requests", "--extend-browser-requests")],
)
def test_echo_budget_hint_names_web_flags(
    dimension: str, flag: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    lines: list[str] = []
    monkeypatch.setattr(cli_main.typer, "echo", lambda text="", **kw: lines.append(str(text)))
    result = cli_main.TurnResult(
        turn_id="t1",
        status=TurnStatus.PAUSED,
        output="",
        reason="budget_exhausted",
        budget_dimension=dimension,
    )
    cli_main._echo_budget_hint(result)
    assert lines and flag in lines[0] and dimension in lines[0]


runner = CliRunner()
_CLEAN_ENV = {
    "PRAXIS_MODEL_ID": None,
    "PRAXIS_MODEL": None,
    "PRAXIS_BASE_URL": None,
    "PRAXIS_API_KEY": None,
    "OPENAI_API_KEY": None,
}


def test_cli_resume_extend_web_flags_write_extension(workspace: Path, tmp_path: Path) -> None:
    script = tmp_path / "steps.json"
    script.write_text(
        json.dumps(
            [
                {"content": "", "tool_calls": [{"call_id": "c1", "name": "list_files", "arguments": {}}]},
                {"content": "", "tool_calls": [{"call_id": "c2", "name": "list_files", "arguments": {}}]},
                {"content": "done"},
            ]
        ),
        encoding="utf-8",
    )
    paused = runner.invoke(
        app,
        ["run", "go", "--workspace", str(workspace), "--thread", "w1", "--scripted", str(script), "--max-steps", "1"],
        env=_CLEAN_ENV,
    )
    assert paused.exit_code == 2, paused.output
    match = re.search(r"turn ([0-9a-f]{32})", paused.output)
    assert match is not None, paused.output
    turn_id = match.group(1)
    resumed = runner.invoke(
        app,
        [
            "resume", turn_id, "--workspace", str(workspace), "--thread", "w1", "--scripted", str(script),
            "--extend-steps", "5", "--extend-web-requests", "5", "--extend-browser-requests", "7",
        ],
        env=_CLEAN_ENV,
    )
    assert resumed.exit_code == 0, resumed.output
    reader = open_session(workspace, ScriptedAdapter([]), thread_id="w1")
    try:
        extensions = [i for i in reader.items() if i.kind == ItemKind.BUDGET_EXTENSION]
        assert len(extensions) == 1
        assert extensions[0].payload == {
            "limits_delta": {"steps": 5, "web_requests": 5, "browser_requests": 7}
        }
    finally:
        reader.close()