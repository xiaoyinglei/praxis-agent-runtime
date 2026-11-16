"""Task 21 Step 2:真实故障注入(lean 版——只补任务级测试未证明的**端到端形状**)。

控制器 scope-tighten(2026-11-16):任务级测试已拥有的运行面不重复实现,
此处只经**真实管道/真实会话**证明集成形状:

(a) MCP server 会话中途被 SIGKILL ⇒ 下一次调用有界 ``mcp_call_failed``、
    bundle 收尾干净、ps 无孤儿、重开正常重连(T16/T19 拥有握手挂起/崩溃
    隔离/关闭超时的机制面;此处补"会话飞行中死亡"的管道形状);
(b) skill 文件在两次调用之间被改写 ⇒ ``skill_content_changed``;改成带
    约束 ⇒ ``unsupported_skill_constraints``——经真实 ``pipeline.execute``
    (T8 拥有工具级漂移映射;此处补 item 落库的管道形状);
(c) 配置字节在 connect 与 call 之间漂移 ⇒ ``mcp_config_changed`` 经真实
    ``execute``(T17 拥有工具级守卫;此处补管道形状);
(d) gzip bomb 经真实 web_fetch + 真实管道 + 真实授权闭包端到端:network
    allow ⇒ dispatch 后流式解码超限 ``web_fetch_failed:max_bytes_exceeded``;
    默认 network ask ⇒ decide 第 2 步先暂停,**零网络**(S8 集成形状)。

**不重复**(控制器点名,任务级测试拥有):
- hook 超时/整组 kill/有界排空/取消三连:T4
  ``test_hooks_runner.py::test_timeout_kills_entire_process_group``、
  ``::test_outer_cancellation_cleans_up_and_reraises``;真实崩溃面由
  ``test_kill9_hook_inflight.py``(本阶段)拥有。
- materialize 符号链接逃逸/终级 symlink/FIFO:T9
  ``test_skills_materialize.py::test_source_symlink_outside_skill_root_is_rejected``、
  ``::test_destination_symlink_escaping_workspace_is_invalid_input``、
  ``::test_fifo_asset_is_not_a_regular_file``。
- DNS rebinding/TLS 中途失败的客户端机制:T10
  ``test_web_urls_dns.py``(全答案预检/数字直连/字节前回退);此处 (d)
  选 gzip bomb 作为唯一端到端网络故障样本(同一 resolver/backend 接缝)。

fixture 分层与 T16/T19 同纪律:真 mcp SDK + 真子进程,沙箱用
``/usr/bin/env`` 透传 spec(机制证明,不冒充安全边界;真 seatbelt 子集由
T16 Tier A 拥有)。每个用例结束 ps 复查无孤儿(2026-11-13 事故教训)。
"""

from __future__ import annotations

import asyncio
import gzip
import os
import signal
import subprocess
import sys
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest
import yaml

from praxis.adapters.mcp.manager import McpBundle, connect_mcp_servers
from praxis.adapters.models.scripted import ScriptedAdapter
from praxis.adapters.web.client import PublicHttpClient
from praxis.adapters.web.limits import WebLimits
from praxis.adapters.web.tools import create_web_fetch_tools
from praxis.composition import build_session, make_web_authorizer
from praxis.core.events import EventBus
from praxis.core.pipeline import ExecCtx, execute
from praxis.protocol.floor import host_config_dir
from praxis.protocol.items import ItemKind
from praxis.protocol.models import ModelResponse, ModelToolCall, Usage, WireRequest
from praxis.protocol.permission import PermissionContext
from praxis.protocol.policy import NetworkPolicy, PermissionPolicy
from praxis.protocol.sandbox import SandboxSpec, ToolExecContext
from praxis.protocol.tools import Tool, ToolCall
from praxis.store.log import Store
from praxis.store.turns import create_turn
from praxis.tools.builtins.child_env import child_env
from praxis.tools.builtins.skills import create_invoke_skill_tool, load_skill_catalog
from tests.praxis.test_web_urls_dns import PUBLIC_IP, FakeBackend, FakeResolver, FakeStream, response_chunks

_FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "mcp_probe_server.py"
_OWNER_PREFIX = "mcp-owner:"


# ---------------------------------------------------------------------------
# 脚手架
# ---------------------------------------------------------------------------


def _token() -> str:
    return "t21-" + uuid.uuid4().hex


def _ps_lines(token: str) -> list[str]:
    completed = subprocess.run(
        ["ps", "-eo", "pid=,args="], capture_output=True, text=True, timeout=20, check=False
    )
    return [row for row in completed.stdout.splitlines() if token in row]


def _server_pid(token: str) -> int:
    rows = _ps_lines(token)
    assert rows, f"fixture server not visible in ps: {token}"
    return int(rows[0].split()[0])


async def _wait_process_gone(token: str, timeout: float = 10.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while _ps_lines(token):
        if loop.time() > deadline:
            pytest.fail(f"orphan MCP server still alive: {_ps_lines(token)}")
        await asyncio.sleep(0.05)


def _owner_tasks() -> list[asyncio.Task[Any]]:
    return [task for task in asyncio.all_tasks() if task.get_name().startswith(_OWNER_PREFIX)]


def _write_config(servers: list[dict[str, Any]]) -> Path:
    directory = host_config_dir()
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o700)
    path = directory / "mcp_servers.yaml"
    path.write_text(yaml.safe_dump({"servers": servers}), encoding="utf-8")
    path.chmod(0o600)
    return path


def _stdio_entry(token: str, *, secret: str) -> dict[str, Any]:
    return {
        "name": "probe",
        "enabled": True,
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-I", str(_FIXTURE), "--token", token],
        "env": {"SECRET_MCP_KEY": secret},
        "allow_all_tools": True,
        "startup_timeout_seconds": 10.0,
    }


def _compiler(_permissions: Any, _use: Any, workspace: Path) -> SandboxSpec:
    """Tier B:真 exec 透传前缀(机制证明;安全边界由 T16 Tier A 拥有)。"""
    scratch = workspace / ".mcp-scratch"
    scratch.mkdir(exist_ok=True)
    return SandboxSpec(backend="test-passthrough", argv_prefix=("/usr/bin/env",), scratch_dir=str(scratch))


async def _connect(workspace: Path, policy: PermissionPolicy | None = None) -> McpBundle:
    return await connect_mcp_servers(
        workspace, policy=policy, sandbox_compiler=_compiler, env_builder=child_env
    )


def _resp_call(call_id: str, name: str, arguments: Mapping[str, Any]) -> ModelResponse:
    return ModelResponse(
        content="",
        tool_calls=(ModelToolCall(call_id=call_id, name=name, arguments=dict(arguments)),),
        usage=Usage(input_tokens=1, output_tokens=1),
        finish_reason="tool_calls",
    )


_RESP_DONE = ModelResponse(
    content="done", tool_calls=(), usage=Usage(input_tokens=1, output_tokens=1), finish_reason="stop"
)


class _SeamedAdapter(ScriptedAdapter):
    """在消费第 N 条剧本**之前**执行真实故障动作(kill/改文件)。

    缝在模型边界触发:第 N-1 个工具结果已提交、第 N 个调用尚未 dispatch——
    故障恰好落在"会话中途",而不是测试装置的外部摆拍。
    """

    def __init__(self, responses: Sequence[ModelResponse], seams: Mapping[int, Any]) -> None:
        super().__init__(responses)
        self._seams = dict(seams)
        self._served = 0

    async def stream(self, req: WireRequest) -> AsyncIterator[Any]:
        action = self._seams.get(self._served)
        self._served += 1
        if action is not None:
            action()
        async for event in super().stream(req):
            yield event


def _exec_ctx(
    store: Store,
    workspace: Path,
    tools: Mapping[str, Tool],
    *,
    policy: PermissionPolicy | None = None,
    permissions: PermissionContext | None = None,
    turn_id: str = "turn-1",
) -> ExecCtx:
    effective = policy if policy is not None else PermissionPolicy()
    create_turn(store, turn_id=turn_id, thread_id="thread-1", binding={})
    return ExecCtx(
        store=store,
        bus=EventBus(),
        workspace=workspace,
        tools=dict(tools),
        permissions=permissions
        if permissions is not None
        else PermissionContext(workspace=str(workspace), policy=effective, mode="on-request"),
        turn_id=turn_id,
        thread_id="thread-1",
        policy_revision=effective.revision,
    )


def _results(items: Sequence[Any], call_id: str) -> list[Any]:
    return [
        i for i in items if i.kind == ItemKind.TOOL_RESULT and i.payload.get("call_id") == call_id
    ]


# ---------------------------------------------------------------------------
# (a) MCP server 会话中途 SIGKILL:有界失败 + 干净收尾 + 重连
# ---------------------------------------------------------------------------


def test_mcp_server_killed_mid_session_bounded_failure_and_clean_reconnect(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    token = _token()
    secret = "sk-t21-" + uuid.uuid4().hex
    _write_config([_stdio_entry(token, secret=secret)])

    async def scenario() -> None:
        bundle = await _connect(workspace)
        assert [s.connected for s in bundle.statuses] == [True]
        pid = _server_pid(token)
        # 缝:第二条剧本消费前(首个 echo 已提交)SIGKILL 真 server 进程。
        adapter = _SeamedAdapter(
            [
                _resp_call("c1", "mcp__probe__echo", {"message": "first"}),
                _resp_call("c2", "mcp__probe__echo", {"message": "second"}),
                _RESP_DONE,
            ],
            {1: lambda: os.kill(pid, signal.SIGKILL)},
        )
        session = build_session(
            workspace, adapter, policy=PermissionPolicy(), extra_tools=bundle.tools, integrations=(bundle,)
        )
        try:
            result = await session.submit("fault: mcp kill mid-session")
            assert result.status.value == "completed"
            items = session.items()
            first = _results(items, "c1")
            second = _results(items, "c2")
            assert len(first) == 1 and first[0].payload.get("status") == "ok"
            assert "echo: first" in str(first[0].payload.get("content"))
            # 死亡后的调用:有界 mcp_call_failed(异常组摊平,≤512,秘密不进消息)。
            assert len(second) == 1
            assert second[0].payload.get("error_code") == "mcp_call_failed"
            message = str(second[0].payload.get("error_message"))
            assert 0 < len(message) <= 512
            assert secret not in message
            assert "TaskGroup" not in message
            assert second[0].payload.get("metadata", {}).get("mcp_server") == "probe"
        finally:
            await session.aclose()  # integrations → bundle.aclose(幂等收尾)
        # 收尾干净:owner task 零残留、进程组零孤儿。
        assert _owner_tasks() == []
        await _wait_process_gone(token)
        # 重开正常重连:同一配置新 bundle,echo 再次可用。
        bundle2 = await _connect(workspace)
        try:
            assert [s.connected for s in bundle2.statuses] == [True]
            (echo,) = [t for t in bundle2.tools if t.definition.name == "mcp__probe__echo"]
            ok = await echo.run({"message": "again"}, ToolExecContext())
            assert ok.status.value == "ok"
            assert "echo: again" in ok.content[0].data["text"]
        finally:
            await bundle2.aclose()
        assert _owner_tasks() == []
        await _wait_process_gone(token)

    asyncio.run(scenario())
    assert _ps_lines(token) == []


# ---------------------------------------------------------------------------
# (b) skill 漂移经真实管道:改写 ⇒ content_changed;约束注入 ⇒ unsupported
# ---------------------------------------------------------------------------


def test_skill_rewrite_and_constraint_injection_through_real_execute(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / ".git").mkdir()
    skill_dir = workspace / ".agents" / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        "---\nname: demo\ndescription: test skill\n---\noriginal instructions\n", encoding="utf-8"
    )
    catalog = load_skill_catalog(workspace)
    tool = create_invoke_skill_tool(catalog)
    store = Store(workspace / "store.db")
    try:
        ctx = _exec_ctx(store, workspace, {"invoke_skill": tool})
        first = asyncio.run(execute(ToolCall("c1", "invoke_skill", {"name": "demo"}), ctx))
        assert first.status.value == "ok"
        assert first.metadata["skill_id"] == "demo"
        # 会话中途改写正文 ⇒ 下一次激活重读重算 ⇒ skill_content_changed。
        skill_file.write_text(
            "---\nname: demo\ndescription: test skill\n---\nrewritten instructions\n", encoding="utf-8"
        )
        second = asyncio.run(execute(ToolCall("c2", "invoke_skill", {"name": "demo"}), ctx))
        assert second.status.value == "error"
        assert second.error_code == "skill_content_changed"
        # 约束注入(装载期排除的字段漂移进来)⇒ unsupported_skill_constraints。
        skill_file.write_text(
            "---\nname: demo\ndescription: test skill\nallowed_tools:\n  - bash\n---\nrewritten\n",
            encoding="utf-8",
        )
        third = asyncio.run(execute(ToolCall("c3", "invoke_skill", {"name": "demo"}), ctx))
        assert third.error_code == "unsupported_skill_constraints"
        # 管道形状:三次调用各有一份 tool_result item 落库,错误不炸穿 turn。
        items = store.read_turn_items("turn-1")
        assert [r.payload.get("status") for r in _results(items, "c1")] == ["ok"]
        assert [r.payload.get("error_code") for r in _results(items, "c2")] == ["skill_content_changed"]
        assert [r.payload.get("error_code") for r in _results(items, "c3")] == ["unsupported_skill_constraints"]
        # 失败结果不授予激活态(core.skill_state 只投影成功项)。
        from praxis.core.skill_state import latest_active_skill_ids

        assert latest_active_skill_ids(items) == frozenset({"demo"})  # 仅 c1 成功
    finally:
        store.close()


# ---------------------------------------------------------------------------
# (c) 配置字节漂移经真实管道:mcp_config_changed(唯一调用期守卫)
# ---------------------------------------------------------------------------


def test_config_sha_drift_between_connect_and_call_through_real_execute(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    token = _token()
    secret = "sk-t21-" + uuid.uuid4().hex
    config_path = _write_config([_stdio_entry(token, secret=secret)])

    async def scenario() -> None:
        bundle = await _connect(workspace)
        try:
            (echo,) = [t for t in bundle.tools if t.definition.name == "mcp__probe__echo"]
            store = Store(workspace / "store.db")
            try:
                ctx = _exec_ctx(store, workspace, {echo.definition.name: echo})
                first = await execute(ToolCall("c1", echo.definition.name, {"message": "before"}), ctx)
                assert first.status.value == "ok"
                # connect 之后改写配置字节(仍是合法 YAML;信任门在下次加载才跑,
                # 调用期守卫只比 sha256)。
                config_path.write_text(
                    config_path.read_text(encoding="utf-8") + "# drift after connect\n", encoding="utf-8"
                )
                second = await execute(ToolCall("c2", echo.definition.name, {"message": "after"}), ctx)
                assert second.status.value == "error"
                assert second.error_code == "mcp_config_changed"
                assert second.error_message == "config changed; reconnect"
                assert second.metadata["mcp_server"] == "probe"
                # server 未被漂移守卫杀死(重连问题,不是进程问题)。
                assert _ps_lines(token)
                items = store.read_turn_items("turn-1")
                assert [r.payload.get("error_code") for r in _results(items, "c2")] == ["mcp_config_changed"]
            finally:
                store.close()
        finally:
            await bundle.aclose()
        assert _owner_tasks() == []
        await _wait_process_gone(token)

    asyncio.run(scenario())
    assert _ps_lines(token) == []


# ---------------------------------------------------------------------------
# (d) gzip bomb 端到端:真实 web_fetch + 真实管道 + 真实授权闭包
# ---------------------------------------------------------------------------


def test_gzip_bomb_end_to_end_through_pipeline_and_permission(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    bomb = gzip.compress(b"A" * 200_000)
    assert len(bomb) < 4096  # 压缩体很小,解码体远超预算

    def _client(permissions: PermissionContext) -> tuple[PublicHttpClient, FakeBackend]:
        backend = FakeBackend(
            [
                FakeStream(
                    response_chunks(
                        body=bomb,
                        headers=[("Content-Encoding", "gzip"), ("Content-Type", "text/plain")],
                    )
                )
            ]
        )
        client = PublicHttpClient(
            authorize=make_web_authorizer(lambda: permissions),  # 生产授权闭包
            limits=WebLimits(max_bytes=4096),
            resolver=FakeResolver([PUBLIC_IP]),
            backend=backend,
            environ={},
        )
        return client, backend

    store = Store(workspace / "store.db")
    try:
        # ① network allow:decide 放行 ⇒ dispatch ⇒ 流式解码超限 ⇒ 有界错误码。
        allow_policy = PermissionPolicy(network=NetworkPolicy(decision="allow"))
        permissions = PermissionContext(workspace=str(workspace), policy=allow_policy, mode="on-request")
        client, backend = _client(permissions)
        (tool,) = create_web_fetch_tools(client, limits=WebLimits(max_bytes=4096))
        # 授权闭包与管道消费**同一**权限快照对象(live 单源)。
        ctx = _exec_ctx(store, workspace, {"web_fetch": tool}, policy=allow_policy, permissions=permissions)
        result = asyncio.run(execute(ToolCall("c1", "web_fetch", {"url": "https://example.com/"}), ctx))
        assert result.status.value == "error"
        assert result.error_code == "web_fetch_failed:max_bytes_exceeded"
        assert len(backend.connects) == 1  # 真的触网(假 backend),不是批准前被拦
        items = store.read_turn_items("turn-1")
        assert any(i.kind == ItemKind.TOOL_CALL and i.payload.get("call_id") == "c1" for i in items)
        assert [r.payload.get("error_code") for r in _results(items, "c1")] == ["web_fetch_failed:max_bytes_exceeded"]

        # ② 默认 network ask:decide 第 2 步先于任何网络 ⇒ PAUSED + 审批请求,
        # 零连接(S8:URL 面 ask 在管道层生效,不是 deny-only 回调)。
        store2 = Store(workspace / "store2.db")
        try:
            default_policy = PermissionPolicy()
            permissions2 = PermissionContext(workspace=str(workspace), policy=default_policy, mode="on-request")
            client2, backend2 = _client(permissions2)
            (tool2,) = create_web_fetch_tools(client2, limits=WebLimits(max_bytes=4096))
            ctx2 = _exec_ctx(
                store2, workspace, {"web_fetch": tool2}, policy=default_policy, permissions=permissions2, turn_id="turn-2"
            )
            paused = asyncio.run(execute(ToolCall("c1", "web_fetch", {"url": "https://example.com/"}), ctx2))
            assert paused.status.value == "paused"
            assert backend2.connects == []
            items2 = store2.read_turn_items("turn-2")
            assert any(i.kind == ItemKind.APPROVAL_REQUEST for i in items2)
            assert not any(i.kind == ItemKind.TOOL_CALL for i in items2)
        finally:
            store2.close()
    finally:
        store.close()