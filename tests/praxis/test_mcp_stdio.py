"""Task 16:MCP stdio owner-task 生命周期 + S7 envp 修复(3B §5.10 主案)。

**分层(诚实记录:fake 不得掩盖生产路径断裂)**:
- **Tier A —— 真 seatbelt**(``probe_backend().available`` 才跑,darwin):
  ``compile_sandbox`` 真编译 profile,server 真的在沙箱内被 execve。S7 的
  安全语义(密钥只在 envp、宿主哨兵不进子进程、argv 无 ``KEY=VALUE``)、
  沙箱真实性(``fs_probe`` 读工作区外宿主文件被内核拒绝)与取消/超时后的
  进程回收都在这一层证明。
- **Tier B —— 真子进程 + passthrough 前缀**(全平台):``SandboxSpec`` 的
  ``argv_prefix`` 取 ``("/usr/bin/env",)``——一个**真的 exec 透传程序**
  (不是假 spec 对象:SandboxSpec 要求非空前缀,``/usr/bin/env`` 无 operand
  时原样 exec 后续命令并保留 envp)。它只验证 owner-task 机制、argv 拼装与
  envp 语义,**不构成安全边界**;边界由 Tier A 证明。两层的 ``env_builder``
  都是生产的 ``child_env``(不是 fake),所以"最小环境"语义两层同源。
- **fail-closed**(``sandbox_spec is None`` ⇒ ``SandboxRefusal``)全平台必跑。

**证据分工(防止过度信任 ps)**:六路证据=运行面证明 envp 送达;argv 形态
回归由构造期护栏(``_argv_violations`` 直接判 ``_stdio_parameters`` 的产物)
+ T20 静态规则负责。原因是实测确认的盲区:把旧病灶形态
``/usr/bin/env -i SECRET_MCP_KEY=<secret> … cmd`` 原样派生,**六路断言全绿**
——密钥暴露发生在 exec 窗口期,而 server 自报的 ``sys.argv`` 与 ``ps`` 看到的
都是内层命令的 argv(外层 env 已 exec 掉),稳态行为探针看不见它。因此另有
一条负向构造用例证明护栏对旧形态确实报警(不是空转)。

真实证据面:真 ``mcp`` SDK(client + server)、真子进程、真 ``ps`` 复查。
同步点一律"等驱动任务/等进程消失"(M9),不用固定时长 sleep 当同步手段;
每个用例结束都必须无 owner task 残留、无孤儿进程(2026-11-13 事故教训)。
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import uuid
from collections.abc import Iterator
from contextlib import AsyncExitStack, contextmanager
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

from praxis.adapters.mcp.config import StdioServerConfig
# _stdio_parameters 是刻意直取的私有构造点:argv 形态护栏必须钉在"重新引入
# 旧病灶的那一行"上,而不是只靠运行面探针(见模块 docstring 的盲区说明)。
from praxis.adapters.mcp.lifecycle import McpConnection, _stdio_parameters, owned_stdio_connection
from praxis.adapters.sandbox.base import probe_backend
from praxis.adapters.sandbox.compile import compile_sandbox, finalize_sandbox
from praxis.protocol.permission import PermissionContext
from praxis.protocol.policy import PermissionPolicy
from praxis.protocol.sandbox import SandboxRefusal, SandboxSpec
from praxis.protocol.tools import ResolvedUse, ToolEffect, ToolTarget
from praxis.tools.builtins.child_env import child_env

_AVAILABLE = probe_backend().available
_FIXTURE = Path(__file__).resolve().parent / "fixtures" / "mcp_probe_server.py"
_OWNER_PREFIX = "mcp-owner:"
_NO_SANDBOX_REASON = (
    "mcp stdio server requires a sandbox backend; configure bwrap/seatbelt or run on a supported platform"
)
_MISSING_BINARY = "/nonexistent-praxis-mcp-probe-binary"
_ENV_PAIR = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")


# ---------------------------------------------------------------------------
# 脚手架:spec/config/探针(ps 与 tool result 两路独立证据)
# ---------------------------------------------------------------------------


def _token() -> str:
    return "probe-" + uuid.uuid4().hex


def _secret() -> str:
    return "sk-mcp-" + uuid.uuid4().hex


@contextmanager
def _spec(workspace: Path, *, real: bool) -> Iterator[SandboxSpec]:
    """Tier A 真编译 seatbelt;Tier B 真 exec 透传前缀(见模块 docstring)。"""
    if not real:
        scratch = workspace / "scratch"
        scratch.mkdir(exist_ok=True)
        yield SandboxSpec(
            backend="test-passthrough",
            argv_prefix=("/usr/bin/env",),
            scratch_dir=str(scratch),
        )
        return
    permissions = PermissionContext(
        workspace=str(workspace),
        policy=PermissionPolicy(),
        mode="bypass",
        sandbox=probe_backend(),
    )
    use = ResolvedUse(
        effects=frozenset({ToolEffect.EXECUTE_PROCESS}),
        targets=(ToolTarget(kind="path", value=str(workspace)),),
    )
    spec = compile_sandbox(permissions, use, workspace)
    try:
        yield spec
    finally:
        finalize_sandbox(spec, workspace)


def _config(
    workspace: Path,
    token: str,
    *,
    secret: str,
    extra_args: tuple[str, ...] = (),
    command: str | None = None,
    startup_timeout_seconds: float = 10.0,
    name: str = "probe",
) -> StdioServerConfig:
    return StdioServerConfig(
        name=name,
        command=sys.executable if command is None else command,
        args=("-I", str(_FIXTURE), "--token", token, *extra_args),
        env={"SECRET_MCP_KEY": secret, "LC_ALL": "C.UTF-8"},
        cwd=str(workspace),
        tools_allowlist=frozenset(),
        allow_all_tools=True,
        startup_timeout_seconds=startup_timeout_seconds,
    )


def _connect(
    config: StdioServerConfig,
    workspace: Path,
    spec: SandboxSpec | None,
    *,
    close_timeout_seconds: float = 5.0,
) -> Any:
    return owned_stdio_connection(
        config,
        workspace=workspace,
        sandbox_spec=spec,
        env_builder=child_env,
        close_timeout_seconds=close_timeout_seconds,
    )


def _argv_violations(args: list[str]) -> list[str]:
    """构造期护栏:argv 里不得出现 env 重置形态(S7 回归的确定性守卫)。

    三条判据(controller 钉死):零 ``KEY=VALUE``、零 ``/usr/bin/env``、零 ``-i``。
    ``-i`` 是粗判据(合法命令也可能带 ``-i``),本护栏只作用于本夹具的构造产物;
    全仓静态禁令归 T20。违规描述只点名键名,不回显值(秘密不进失败输出)。
    """
    violations = [f"env pair in argv: {arg.split('=', 1)[0]}=" for arg in args if _ENV_PAIR.match(arg)]
    if "/usr/bin/env" in args:
        violations.append("env reset binary in argv: /usr/bin/env")
    if "-i" in args:
        violations.append("env -i flag in argv")
    return violations


def _leaves(exc: BaseException) -> list[BaseException]:
    """摊平 exception group,取叶子异常。

    实测形状(mcp 1.27 + anyio 4):启动超时经 ``stdio_client`` 与
    ``ClientSession`` 两层 task group,到调用方手里是
    ``ExceptionGroup(ExceptionGroup(TimeoutError))``。断言只钉"叶子含
    TimeoutError",不把包装层数当契约;**T19 的 diagnostic 文案必须同样
    摊平**,否则超时只会显示 "unhandled errors in a TaskGroup"。
    """
    if isinstance(exc, BaseExceptionGroup):
        return [leaf for sub in exc.exceptions for leaf in _leaves(sub)]
    return [exc]


async def _call_text(conn: McpConnection, tool: str, arguments: dict[str, Any]) -> str:
    result = await conn.session.call_tool(tool, arguments)
    assert result.isError is not True, result.content
    return str(result.content[0].text)


def _ps_rows() -> list[str]:
    completed = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    return completed.stdout.splitlines()


def _ps_lines(token: str) -> list[str]:
    return [row for row in _ps_rows() if token in row]


def _owner_tasks() -> list[asyncio.Task[Any]]:
    return [task for task in asyncio.all_tasks() if task.get_name().startswith(_OWNER_PREFIX)]


async def _wait_owner(name: str, timeout: float = 5.0) -> asyncio.Task[Any]:
    """同步点:等 owner task 真的出现(不是 sleep 猜时机)。"""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        for task in asyncio.all_tasks():
            if task.get_name() == name:
                return task
        if loop.time() > deadline:
            pytest.fail(f"owner task {name} never started")
        await asyncio.sleep(0.01)


async def _wait_process_gone(token: str, timeout: float = 10.0) -> None:
    """同步点:等子进程(组)真的消失;超时即孤儿,直接失败。"""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        rows = _ps_lines(token)
        if not rows:
            return
        if loop.time() > deadline:
            pytest.fail(f"orphan MCP server still alive: {rows}")
        await asyncio.sleep(0.05)


async def _assert_s7_evidence(
    conn: McpConnection,
    *,
    secret: str,
    token: str,
    scratch: Path,
    workspace: Path,
    real_sandbox: bool,
) -> None:
    """S7 主案:密钥只经 envp;argv 与宿主环境两路都干净。"""
    probe = json.loads(await _call_text(conn, "env_probe", {}))
    assert probe["token"] == token
    # ① 密钥经 execve envp 到达 server(工具结果即证据)。
    assert probe["secret"] == secret
    assert probe["env"]["SECRET_MCP_KEY"] == secret
    # ② 宿主哨兵/宿主配置变量不进子进程(child_env 白名单 + SDK 只继承 6 个安全键)。
    assert probe["sentinel"] is None
    assert "PRAXIS_HOST_ONLY_SENTINEL" not in probe["env"]
    assert "PRAXIS_CONFIG_DIR" not in probe["env"]
    # ③ env_builder 语义生效,且 config.env 覆盖 builder(拼装顺序 {**builder, **config})。
    assert probe["env"]["HOME"] == os.path.realpath(scratch)
    assert probe["env"]["LC_ALL"] == "C.UTF-8"
    # ④ server 自报 argv:零 KEY=VALUE、零密钥明文。
    argv_text = " ".join(probe["argv"])
    assert "SECRET_MCP_KEY" not in argv_text
    assert secret not in argv_text
    assert token in probe["argv"]
    assert not any(part.startswith("SECRET_MCP_KEY=") for part in probe["argv"])
    # ⑤ 测试侧独立复查(darwin ps 探针):本实例与全宿主 argv 都不含密钥。
    rows = _ps_lines(token)
    assert rows, "connected server must be visible in ps"
    for row in rows:
        assert "SECRET_MCP_KEY" not in row
        assert secret not in row
    assert secret not in "\n".join(_ps_rows())
    # ⑥ 沙箱真实性(行为证据,不看 spec 对象自证):工作区内可读;
    #    工作区外宿主文件在真 seatbelt 下必须被内核拒绝,passthrough 下读得到。
    inside = workspace / "inside.txt"
    inside.write_text("ok", encoding="utf-8")
    outside = workspace.parent / "outside.txt"
    outside.write_text("host-only", encoding="utf-8")
    allowed = json.loads(await _call_text(conn, "fs_probe", {"path": str(inside)}))
    other = json.loads(await _call_text(conn, "fs_probe", {"path": str(outside)}))
    assert allowed["read_ok"] is True
    assert other["read_ok"] is (not real_sandbox)


# ---------------------------------------------------------------------------
# Tier B:owner-task 机制 + envp 语义(真子进程,passthrough 前缀,全平台)
# ---------------------------------------------------------------------------


def test_ready_yields_listed_tools_and_named_owner_task(workspace: Path) -> None:
    token, secret = _token(), _secret()

    async def scenario() -> None:
        with _spec(workspace, real=False) as spec:
            config = _config(workspace, token, secret=secret)
            async with _connect(config, workspace, spec) as conn:
                assert isinstance(conn, McpConnection)
                assert conn.server_name == "probe"
                # initialize + list_tools 在 ready 之前完成。
                assert tuple(tool.name for tool in conn.tools) == ("echo", "env_probe", "fs_probe")
                assert [task.get_name() for task in _owner_tasks()] == ["mcp-owner:probe"]
                assert await _call_text(conn, "echo", {"message": "hi"}) == "echo: hi"
                with pytest.raises(FrozenInstanceError):
                    conn.server_name = "other"
            assert _owner_tasks() == []
        await _wait_process_gone(token)

    asyncio.run(scenario())


def test_s7_secret_travels_via_envp_only_passthrough(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    token, secret = _token(), _secret()
    monkeypatch.setenv("PRAXIS_HOST_ONLY_SENTINEL", "host-only-" + uuid.uuid4().hex)

    async def scenario() -> None:
        with _spec(workspace, real=False) as spec:
            config = _config(workspace, token, secret=secret)
            async with _connect(config, workspace, spec) as conn:
                await _assert_s7_evidence(
                    conn,
                    secret=secret,
                    token=token,
                    scratch=Path(spec.scratch_dir),
                    workspace=workspace,
                    real_sandbox=False,
                )
            assert _owner_tasks() == []
        await _wait_process_gone(token)

    asyncio.run(scenario())


def test_construction_argv_carries_no_env_reset(workspace: Path) -> None:
    """S7 构造期护栏:在 ``_stdio_parameters`` 的产物上直接判 argv 形态。

    运行面六路证据只证明 envp 送达,看不见旧病灶的 exec 窗口期暴露(见模块
    docstring);这条是"重新引入旧形态即红"的确定性回归守卫。
    """
    token, secret = _token(), _secret()
    with _spec(workspace, real=False) as spec:
        config = _config(workspace, token, secret=secret)
        params = _stdio_parameters(config, workspace=workspace, spec=spec, env_builder=child_env)
        assert _argv_violations(params.args) == []
        # 裁决钉死的拼装形状逐字核对。
        assert params.command == spec.argv_prefix[0]
        assert params.args == [*spec.argv_prefix[1:], config.command, *config.args]
        assert params.env == {**child_env(Path(spec.scratch_dir), workspace), **config.env}
        assert params.env["SECRET_MCP_KEY"] == secret
        assert params.cwd == config.cwd
        assert secret not in " ".join(params.args)


def test_guard_flags_the_legacy_env_reset_shape(workspace: Path) -> None:
    """负向构造:旧病灶形态必须被护栏报警(证明护栏不是空转)。

    只构造 argv,**绝不派生**——旧形态会把密钥写进真实命令行。同一份输入下
    正解产物必须干净,两者对照即 S7 的构造期证据。
    """
    token, secret = _token(), _secret()
    with _spec(workspace, real=False) as spec:
        config = _config(workspace, token, secret=secret)
        environment = {**child_env(Path(spec.scratch_dir), workspace), **config.env}
        legacy_args = [
            *spec.argv_prefix[1:],
            "/usr/bin/env",
            "-i",
            *(f"{key}={value}" for key, value in sorted(environment.items())),
            config.command,
            *config.args,
        ]
        violations = _argv_violations(legacy_args)
        assert "env pair in argv: SECRET_MCP_KEY=" in violations
        assert "env reset binary in argv: /usr/bin/env" in violations
        assert "env -i flag in argv" in violations
        assert secret in " ".join(legacy_args)  # 旧形态确实把密钥明文写进 argv
        params = _stdio_parameters(config, workspace=workspace, spec=spec, env_builder=child_env)
        assert _argv_violations(params.args) == []
        assert secret not in " ".join(params.args)


def test_missing_sandbox_refuses_without_spawning(workspace: Path) -> None:
    """R-PF-5:无沙箱后端一律 fail-closed,绝不裸宿主派生(同 search_text 先例)。"""
    token = _token()

    async def scenario() -> None:
        config = _config(workspace, token, secret=_secret())
        with pytest.raises(SandboxRefusal) as excinfo:
            async with _connect(config, workspace, None):
                pytest.fail("must not connect without a sandbox backend")
        assert str(excinfo.value) == _NO_SANDBOX_REASON
        assert _owner_tasks() == []
        assert _ps_lines(token) == []

    asyncio.run(scenario())


def test_hung_handshake_times_out_without_task_or_process_leak(workspace: Path) -> None:
    token = _token()

    async def scenario() -> None:
        with _spec(workspace, real=False) as spec:
            config = _config(
                workspace,
                token,
                secret=_secret(),
                extra_args=("--startup-delay", "5"),
                startup_timeout_seconds=0.5,
            )
            with pytest.raises((TimeoutError, BaseExceptionGroup)) as excinfo:
                async with _connect(config, workspace, spec):
                    pytest.fail("hung handshake must not become ready")
            assert any(isinstance(leaf, TimeoutError) for leaf in _leaves(excinfo.value))
            assert _owner_tasks() == []
        await _wait_process_gone(token)

    asyncio.run(scenario())


def test_caller_cancel_before_ready_reaps_owner_and_child(workspace: Path) -> None:
    token = _token()

    async def scenario() -> None:
        with _spec(workspace, real=False) as spec:
            config = _config(
                workspace,
                token,
                secret=_secret(),
                extra_args=("--startup-delay", "5"),
            )

            async def opener() -> None:
                async with _connect(config, workspace, spec):
                    pytest.fail("must not become ready before the caller cancels")

            task = asyncio.create_task(opener(), name="opener")
            await _wait_owner("mcp-owner:probe")
            task.cancel()
            outcomes = await asyncio.gather(task, return_exceptions=True)
            assert isinstance(outcomes[0], asyncio.CancelledError)
            assert _owner_tasks() == []
        await _wait_process_gone(token)

    asyncio.run(scenario())


def test_caller_cancel_after_ready_closes_owner_and_child(workspace: Path) -> None:
    """M2:取消落在 yield body 内(已就绪)⇒ 同一姿势有界关闭,无 owner/孤儿。

    同步点是 body 自己 set 的 event(不是 sleep 猜时机);body 里的长 sleep 只是
    取消的落点。与 ready 前取消的区别:此时 ``ready`` 已完成,关闭走
    ``stop.set()`` → owner 自己退出栈,而不是 ``task.cancel()`` 腿。
    """
    token = _token()

    async def scenario() -> None:
        with _spec(workspace, real=False) as spec:
            config = _config(workspace, token, secret=_secret())
            in_body = asyncio.Event()

            async def opener() -> None:
                async with _connect(config, workspace, spec) as conn:
                    assert await _call_text(conn, "echo", {"message": "ready"}) == "echo: ready"
                    in_body.set()
                    await asyncio.sleep(30)

            task = asyncio.create_task(opener(), name="opener")
            await asyncio.wait_for(in_body.wait(), 10.0)
            task.cancel()
            outcomes = await asyncio.gather(task, return_exceptions=True)
            assert isinstance(outcomes[0], asyncio.CancelledError)
            assert _owner_tasks() == []
        await _wait_process_gone(token)

    asyncio.run(scenario())


@pytest.mark.parametrize("failure", ["missing-binary", "dies-in-handshake"])
def test_one_server_failure_leaves_the_other_usable(workspace: Path, failure: str) -> None:
    """单 owner 只关自己的栈:一个 server 起不来,另一个照常可用(无自动重试)。"""
    good_token, bad_token = _token(), _token()

    async def scenario() -> None:
        with _spec(workspace, real=False) as spec:
            good = _config(workspace, good_token, secret=_secret())
            bad = _config(
                workspace,
                bad_token,
                secret=_secret(),
                command=_MISSING_BINARY if failure == "missing-binary" else None,
                extra_args=("--exit-immediately",) if failure == "dies-in-handshake" else (),
                startup_timeout_seconds=5.0,
                name="broken",
            )
            async with _connect(good, workspace, spec) as healthy:
                with pytest.raises(Exception) as excinfo:  # noqa: B017 — 失败面按传输异常原样上抛,T19 逐 server 捕获
                    async with _connect(bad, workspace, spec):
                        pytest.fail("broken server must not become ready")
                assert not isinstance(excinfo.value, asyncio.CancelledError)
                # 只有坏 server 的 owner 消失;健康 server 的 owner 仍在。
                assert [task.get_name() for task in _owner_tasks()] == ["mcp-owner:probe"]
                assert await _call_text(healthy, "echo", {"message": "still-alive"}) == "echo: still-alive"
            assert _owner_tasks() == []
        await _wait_process_gone(good_token)
        await _wait_process_gone(bad_token)

    asyncio.run(scenario())


def test_close_timeout_bounds_shutdown_and_reaps_child(workspace: Path) -> None:
    """关闭挂起(stdin EOF 后仍不退)⇒ close_timeout 兜住,cancel+gather 后无孤儿。"""
    token = _token()

    async def scenario() -> None:
        with _spec(workspace, real=False) as spec:
            config = _config(workspace, token, secret=_secret(), extra_args=("--exit-delay", "5"))
            async with _connect(config, workspace, spec, close_timeout_seconds=0.2) as conn:
                assert await _call_text(conn, "echo", {"message": "bye"}) == "echo: bye"
            assert _owner_tasks() == []
        await _wait_process_gone(token)

    asyncio.run(scenario())


def test_close_from_a_different_task_is_safe(workspace: Path) -> None:
    """owner task 的存在理由:SDK 的 AnyIO cancel scope 是**任务局部**的。

    在 task A 进入连接、在 task B 关闭,也必须干净退出——裸 AsyncExitStack
    跨 task 关闭会报 "exit cancel scope in a different task" 并泄漏子进程。
    """
    token = _token()

    async def scenario() -> None:
        with _spec(workspace, real=False) as spec:
            config = _config(workspace, token, secret=_secret())
            stack = AsyncExitStack()
            conn = await stack.enter_async_context(_connect(config, workspace, spec))
            assert await _call_text(conn, "echo", {"message": "cross-task"}) == "echo: cross-task"
            await asyncio.wait_for(asyncio.create_task(stack.aclose(), name="closer"), 10.0)
            assert _owner_tasks() == []
        await _wait_process_gone(token)

    asyncio.run(scenario())


# ---------------------------------------------------------------------------
# Tier A:真 seatbelt(darwin;安全边界与生产姿势的进程回收)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _AVAILABLE, reason="real seatbelt needs a sandbox backend")
def test_s7_secret_travels_via_envp_only_real_seatbelt(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    token, secret = _token(), _secret()
    monkeypatch.setenv("PRAXIS_HOST_ONLY_SENTINEL", "host-only-" + uuid.uuid4().hex)

    async def scenario() -> None:
        with _spec(workspace, real=True) as spec:
            assert spec.backend == "seatbelt"
            assert spec.argv_prefix[0] == "/usr/bin/sandbox-exec"
            config = _config(workspace, token, secret=secret)
            # 生产姿势同样过构造期护栏:真 seatbelt 前缀下 argv 也只有前缀+命令。
            params = _stdio_parameters(config, workspace=workspace, spec=spec, env_builder=child_env)
            assert _argv_violations(params.args) == []
            async with _connect(config, workspace, spec) as conn:
                await _assert_s7_evidence(
                    conn,
                    secret=secret,
                    token=token,
                    scratch=Path(spec.scratch_dir),
                    workspace=workspace,
                    real_sandbox=True,
                )
            assert _owner_tasks() == []
        await _wait_process_gone(token)

    asyncio.run(scenario())


@pytest.mark.skipif(not _AVAILABLE, reason="real seatbelt needs a sandbox backend")
def test_real_seatbelt_hung_handshake_reaps_sandboxed_child(workspace: Path) -> None:
    token = _token()

    async def scenario() -> None:
        with _spec(workspace, real=True) as spec:
            config = _config(
                workspace,
                token,
                secret=_secret(),
                extra_args=("--startup-delay", "5"),
                startup_timeout_seconds=0.5,
            )
            with pytest.raises((TimeoutError, BaseExceptionGroup)) as excinfo:
                async with _connect(config, workspace, spec):
                    pytest.fail("hung handshake must not become ready")
            assert any(isinstance(leaf, TimeoutError) for leaf in _leaves(excinfo.value))
            assert _owner_tasks() == []
        await _wait_process_gone(token)

    asyncio.run(scenario())


@pytest.mark.skipif(not _AVAILABLE, reason="real seatbelt needs a sandbox backend")
def test_real_seatbelt_caller_cancel_before_ready_reaps_child(workspace: Path) -> None:
    token = _token()

    async def scenario() -> None:
        with _spec(workspace, real=True) as spec:
            config = _config(
                workspace,
                token,
                secret=_secret(),
                extra_args=("--startup-delay", "5"),
            )

            async def opener() -> None:
                async with _connect(config, workspace, spec):
                    pytest.fail("must not become ready before the caller cancels")

            task = asyncio.create_task(opener(), name="opener")
            await _wait_owner("mcp-owner:probe")
            task.cancel()
            outcomes = await asyncio.gather(task, return_exceptions=True)
            assert isinstance(outcomes[0], asyncio.CancelledError)
            assert _owner_tasks() == []
        await _wait_process_gone(token)

    asyncio.run(scenario())
