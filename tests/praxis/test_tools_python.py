"""Task 16:run_python 内置工具 + R-T16-2 子进程环境最小化回归。

覆盖:
- schema:code XOR script_path(二选一必填)、脚本落点围栏、working_dir 围栏;
- 先 compile 语法校验(执行前拒绝,不产生副作用);
- ``-I`` 隔离(``sys.flags.ignore_environment`` 为 True);
- 进程组超时(SIGTERM→宽限→SIGKILL,不留活子进程);
- stdout/stderr 各自 ≤16_000 头尾保留,截断标记逐字一致;
- 非零退出 → python_failed;超时 → TIMEOUT/timeout_cancelled;
- R-T16-2:宿主环境变量(含凭据)在沙箱命令与 run_python 中一律不可见;
  最小 PATH 仍能找到工具链;GIT_CONFIG_GLOBAL=/dev/null 生效。

真实沙箱用例一律 ``skipif(not probe_backend().available)``;其他用例走
``ToolExecContext()`` 直连路径(同一份 child_env,不存在透传分支)。
仓库无 pytest-asyncio:asyncio.run 包装。
"""

from __future__ import annotations

import asyncio
import os
import re
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from praxis.adapters.sandbox.base import probe_backend
from praxis.adapters.sandbox.compile import compile_sandbox, finalize_sandbox
from praxis.protocol.permission import PermissionContext
from praxis.protocol.policy import PathRule, PermissionPolicy
from praxis.protocol.sandbox import SandboxSpec, ToolExecContext
from praxis.protocol.tools import (
    CancellationMode,
    InterruptBehavior,
    ResolvedUse,
    ResultStatus,
    ToolEffect,
    ToolTarget,
    ToolValidationError,
)
from praxis.tools.builtins.child_env import child_env
from praxis.tools.builtins.command import create_bash_tool
from praxis.tools.builtins.python_tool import create_run_python_tool

_AVAILABLE = probe_backend()
_TRUNCATION_MARKER = "\n... output truncated; preserved head and tail ...\n"
_SECRET_VAR = "PRAXIS_TEST_SECRET"


def run(tool: Any, arguments: Mapping[str, Any], *, spec: SandboxSpec | None = None) -> Any:
    return asyncio.run(tool.run(arguments, ToolExecContext(sandbox=spec)))


def sandbox_spec(workspace: Path) -> SandboxSpec:
    """编译一份真实 seatbelt spec;临时 workspace 需显式提供受信解释器链接。

    ``_trusted_toolchain_roots`` 只从 ``<workspace>/.venv/bin/python`` 解析
    受信 CPython;测试 workspace 是 tmp_path,不链接则 sys.executable 在
    seatbelt 中不可读(生产工作区本就有 .venv)。
    """
    venv_bin = workspace / ".venv" / "bin"
    venv_bin.mkdir(parents=True, exist_ok=True)
    link = venv_bin / "python"
    if not link.exists():
        link.symlink_to(Path(sys.executable).resolve())
    context = PermissionContext(
        workspace=str(workspace),
        # 测试 workspace 位于 tmp_path,而实现按契约执行 sys.executable(本项目
        # venv 的解释器)。生产工作区自带 .venv 因而天然可读;测试显式授予
        # 解释器目录只读访问,模拟同一形态。
        policy=PermissionPolicy(
            filesystem=(
                PathRule(path=str(Path(sys.executable).parent), access="read"),
            )
        ),
        mode="bypass",
    )
    use = ResolvedUse(
        effects=frozenset({ToolEffect.READ_WORKSPACE, ToolEffect.EXECUTE_PROCESS}),
        targets=(ToolTarget(kind="path", value=str(workspace)),),
    )
    return compile_sandbox(context, use, workspace)


def assert_dead(pidfile: Path, budget: float = 8.0) -> None:
    pid = int(pidfile.read_text().strip())
    deadline = time.monotonic() + budget
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        assert time.monotonic() < deadline, f"process {pid} survived; expected group convergence"
        time.sleep(0.1)


# ---------------------------------------------------------------------------
# schema / 围栏 / 校验前置
# ---------------------------------------------------------------------------


def test_code_xor_script_path(tmp_path: Path) -> None:
    tool = create_run_python_tool(tmp_path)
    for arguments in ({"code": "print(1)", "script_path": "s.py"}, {}):
        with pytest.raises(ToolValidationError) as caught:
            tool.validate_input(arguments)
        assert caught.value.message == "exactly one of code/script_path is required"


def test_tool_definition_shape(tmp_path: Path) -> None:
    tool = create_run_python_tool(tmp_path)
    definition = tool.definition
    assert definition.name == "run_python"
    properties = definition.input_schema["properties"]
    assert "workspace_write" not in properties
    assert "network" not in properties
    assert "timeout_seconds" in properties
    assert tool.timeout_seconds == 605
    assert tool.max_model_output_bytes == 150_000
    assert tool.idempotent is False
    assert tool.concurrency_safe is False
    assert tool.cancellation_mode is CancellationMode.MANAGED_PROCESS
    assert tool.interrupt_behavior is InterruptBehavior.CANCEL
    assert tool.static_effects == frozenset(
        {ToolEffect.READ_WORKSPACE, ToolEffect.EXECUTE_PROCESS}
    )
    use = tool.resolve_use({"code": "print(1)"})
    assert use.effects == frozenset({ToolEffect.READ_WORKSPACE, ToolEffect.EXECUTE_PROCESS})
    assert use.targets == (ToolTarget(kind="path", value=str(tmp_path)),)
    (tmp_path / "sub").mkdir()
    use = tool.resolve_use({"code": "print(1)", "working_dir": "sub"})
    assert use.targets == (ToolTarget(kind="path", value=str(tmp_path / "sub")),)


def test_timeout_seconds_rejects_null_and_out_of_range(tmp_path: Path) -> None:
    tool = create_run_python_tool(tmp_path)
    for value in (None, 0, 601):
        with pytest.raises(ToolValidationError):
            tool.validate_input({"code": "print(1)", "timeout_seconds": value})
    tool.validate_input({"code": "print(1)", "timeout_seconds": 600})


def test_syntax_error_rejected_before_execution(tmp_path: Path) -> None:
    tool = create_run_python_tool(tmp_path)
    result = run(tool, {"code": "open('side-effect', 'w').write('ran')\ndef broken(:\n"})
    assert result.status is ResultStatus.ERROR
    assert result.is_error
    assert result.error_code == "invalid_input"
    assert not (tmp_path / "side-effect").exists()


def test_script_path_must_stay_inside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("print('outside')", encoding="utf-8")
    tool = create_run_python_tool(workspace)
    result = run(tool, {"script_path": str(outside)})
    assert result.error_code == "invalid_input"
    assert "outside the workspace" in result.error_message


def test_working_dir_must_be_inside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    tool = create_run_python_tool(workspace)
    result = run(tool, {"code": "print(1)", "working_dir": "missing"})
    assert result.error_code == "invalid_input"
    with pytest.raises(ToolValidationError):
        tool.resolve_use({"code": "print(1)", "working_dir": "../escape"})


# ---------------------------------------------------------------------------
# 执行语义(直连路径;child_env 与沙箱路径同一份)
# ---------------------------------------------------------------------------


def test_runs_code_with_isolated_interpreter(tmp_path: Path) -> None:
    result = run(
        create_run_python_tool(tmp_path),
        {"code": "import sys; print(bool(sys.flags.ignore_environment))"},
    )
    assert result.status is ResultStatus.OK
    assert result.text.strip() == "True"
    assert re.fullmatch(r"\d+\.\d+\.\d+", str(result.metadata["python_version"]))
    assert result.metadata["exit_code"] == 0
    assert result.metadata["timed_out"] is False
    assert result.metadata["duration_ms"] >= 0


def test_script_path_runs_with_working_dir(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "show.py").write_text(
        "import os\nprint(os.getcwd())\n", encoding="utf-8"
    )
    result = run(
        create_run_python_tool(tmp_path),
        {"script_path": "sub/show.py", "working_dir": "sub"},
    )
    assert result.status is ResultStatus.OK
    assert result.text.strip() == str((tmp_path / "sub").resolve())


def test_nonzero_exit_is_python_failed(tmp_path: Path) -> None:
    result = run(
        create_run_python_tool(tmp_path),
        {"code": "import sys; print('boom'); sys.exit(3)"},
    )
    assert result.status is ResultStatus.ERROR
    assert result.is_error
    assert result.error_code == "python_failed"
    assert result.error_message == (
        "Python code exited with status 3; inspect stdout/stderr and "
        "change the code rather than retrying unchanged"
    )
    assert result.metadata["exit_code"] == 3
    assert "boom" in result.text


def test_output_truncation_preserves_head_and_tail(tmp_path: Path) -> None:
    result = run(
        create_run_python_tool(tmp_path),
        {"code": "import sys; sys.stdout.write('H' * 10_000 + 'M' * 100_000 + 'T' * 10_000)"},
    )
    assert result.status is ResultStatus.OK
    assert result.truncated
    assert _TRUNCATION_MARKER in result.text
    assert result.text.startswith("H")
    assert result.text.endswith("T")
    assert len(result.text.encode("utf-8")) == 16_000


def test_timeout_kills_process_group(tmp_path: Path) -> None:
    pidfile = tmp_path / "pid"
    result = run(
        create_run_python_tool(tmp_path),
        {
            "code": (
                "import os, time\n"
                "open('pid', 'w').write(str(os.getpid()))\n"
                "time.sleep(30)\n"
            ),
            "timeout_seconds": 1,
        },
    )
    assert result.status is ResultStatus.TIMEOUT
    assert result.is_error
    assert result.error_code == "timeout_cancelled"
    assert result.metadata["timed_out"] is True
    assert_dead(pidfile)


def test_temporary_script_is_removed(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    spec = SandboxSpec(
        backend="seatbelt",
        argv_prefix=("/bin/sh", "-c", 'exec "$0" "$@"'),
        scratch_dir=str(scratch),
    )
    # 用 /bin/sh 直通前缀模拟 spec 的 scratch 定位(不触发 seatbelt)。
    result = run(
        create_run_python_tool(tmp_path),
        {"code": "print('scratch')"},
        spec=spec,
    )
    assert result.status is ResultStatus.OK
    assert result.text.strip() == "scratch"
    assert list(scratch.iterdir()) == []


# ---------------------------------------------------------------------------
# R-T16-2:子进程环境最小化(安全必修)
# ---------------------------------------------------------------------------


def test_child_env_is_allowlist_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_SECRET_VAR, "leak-me")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    env = child_env(scratch, tmp_path)
    assert _SECRET_VAR not in env
    assert env["HOME"] == os.path.realpath(scratch)
    assert env["TMPDIR"] == os.path.realpath(scratch)
    assert env["TEMP"] == os.path.realpath(scratch)
    assert env["TMP"] == os.path.realpath(scratch)
    assert env["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert env["LC_ALL"] == "C"
    assert os.environ["PATH"] not in env["PATH"].split(os.pathsep)
    assert env["PATH"].split(os.pathsep) == [
        os.path.join(os.path.realpath(tmp_path), ".venv", "bin"),
        os.path.join(os.path.realpath(tmp_path), "node_modules", ".bin"),
        "/opt/homebrew/bin",
        "/usr/local/bin",
        "/usr/bin",
        "/bin",
    ]


def test_direct_path_has_no_host_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(_SECRET_VAR, "leak-me")
    bash = run(create_bash_tool(tmp_path), {"command": f"printenv {_SECRET_VAR}"})
    assert bash.metadata["exit_code"] != 0
    assert "leak-me" not in bash.text
    python = run(
        create_run_python_tool(tmp_path),
        {"code": f"import os; print(os.environ.get({_SECRET_VAR!r}))"},
    )
    assert python.text.strip() == "None"


@pytest.mark.skipif(not _AVAILABLE.available, reason="no sandbox backend")
def test_sandboxed_commands_cannot_read_host_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(_SECRET_VAR, "leak-me")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    spec = sandbox_spec(workspace)
    try:
        bash = run(
            create_bash_tool(workspace),
            {"command": f"printenv {_SECRET_VAR}"},
            spec=spec,
        )
        assert bash.metadata["exit_code"] != 0
        assert "leak-me" not in bash.text
        python = run(
            create_run_python_tool(workspace),
            {"code": f"import os; print(os.environ.get({_SECRET_VAR!r}))"},
            spec=spec,
        )
        assert python.metadata["exit_code"] == 0
        assert python.text.strip() == "None"
    finally:
        finalize_sandbox(spec, workspace)


@pytest.mark.skipif(not _AVAILABLE.available, reason="no sandbox backend")
def test_sandboxed_minimal_path_keeps_toolchain(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    spec = sandbox_spec(workspace)
    try:
        result = run(
            create_bash_tool(workspace),
            {"command": "python3 --version && git --version"},
            spec=spec,
        )
        assert result.metadata["exit_code"] == 0, result.text
        assert "Python 3" in result.text
        assert "git version" in result.text
    finally:
        finalize_sandbox(spec, workspace)


@pytest.mark.skipif(not _AVAILABLE.available, reason="no sandbox backend")
def test_sandboxed_git_cannot_read_host_global_config(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    spec = sandbox_spec(workspace)
    try:
        result = run(
            create_bash_tool(workspace),
            {
                "command": (
                    "echo GIT_CONFIG_GLOBAL=$GIT_CONFIG_GLOBAL; "
                    "git config --global --list"
                )
            },
            spec=spec,
        )
        text = result.text
        lines = [line for line in text.splitlines() if line]
        assert "GIT_CONFIG_GLOBAL=/dev/null" in lines
        # 宿主 ~/.gitconfig 存在(user.name 等),沙箱内 --list 必须为空:
        # GIT_CONFIG_GLOBAL=/dev/null 生效,宿主全局配置不可见。
        assert not any(
            line.startswith(("user.", "color.", "core.", "init.")) for line in lines
        )
    finally:
        finalize_sandbox(spec, workspace)
