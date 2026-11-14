"""Task 18 Step 1:旧工具行为对照矩阵(执行文档 §4.5;设计 §7.3/§8)。

纪律(brief + addenda)：

- 每个数值/错误码/消息都带**旧代码 file:line**出处；只读旧代码,不读旧测试；
- 阈值以 praxis 的**公开契约**钉住——工具 ``validate_input``(运行时真校验)、
  tool 的 ``definition``/``timeout_seconds``/``max_model_output_bytes``、以及
  真实执行行为(真子进程/真文件系统),不 import 任何模块私有常量；
- 有意差异显式断言而非省略：``run_python``/``apply_patch`` 不接收
  ``workspace_write``/``network``(D1 授权来自 policy)；``search_text`` 输出改为
  模型可读文本行 + ``metadata``(旧为 structured_content 主体)；``apply_patch``
  的失败提示语义一致但去掉 ``Codex`` 前缀(H-4:不携带第三方产品名)；
- B33(B33/findings 补充节;旧 ``shell.py:1089-1125``):praxis ``child_env``
  是唯一构造入口,宿主环境变量一律不继承;
- B20(旧 ``shell.py:822-840`` 把命令结束绑在管道 EOF):praxis 以主进程退出
  为判据,后台孙进程持写端时也不得绑架返回。

旧代码出处逐项：
- search 六上限：``agent_runtime/tools/builtins/search_worker.py:27-32``
  (旧 search.py:29/37/79-100 的输入上限同族)；
- inspect_data_file：``agent_runtime/tools/builtins/data.py:34-37``；
- apply_patch：``agent_runtime/tools/builtins/codex_patch.py:47``(30_000)、
  ``:82``(patch_failed)、``:104``("not transactional")；
- update_plan：``agent_runtime/tools/builtins/planning.py:46-91``；
- bash：``agent_runtime/tools/builtins/shell.py:53-54``(16_000/600)、
  ``:88-108``(默认 120、yield ≤60)、``:322``(tool timeout 605)；
- manage_command：``agent_runtime/tools/builtins/command_sessions.py:39``(32)、
  ``:52-56``(wait ≤60、read ≤64_000)；
- run_python：``agent_runtime/tools/builtins/shell.py:201``(100_000)、
  ``:228``(≤600)、``:56``(头尾截断标记文案)。
"""

from __future__ import annotations

import asyncio
import os
import re
import subprocess
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from praxis.protocol.sandbox import SandboxSpec, ToolExecContext
from praxis.protocol.tools import Tool, ToolValidationError
from praxis.tools.builtins.child_env import child_env
from praxis.tools.builtins.command import create_bash_tool
from praxis.tools.builtins.command_sessions import (
    CommandSessionError,
    CommandSessions,
    create_manage_command_tool,
)
from praxis.tools.builtins.data import create_inspect_data_file_tool
from praxis.tools.builtins.patch import create_apply_patch_tool
from praxis.tools.builtins.planning import create_update_plan_tool
from praxis.tools.builtins.python_tool import create_run_python_tool
from praxis.tools.builtins.search import create_search_text_tool
from praxis.tools.builtins.search_worker import LIMITS

_TRUNCATION_MARKER = "\n... output truncated; preserved head and tail ...\n"
_PATCH_FAILURE_MESSAGE = (
    "patch failed. Earlier hunks may already have changed files; "
    "inspect the workspace before retrying."
)


def _search_exec_ctx() -> ToolExecContext:
    """S-1:search_text 是 EXECUTE_PROCESS;对齐测试只验证工具自身契约,
    用 identity 前缀模拟管道已编译的沙箱(真实 seatbelt 强制在 acceptance/S-1)。"""
    return ToolExecContext(
        sandbox=SandboxSpec(
            backend="seatbelt",
            argv_prefix=("/usr/bin/env",),
            scratch_dir=tempfile.gettempdir(),
        )
    )


def _run(tool: Tool, arguments: Mapping[str, Any]) -> Any:
    exec_ctx = _search_exec_ctx() if tool.definition.name == "search_text" else ToolExecContext()
    return asyncio.run(tool.run(arguments, exec_ctx))


def _schema_properties(tool: Tool) -> Mapping[str, Any]:
    from praxis.protocol.json import thaw_json

    schema = thaw_json(tool.definition.input_schema)
    assert isinstance(schema, Mapping)
    properties = schema.get("properties")
    assert isinstance(properties, Mapping)
    return properties


def _assert_rejects(tool: Tool, arguments: Mapping[str, Any]) -> ToolValidationError:
    with pytest.raises(ToolValidationError) as excinfo:
        tool.validate_input(dict(arguments))
    return excinfo.value


# ---------------------------------------------------------------------------
# search_text：六上限 + truncated 触发
# ---------------------------------------------------------------------------


def test_search_text_six_limits_match_old_worker(tmp_path: Path) -> None:
    # 旧 search_worker.py:27-32 的六项硬上限,经 metadata["limits"] 原样公开。
    assert dict(LIMITS) == {
        "max_file_bytes": 2_000_000,
        "max_total_bytes": 64_000_000,
        "max_files": 20_000,
        "max_output_bytes": 1_500_000,
        "max_backend_bytes": 8_000_000,
        "max_line_chars": 500,
    }
    (tmp_path / "one.txt").write_text("needle\n", encoding="utf-8")
    result = _run(create_search_text_tool(tmp_path), {"pattern": "needle"})
    # mappingproxy 与 dict 等价比较;这是工具对外的公开 limits 契约。
    assert dict(result.metadata["limits"]) == dict(LIMITS)


def test_search_text_input_bounds_reject_at_limit_plus_one(tmp_path: Path) -> None:
    # 上限值与旧 SearchTextInput(search.py:35-71)一致:pattern 2000、
    # path 4096、glob 512、context 50、max_results 200。
    tool = create_search_text_tool(tmp_path)
    assert len(tool.validate_input({"pattern": "x" * 2_000})["pattern"]) == 2_000
    assert len(tool.validate_input({"pattern": "x", "path": "p" * 4_096})["path"]) == 4_096
    assert tool.validate_input({"pattern": "x", "context_lines": 50})["context_lines"] == 50
    assert tool.validate_input({"pattern": "x", "max_results": 200})["max_results"] == 200

    _assert_rejects(tool, {"pattern": "x" * 2_001})
    _assert_rejects(tool, {"pattern": "x", "path": "p" * 4_097})
    _assert_rejects(tool, {"pattern": "x", "glob": "g" * 513})
    _assert_rejects(tool, {"pattern": "x", "context_lines": 51})
    _assert_rejects(tool, {"pattern": "x", "max_results": 201})


def test_search_text_truncated_when_results_exceed_cap(tmp_path: Path) -> None:
    # 旧 search.py 的 truncated 语义:结果达到 max_results 即标不完整。
    (tmp_path / "hits.txt").write_text(
        "".join(f"needle {i}\n" for i in range(5)), encoding="utf-8"
    )
    tool = create_search_text_tool(tmp_path)
    result = _run(tool, {"pattern": "needle", "max_results": 2})
    assert result.truncated is True
    assert result.metadata["total_matches"] == 2
    assert result.metadata["truncated"] is True


def test_search_text_output_is_model_readable_text_with_metadata(tmp_path: Path) -> None:
    # addenda 第 2 条:有意差异——旧实现以 structured_content 为主体,
    # praxis 输出 "<path>:<line>: <text>" 文本行,诊断信息进 metadata。
    (tmp_path / "a.txt").write_text("alpha needle\nbeta\n", encoding="utf-8")
    result = _run(create_search_text_tool(tmp_path), {"pattern": "needle"})
    text = result.content[0].data["text"]
    assert re.search(r"^a\.txt:1: alpha needle$", text, re.MULTILINE)
    assert result.metadata["backend"] == "ripgrep-bin"
    assert "total_matches" in result.metadata and "limits" in result.metadata
    assert not hasattr(result, "structured_content")


# ---------------------------------------------------------------------------
# inspect_data_file：200 MiB / 2000 格 / 32000 字符
# ---------------------------------------------------------------------------


def test_inspect_data_file_limits_declared_and_200mib_rejected(tmp_path: Path) -> None:
    # 旧 data.py:34-37 _MAX_INSPECTABLE_BYTES=200MiB / CELLS=2000 / CHARS=32000。
    tool = create_inspect_data_file_tool(tmp_path)
    description = tool.definition.description
    assert "2000-cell budget" in description
    assert "hard cap 32000" in description
    assert "Files above 200 MiB are rejected" in description
    # 稀疏文件:逻辑尺寸 200MiB+1,无需真实分配盘空间,准入检查先于读取。
    with (tmp_path / "huge.csv").open("wb") as stream:
        stream.truncate(200 * 1024 * 1024 + 1)
    result = _run(tool, {"path": "huge.csv"})
    assert result.status.value == "error"
    assert result.error_code == "data_inspection_failed"
    assert "exceeds the 200 MiB inspection limit" in result.content[0].data["text"]


def test_inspect_data_file_grid_budget_is_exactly_2000_cells(tmp_path: Path) -> None:
    # 2000 格整页不截断;第 2001 格(21×100)才触发 truncated + 前进游标。
    import csv

    def write(name: str, rows: Sequence[Sequence[str]]) -> None:
        with (tmp_path / name).open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerows(rows)

    tool = create_inspect_data_file_tool(tmp_path)
    write("cells2000.csv", [["x"] * 100 for _ in range(20)])
    result = _run(tool, {"path": "cells2000.csv", "max_rows": 100, "max_columns": 100})
    assert result.truncated is False
    assert result.metadata["next_row"] is None

    write("cells2001.csv", [["x"] * 100 for _ in range(21)])
    result = _run(tool, {"path": "cells2001.csv", "max_rows": 100, "max_columns": 100})
    assert result.truncated is True
    assert result.metadata["next_row"] == 21
    assert list(result.metadata["truncated_cells"]) == [(21, 1)]


def test_inspect_data_file_character_budget_is_exactly_32000(tmp_path: Path) -> None:
    # 32000 字符正好放满;32001 字符(33×1000)必须截断并记录源坐标。
    import csv

    def write(name: str, row: Sequence[str]) -> None:
        with (tmp_path / name).open("w", newline="", encoding="utf-8") as stream:
            csv.writer(stream).writerow(row)

    tool = create_inspect_data_file_tool(tmp_path)
    write("chars32000.csv", ["a" * 1_000] * 32)
    result = _run(
        tool,
        {"path": "chars32000.csv", "max_rows": 1, "max_columns": 100, "max_characters": 100_000},
    )
    assert result.truncated is False
    assert list(result.metadata["truncated_cells"]) == []

    write("chars32001.csv", ["a" * 1_000] * 32 + ["b"])
    result = _run(
        tool,
        {"path": "chars32001.csv", "max_rows": 1, "max_columns": 100, "max_characters": 100_000},
    )
    assert result.truncated is True
    assert list(result.metadata["truncated_cells"]) == [(1, 33)]


# ---------------------------------------------------------------------------
# apply_patch：30_000 字符 + patch_failed 非事务消息
# ---------------------------------------------------------------------------


def test_apply_patch_rejects_patch_over_30000_chars(tmp_path: Path) -> None:
    # 旧 codex_patch.py:47 max_length=30_000。
    tool = create_apply_patch_tool(tmp_path)
    assert len(tool.validate_input({"patch": "x" * 30_000})["patch"]) == 30_000
    _assert_rejects(tool, {"patch": "x" * 30_001})


def test_apply_patch_is_not_transactional_and_uses_patch_failed_message(tmp_path: Path) -> None:
    # 旧 codex_patch.py:82(错误码)与 :104(先前的 hunk 可能已写盘)。
    # H-4:错误码与提示**语义一致**,但文案有意去掉第三方产品名前缀
    # (``Codex patch failed.`` → ``patch failed.``);断言仍钉死 praxis 的完整文案。
    (tmp_path / "a.py").write_text("old-a\n", encoding="utf-8")
    (tmp_path / "b.py").write_text("actual\n", encoding="utf-8")
    patch = (
        "*** Begin Patch\n"
        "*** Update File: a.py\n@@\n-old-a\n+new-a\n"
        "*** Update File: b.py\n@@\n-missing-line\n+new-b\n"
        "*** End Patch\n"
    )
    result = _run(create_apply_patch_tool(tmp_path), {"patch": patch})
    assert result.status.value == "error"
    assert result.error_code == "patch_failed"
    assert result.error_message == _PATCH_FAILURE_MESSAGE
    # 非事务:前一个文件的 hunk 已真实落盘,后一个未动。
    assert (tmp_path / "a.py").read_text(encoding="utf-8") == "new-a\n"
    assert (tmp_path / "b.py").read_text(encoding="utf-8") == "actual\n"
    assert [Path(p).name for p in result.metadata["applied_files"]] == ["a.py"]


def test_apply_patch_does_not_accept_policy_flags(tmp_path: Path) -> None:
    # addenda 第 2 条/D1:授权只来自 policy,工具不接收 workspace_write/network。
    tool = create_apply_patch_tool(tmp_path)
    assert "workspace_write" not in _schema_properties(tool)
    assert "network" not in _schema_properties(tool)
    _assert_rejects(tool, {"patch": "x", "workspace_write": True})


# ---------------------------------------------------------------------------
# update_plan：1..20 + 单一 in_progress
# ---------------------------------------------------------------------------


def test_update_plan_step_bounds_are_1_to_20(tmp_path: Path) -> None:
    # 旧 planning.py:46-91 UpdatePlanInput min_length=1/max_length=20。
    tool = create_update_plan_tool()
    one = [{"step": "only", "status": "pending"}]
    twenty = [{"step": f"s{i}", "status": "pending"} for i in range(20)]
    assert len(tool.validate_input({"plan": one})["plan"]) == 1
    assert len(tool.validate_input({"plan": twenty})["plan"]) == 20
    _assert_rejects(tool, {"plan": []})
    _assert_rejects(tool, {"plan": twenty + [{"step": "s21", "status": "pending"}]})


def test_update_plan_allows_at_most_one_in_progress(tmp_path: Path) -> None:
    tool = create_update_plan_tool()
    ok = tool.validate_input(
        {"plan": [{"step": "a", "status": "in_progress"}, {"step": "b", "status": "pending"}]}
    )
    assert [step["status"] for step in ok["plan"]] == ["in_progress", "pending"]
    error = _assert_rejects(
        tool,
        {
            "plan": [
                {"step": "a", "status": "in_progress"},
                {"step": "b", "status": "in_progress"},
            ]
        },
    )
    # 旧 planning.py:88-89 的消息逐字。
    assert error.message == "At most one step can be in_progress at a time"

    result = _run(
        tool,
        {"plan": [{"step": "a", "status": "in_progress"}, {"step": "b", "status": "pending"}]},
    )
    assert result.status.value == "ok"
    assert result.metadata["accepted"] is True
    assert result.metadata["authority"] == "advisory"


# ---------------------------------------------------------------------------
# bash：120/600/60/605 + yield_time_seconds
# ---------------------------------------------------------------------------


def test_bash_timeout_and_yield_bounds_match_old_shell(tmp_path: Path) -> None:
    # 旧 shell.py:54 _MAX_COMMAND_TIMEOUT_SECONDS=600;:88-108 默认 120、yield ≤60;
    # :322 工具级硬超时 605(比命令超时多出的 5s 用于进程组收尸)。
    tool = create_bash_tool(tmp_path)
    assert tool.timeout_seconds == 605
    properties = _schema_properties(tool)
    assert "1..600 (default 120)" in properties["timeout_seconds"]["description"]
    assert "0 < x <= 60" in properties["yield_time_seconds"]["description"]

    assert tool.validate_input({"command": "true", "timeout_seconds": 600})["timeout_seconds"] == 600
    assert tool.validate_input({"command": "true", "yield_time_seconds": 60})["yield_time_seconds"] == 60
    _assert_rejects(tool, {"command": "true", "timeout_seconds": 601})
    _assert_rejects(tool, {"command": "true", "timeout_seconds": 0})
    _assert_rejects(tool, {"command": "true", "yield_time_seconds": 0})
    _assert_rejects(tool, {"command": "true", "yield_time_seconds": 61})


def test_bash_without_yield_has_no_session_fields(tmp_path: Path) -> None:
    # yield_time_seconds 缺省 = 旧前台行为:结果没有 session_id,进程组已收尾。
    result = _run(create_bash_tool(tmp_path), {"command": "echo parity"})
    assert result.content[0].data["text"] == "parity\n"
    assert "session_id" not in result.metadata
    assert result.metadata["state"] == "completed"


# ---------------------------------------------------------------------------
# manage_command：wait≤60 / read≤64_000 / 容量 32
# ---------------------------------------------------------------------------


def test_manage_command_schema_bounds_match_old_command_sessions(tmp_path: Path) -> None:
    # 旧 command_sessions.py:52-56 wait_seconds le=60、max_bytes le=64_000。
    tool = create_manage_command_tool(CommandSessions(tmp_path, jobs_dir=tmp_path / "jobs"))
    properties = _schema_properties(tool)
    assert properties["wait_seconds"]["maximum"] == 60
    assert properties["wait_seconds"]["default"] == 10
    assert properties["max_bytes"]["maximum"] == 64_000
    assert properties["max_bytes"]["default"] == 16_000

    session_id = "cmd_" + "0" * 32
    assert tool.validate_input({"session_id": session_id, "wait_seconds": 60})["wait_seconds"] == 60
    assert tool.validate_input({"session_id": session_id, "max_bytes": 64_000})["max_bytes"] == 64_000
    _assert_rejects(tool, {"session_id": session_id, "wait_seconds": 61})
    _assert_rejects(tool, {"session_id": session_id, "max_bytes": 64_001})


class _CompletingExecution:
    """容量用例的假执行器:start 后很快终态,但结果从未被取回(observed=False)。"""

    def __init__(self) -> None:
        self.pid = os.getpid()
        self.truncated = False

    async def wait_terminal(self, timeout: float) -> Any:
        await asyncio.sleep(0.02)
        return SimpleNamespace(exit_code=0, timed_out=False)

    async def cancel(self) -> None:
        return None

    def stdout_bytes(self) -> bytes:
        return b""

    def stderr_bytes(self) -> bytes:
        return b""


def test_command_sessions_capacity_is_32(tmp_path: Path) -> None:
    # 旧 command_sessions.py:39 _MAX_RETAINED_SESSIONS=32。
    async def scenario() -> None:
        sessions = CommandSessions(tmp_path, jobs_dir=tmp_path / "jobs")
        try:
            for index in range(32):
                handle = await sessions.start(
                    lambda: _async_value(_CompletingExecution()),
                    command_prefix=f"echo {index}",
                    timeout_seconds=60,
                    yield_time_seconds=0,
                )
                assert handle.status == "running"
                await asyncio.sleep(0.03)  # 让后台终态落地但**不取回**(observed 保持 False)
            with pytest.raises(CommandSessionError) as excinfo:
                await sessions.start(
                    lambda: _async_value(_CompletingExecution()),
                    command_prefix="echo overflow",
                    timeout_seconds=60,
                    yield_time_seconds=0,
                )
            assert excinfo.value.code == "command_session_capacity"
        finally:
            await sessions.aclose()

    asyncio.run(scenario())
    assert list((tmp_path / "jobs").glob("*.json")) == []


async def _async_value(value: Any) -> Any:
    return value


# ---------------------------------------------------------------------------
# run_python：100_000 字符 / ≤600 秒 / 16_000 头尾截断
# ---------------------------------------------------------------------------


def test_run_python_code_and_timeout_bounds(tmp_path: Path) -> None:
    # 旧 shell.py:201 code max_length=100_000;:228 timeout le=600;
    # :322 工具级硬超时 605。
    tool = create_run_python_tool(tmp_path)
    assert tool.timeout_seconds == 605
    assert tool.validate_input({"code": "x" * 100_000})["code"] == "x" * 100_000
    assert tool.validate_input({"code": "print(1)", "timeout_seconds": 600})["timeout_seconds"] == 600
    _assert_rejects(tool, {"code": "x" * 100_001})
    _assert_rejects(tool, {"code": "print(1)", "timeout_seconds": 601})
    _assert_rejects(tool, {"code": "print(1)", "timeout_seconds": None})


def test_run_python_does_not_accept_policy_flags(tmp_path: Path) -> None:
    # addenda 第 2 条/D1:run_python 与 apply_patch 一样不接收写/网旗标。
    tool = create_run_python_tool(tmp_path)
    properties = _schema_properties(tool)
    assert "workspace_write" not in properties
    assert "network" not in properties
    _assert_rejects(tool, {"code": "print(1)", "network": True})


def test_run_python_truncation_marker_is_verbatim_head_and_tail(tmp_path: Path) -> None:
    # 旧 shell.py:56 的标记文案逐字;:2277-2285 头尾各保留一半。
    tool = create_run_python_tool(tmp_path)
    code = 'import sys; sys.stdout.write("A" * 20000 + "B" * 10000)'
    result = _run(tool, {"code": code})
    text = result.content[0].data["text"]
    assert result.truncated is True
    assert _TRUNCATION_MARKER in text
    assert text.startswith("A")
    assert text.endswith("B")
    assert len(text) == 16_000  # 缓冲上限即标记替换后的总长


# ---------------------------------------------------------------------------
# B33 / child_env：宿主环境不继承(intentional differences asserted, not omitted)
# ---------------------------------------------------------------------------


def test_child_env_is_the_only_minimal_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 语义来源:旧 shell.py:1089-1125 刻意不透传宿主 env。praxis 进一步最小化:
    # 旧白名单里的 LANG/PIP_CACHE_DIR/PYTHONDONTWRITEBYTECODE/SHELL/UV_CACHE_DIR/
    # XDG_CACHE_HOME 一律不带(有意差异,显式断言)。
    monkeypatch.setenv("PRAXIS_CANARY", "SECRET-XYZ")
    monkeypatch.setenv("PATH", "/host-only-bin")
    scratch = tmp_path / "scratch"
    workspace = tmp_path / "ws"
    scratch.mkdir()
    workspace.mkdir()
    env = child_env(scratch, workspace)

    assert set(env) == {
        "PATH",
        "HOME",
        "TMPDIR",
        "TEMP",
        "TMP",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_NOSYSTEM",
        "LC_ALL",
    }
    assert env["HOME"] == os.path.realpath(scratch)
    assert env["TMPDIR"] == os.path.realpath(scratch)
    assert env["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert env["GIT_CONFIG_NOSYSTEM"] == "1"
    assert env["LC_ALL"] == "C"
    assert "/host-only-bin" not in env["PATH"]
    assert env["PATH"].split(os.pathsep)[:2] == [
        os.path.join(os.path.realpath(workspace), ".venv", "bin"),
        os.path.join(os.path.realpath(workspace), "node_modules", ".bin"),
    ]
    assert "PRAXIS_CANARY" not in env
    assert not {"LANG", "PIP_CACHE_DIR", "PYTHONDONTWRITEBYTECODE", "SHELL", "UV_CACHE_DIR", "XDG_CACHE_HOME"} & set(env)


def test_bash_child_cannot_see_host_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 真实子进程验证(B33):凭据类宿主变量打印为空,HOME/TMPDIR 指向 scratch。
    monkeypatch.setenv("PRAXIS_CANARY", "SECRET-XYZ")
    result = _run(
        create_bash_tool(tmp_path),
        {"command": 'printf "%s|%s|%s" "$PRAXIS_CANARY" "$HOME" "$TMPDIR"'},
    )
    canary, home, tmpdir = result.content[0].data["text"].split("|")
    assert canary == ""
    assert home == tmpdir
    assert home != os.path.expanduser("~")


# ---------------------------------------------------------------------------
# B20：命令结束以主进程退出为判据,不被后台孙进程的管道 EOF 绑架
# ---------------------------------------------------------------------------


def _pgid_members(pgid: int) -> list[str]:
    completed = subprocess.run(
        ["ps", "-A", "-o", "pid=,pgid="],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    members = []
    for line in completed.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] == str(pgid):
            members.append(parts[0])
    return members


def test_bash_returns_on_main_exit_despite_background_pipe_holder(tmp_path: Path) -> None:
    # 旧 shell.py:822-840 用 gather(stdout EOF, stderr EOF, wait) 判定结束;
    # `sleep 30 &` 持写端时旧实现要等 30s。praxis 等主进程退出后 0.5s 收尾。
    tool = create_bash_tool(tmp_path)
    started = time.monotonic()
    result = _run(tool, {"command": "sleep 30 & echo main-done"})
    elapsed = time.monotonic() - started
    assert result.content[0].data["text"] == "main-done\n"
    assert result.metadata["exit_code"] == 0
    assert elapsed < 10.0, f"main exit was held hostage by grandchild pipe: {elapsed:.1f}s"
    # 收尾必须真的把同组后台进程杀掉(不是逃逸成孤儿)。
    pgid = int(result.metadata["pgid"])
    deadline = time.monotonic() + 10.0
    while _pgid_members(pgid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert _pgid_members(pgid) == []