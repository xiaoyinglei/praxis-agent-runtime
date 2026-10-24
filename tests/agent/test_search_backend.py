from __future__ import annotations

import asyncio
import inspect
import os
import sys
from pathlib import Path

import pytest

from agent_runtime.tools.builtins import search
from agent_runtime.workspace import open_workspace


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_search_runner_is_async_so_timeout_can_interrupt_filesystem_work(tmp_path):
    tool = search.create_search_text_tool(open_workspace(tmp_path, create=True))
    assert inspect.iscoroutinefunction(tool.run)


@pytest.mark.anyio
async def test_search_uses_packaged_ripgrep_with_empty_path(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "")
    (tmp_path / "file.txt").write_text("前缀 needle 后缀\n")
    result = await search.create_search_text_tool(open_workspace(tmp_path)).run(
        {"pattern": "needle", "path": "file.txt"}
    )
    assert result.matches[0].match_start == 3
    assert result.matches[0].match_end == 9
    assert result.searched_file_path == "file.txt"


@pytest.mark.anyio
async def test_search_handles_rust_regex_and_rejects_unsupported_regex(tmp_path):
    (tmp_path / "file.txt").write_text("needle\n")
    tool = search.create_search_text_tool(open_workspace(tmp_path))
    result = await tool.run({"pattern": r"(?<symbol>needle)", "regex": True})
    assert result.matches[0].line_content == "needle"
    failed = await tool.run({"pattern": "(", "regex": True})
    assert tool.normalize_output(failed).error_code == "invalid_arguments"


@pytest.mark.anyio
@pytest.mark.parametrize("newline", ["\n", "\r\n"], ids=["lf", "crlf"])
@pytest.mark.parametrize("final_newline", [False, True], ids=["unterminated", "terminated"])
async def test_search_line_anchors_preserve_context_and_unicode_offsets(tmp_path, newline, final_newline):
    contents = newline.join(["前文", "foobar", "foo", "中文目标"])
    if final_newline:
        contents += newline
    (tmp_path / "file.txt").write_bytes(contents.encode())
    tool = search.create_search_text_tool(open_workspace(tmp_path))

    result = await tool.run({"pattern": "^foo$", "regex": True, "path": "file.txt", "context_lines": 1})
    assert result.model_dump() == {
        "matches": [
            {
                "file_path": "file.txt",
                "line_number": 3,
                "line_content": "foo",
                "match_start": 0,
                "match_end": 3,
                "context_before": ["foobar"],
                "context_after": ["中文目标"],
            }
        ],
        "total_matches": 1,
        "truncated": False,
        "searched_file_path": "file.txt",
    }

    final = await tool.run({"pattern": "目标$", "regex": True, "path": "file.txt", "context_lines": 1})
    assert final.model_dump() == {
        "matches": [
            {
                "file_path": "file.txt",
                "line_number": 4,
                "line_content": "中文目标",
                "match_start": 2,
                "match_end": 4,
                "context_before": ["foo"],
                "context_after": [],
            }
        ],
        "total_matches": 1,
        "truncated": False,
        "searched_file_path": "file.txt",
    }


@pytest.mark.anyio
async def test_search_literal_mode_preserves_carriage_returns_and_literal_anchors(tmp_path):
    (tmp_path / "file.txt").write_bytes("前文foo\rbar后文\r\n^foo$\r\n".encode())
    tool = search.create_search_text_tool(open_workspace(tmp_path))
    result = await tool.run({"pattern": "foo\rbar", "regex": False, "path": "file.txt", "context_lines": 1})
    assert result.total_matches == 1
    assert result.truncated is False
    match = result.matches[0]
    assert match.line_content == "前文foo\rbar后文"
    assert (match.match_start, match.match_end) == (2, 9)
    assert match.context_after == ["^foo$"]
    anchored = await tool.run({"pattern": "^foo$", "regex": False, "path": "file.txt", "context_lines": 0})
    assert anchored.total_matches == 1
    assert anchored.truncated is False
    assert anchored.matches[0].line_number == 2
    assert anchored.matches[0].line_content == "^foo$"
    assert (anchored.matches[0].match_start, anchored.matches[0].match_end) == (0, 5)


@pytest.mark.anyio
async def test_search_caps_file_reads_without_path_read_bytes(tmp_path, monkeypatch):
    target = tmp_path / "huge.txt"
    with target.open("wb") as handle:
        handle.write(b"needle\n" + b"x" * 2_000_000)
        handle.seek(128 * 1024 * 1024)
        handle.write(b"late_marker\n")
    monkeypatch.setattr(Path, "read_bytes", lambda self: pytest.fail("unbounded Path.read_bytes"))
    result = await search.create_search_text_tool(open_workspace(tmp_path)).run(
        {"pattern": "needle", "path": "huge.txt", "context_lines": 0}
    )
    assert result.truncated is True
    assert result.matches[0].line_content == "needle"


@pytest.mark.anyio
@pytest.mark.parametrize("ignore_term", [False, True])
async def test_cancel_search_kills_and_reaps_worker_and_backend(tmp_path, monkeypatch, ignore_term):
    script = tmp_path / "backend"
    marker = tmp_path / "pids"
    script.write_text(
        f"#!{sys.executable}\n"
        "import os,time,signal\n"
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN)\n" if ignore_term else "")
        + f"open({str(marker)!r}, 'w').write(str(os.getpid())+' '+str(os.getppid()))\n"
        "time.sleep(60)\n"
    )
    script.chmod(0o755)
    (tmp_path / "file.txt").write_text("needle\n")
    monkeypatch.setattr(search, "_ripgrep_binary", lambda: script)
    task = asyncio.create_task(
        search.create_search_text_tool(open_workspace(tmp_path)).run({"pattern": "needle", "path": "file.txt"})
    )
    try:
        async with asyncio.timeout(5):
            while not marker.exists():
                await asyncio.sleep(0.01)
        backend_pid, worker_pid = map(int, marker.read_text().split())
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=2)
        for pid in (backend_pid, worker_pid):
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


@pytest.mark.anyio
async def test_search_preserves_denials_ceilings_external_grants_and_symlink_refusal(tmp_path):
    from agent_runtime.permission_policy import PathRule, PermissionPolicy

    root = tmp_path / "workspace"
    workspace = open_workspace(root, create=True)
    denied = root / "denied"
    denied.mkdir()
    (denied / "secret.txt").write_text("needle denied\n")
    (root / "public.txt").write_text("needle public\n")
    external = tmp_path / "external.txt"
    external.write_text("needle external\n")
    (root / "alias").symlink_to(denied, target_is_directory=True)
    policy = PermissionPolicy(filesystem=(PathRule(path=str(external), access="read"),)).resolve(root)
    ceiling = PermissionPolicy(
        filesystem=(
            PathRule(path=str(external), access="read"),
            PathRule(path=str(denied), access="deny"),
        )
    ).resolve(root)
    policy = policy.model_copy(update={"ceilings": (ceiling,)})
    workspace.permission_provider = lambda: policy
    tool = search.create_search_text_tool(workspace)
    result = await tool.run({"pattern": "needle"})
    assert [match.file_path for match in result.matches] == ["public.txt"]
    outside = await tool.run({"pattern": "needle", "path": str(external)})
    assert outside.matches[0].file_path == str(external)
    assert outside.searched_file_path == str(external)
    assert outside.truncated is False
    with pytest.raises(ValueError):
        await tool.run({"pattern": "needle", "path": "denied"})
    (root / "public_alias").symlink_to(root / "public.txt")
    alias = await tool.run({"pattern": "needle", "path": "public_alias"})
    assert alias.matches == []
    assert alias.searched_file_path is None


def test_search_bounds_enumeration_even_when_glob_excludes_every_file(tmp_path, monkeypatch):
    from agent_runtime.tools.builtins import search_worker

    for index in range(10):
        (tmp_path / f"{index}.txt").write_text("needle\n")
    monkeypatch.setattr(search_worker, "_MAX_FILES", 2)
    result = search_worker._search(
        {
            "root": str(tmp_path),
            "permissions": None,
            "request": search.SearchTextInput(pattern="needle", glob="*.py").model_dump(),
            "regex": False,
            "binary": str(search._ripgrep_binary()),
        }
    )
    assert result["matches"] == []
    assert result["truncated"] is True


@pytest.mark.anyio
async def test_backend_json_and_model_output_are_bounded_for_dense_long_line(tmp_path):
    (tmp_path / "dense.txt").write_text("a" * 300_000 + "\n")
    result = await search.create_search_text_tool(open_workspace(tmp_path)).run(
        {
            "pattern": ".",
            "regex": True,
            "path": "dense.txt",
            "context_lines": 0,
        }
    )
    assert result.truncated is True
    assert len(result.model_dump_json().encode()) < 1_501_000


@pytest.mark.anyio
async def test_search_missing_packaged_binary_has_typed_error(tmp_path, monkeypatch):
    def unavailable():
        raise search.importlib.metadata.PackageNotFoundError("ripgrep-bin")

    monkeypatch.setattr(search, "_ripgrep_binary", unavailable)
    tool = search.create_search_text_tool(open_workspace(tmp_path))
    failed = await tool.run({"pattern": "needle"})
    output = tool.normalize_output(failed)
    assert output.error_code == "search_backend_unavailable"
    assert output.is_error is True


@pytest.mark.anyio
async def test_executor_timeout_reaps_real_search_processes_and_keeps_loop_responsive(tmp_path, monkeypatch):
    from dataclasses import replace

    from agent_runtime.tools.executor import ToolExecutor
    from agent_runtime.tools.permissions import ToolExecutionContext
    from agent_runtime.tools.tool import ToolCall, ToolCallOrigin

    backend = tmp_path / "backend"
    marker = tmp_path / "pids"
    backend.write_text(
        f"#!{sys.executable}\nimport os,time\n"
        f"open({str(marker)!r}, 'w').write(str(os.getpid())+' '+str(os.getppid()))\n"
        "time.sleep(60)\n"
    )
    backend.chmod(0o755)
    monkeypatch.setattr(search, "_ripgrep_binary", lambda: backend)
    workspace = open_workspace(tmp_path)
    tool = replace(search.create_search_text_tool(workspace), timeout_seconds=1.0)
    call = ToolCall(
        tool_call_id="timeout_search",
        tool_name="search_text",
        arguments={"pattern": "needle"},
        origin=ToolCallOrigin(request_id="timeout_test", toolset_revision="test", exposed_tool_names=("search_text",)),
    )
    task = asyncio.create_task(
        ToolExecutor({"search_text": tool}).execute(
            call,
            context=ToolExecutionContext(
                workspace_root=tmp_path,
                cwd=tmp_path,
            ),
        )
    )
    ticks = 0
    while not task.done():
        await asyncio.sleep(0.02)
        ticks += 1
    execution = await task
    assert execution.result.error_code == "timeout_cancelled"
    assert ticks >= 5
    assert marker.exists(), "real backend must have started before the deadline"
    for pid in map(int, marker.read_text().split()):
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)


def test_bounded_reader_reads_only_limit_plus_sentinel_byte(tmp_path):
    from agent_runtime.tools.builtins.search_worker import _read_file

    target = tmp_path / "huge.txt"
    with target.open("wb") as handle:
        handle.write(b"a" * 100)
        handle.seek(128 * 1024 * 1024)
        handle.write(b"tail")
    assert _read_file(target, 31) == b"a" * 32


def test_packaged_binary_metadata_survives_prefix_relocation(tmp_path):
    import shutil
    import subprocess

    distribution = search.importlib.metadata.distribution("ripgrep-bin")
    metadata_entry = next(entry for entry in distribution.files if entry.name == "METADATA")
    metadata = Path(str(distribution.locate_file(metadata_entry))).parent
    prefix = tmp_path / "original" / "python"
    scripts = prefix / "bin"
    scripts.mkdir(parents=True)
    shutil.copy2(search._ripgrep_binary(), scripts / "rg")
    site = prefix / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
    shutil.copytree(metadata, site / metadata.name)
    moved = tmp_path / "relocated"
    prefix.parent.rename(moved)
    site = moved / "python" / "lib" / site.parent.name / "site-packages"
    code = (
        f"import sys; sys.path.insert(0, {str(Path(__file__).resolve().parents[2])!r});"
        f"sys.path.insert(0, {str(site)!r});"
        "from agent_runtime.tools.builtins.search import _ripgrep_binary;"
        "import subprocess; p=_ripgrep_binary(); print(p);"
        "print(subprocess.check_output([str(p),'--version'],text=True))"
    )
    output = subprocess.check_output([sys.executable, "-I", "-c", code], env={**os.environ, "PATH": ""}, text=True)
    assert str(moved / "python" / "bin" / "rg") in output
    assert "ripgrep " in output
