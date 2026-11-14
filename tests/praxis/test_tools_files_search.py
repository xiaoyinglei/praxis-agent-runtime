"""Task 10/11:read 行模式、list_files 与 search_text 内置工具。

阶段 1 的 read 只做字节分页;Task 10 按设计 §6 移植清单补齐行模式与
list_files 的确定性目录清单语义。Task 11 追加 search_text(打包 ripgrep +
隔离 worker)与 R-T11-2 的逐条目 deny 过滤。read 既有字节模式用例自
test_tools_builtin.py 迁移至此(断言逐条保留,不弱化)。

仓库无 pytest-asyncio:与 test_tools_builtin.py 同风格,asyncio.run 包装。
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

import pytest

from praxis.core import permission
from praxis.core.pipeline import ExecCtx, execute
from praxis.protocol.permission import PermissionContext
from praxis.protocol.policy import PathRule, PermissionPolicy, path_access, resolve_policy
from praxis.protocol.sandbox import ToolExecContext
from praxis.protocol.tools import (
    CancellationMode,
    InterruptBehavior,
    ResultStatus,
    ToolCall,
    ToolEffect,
    ToolValidationError,
)
from praxis.store.turns import create_turn
from praxis.tools.builtins import search as search_module
from praxis.tools.builtins.filesystem import (
    create_list_files_tool,
    create_read_tool,
    create_write_tool,
)
from praxis.tools.builtins.search import create_search_text_tool
from praxis.tools.builtins.search_worker import LIMITS


def run(tool: Any, arguments: dict[str, Any]) -> Any:
    return asyncio.run(tool.run(arguments, ToolExecContext()))


def run_with(tool: Any, arguments: dict[str, Any], exec_ctx: ToolExecContext) -> Any:
    return asyncio.run(tool.run(arguments, exec_ctx))


def _permission_ctx(workspace: Path, *rules: PathRule) -> PermissionContext:
    return PermissionContext(
        workspace=str(workspace),
        policy=PermissionPolicy(filesystem=tuple(rules)),
        mode="on-request",
    )


# ---------------------------------------------------------------------------
# 自 test_tools_builtin.py 迁移:read 字节模式既有断言(逐条保留)
# ---------------------------------------------------------------------------


def test_read_pages_and_next_offset(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("a" * 100)
    tool = create_read_tool(tmp_path)
    r = run(tool, {"path": "f.txt", "max_bytes": 40})
    assert r.text == "a" * 40
    assert r.metadata["next_offset"] == 40
    assert r.truncated
    # 续页直到读完:offset 链必须精确覆盖 100 字节
    offset = r.metadata["next_offset"]
    total = len(r.text)
    while offset is not None:
        r = run(tool, {"path": "f.txt", "offset": offset, "max_bytes": 40})
        total += len(r.text)
        offset = r.metadata["next_offset"]
    assert total == 100
    assert r.metadata["size_bytes"] == 100


def test_read_missing_file(tmp_path: Path) -> None:
    tool = create_read_tool(tmp_path)
    r = run(tool, {"path": "nope.txt"})
    assert r.status is ResultStatus.ERROR
    assert r.is_error
    assert r.error_code == "not_found"


def test_read_binary_file(tmp_path: Path) -> None:
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01\x02\xff" * 16)
    tool = create_read_tool(tmp_path)
    r = run(tool, {"path": "blob.bin"})
    assert r.status is ResultStatus.ERROR
    assert r.error_code == "binary_file"
    assert r.metadata["is_binary"]
    assert r.metadata["size_bytes"] == 64


def test_pending_tool_call_id_convention(tmp_path: Path) -> None:
    # Task 12 管道写 item 时替换真实 call id;工具层只放占位符
    (tmp_path / "f.txt").write_text("hello")
    r = run(create_read_tool(tmp_path), {"path": "f.txt"})
    assert r.tool_call_id == "pending"
    assert r.tool_name == "read"


def test_read_byte_mode_metadata_shape_unchanged(tmp_path: Path) -> None:
    # 行模式不得污染字节模式 metadata:既有四个键一字不改
    (tmp_path / "f.txt").write_text("a" * 10)
    r = run(create_read_tool(tmp_path), {"path": "f.txt", "max_bytes": 5})
    assert set(r.metadata) == {"is_binary", "size_bytes", "next_offset", "truncated"}


def test_read_byte_mode_multibyte_boundary_no_replacement_char(tmp_path: Path) -> None:
    # max_bytes=4 切在"你"(3 字节)与"好"之间:页尾不得产出 U+FFFD
    (tmp_path / "cn.txt").write_text("你好世界", encoding="utf-8")
    r = run(create_read_tool(tmp_path), {"path": "cn.txt", "max_bytes": 4})
    assert "�" not in r.text
    assert r.text == "你"
    assert r.metadata["next_offset"] == 3


# ---------------------------------------------------------------------------
# read 行模式
# ---------------------------------------------------------------------------


def test_read_lines_returns_line_metadata(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("one\ntwo\nthree\nfour\n")
    r = run(create_read_tool(tmp_path), {"path": "f.txt", "start_line": 2, "max_lines": 2})
    assert r.status is ResultStatus.OK
    assert r.text == "two\nthree"
    assert r.metadata["start_line"] == 2
    assert r.metadata["end_line"] == 3
    assert r.metadata["next_start_line"] == 4
    assert r.metadata["line_count"] == 2
    assert r.truncated


def test_read_lines_max_lines_alone_starts_at_line_one(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("one\ntwo\nthree\n")
    r = run(create_read_tool(tmp_path), {"path": "f.txt", "max_lines": 1})
    assert r.text == "one"
    assert r.metadata["start_line"] == 1
    assert r.metadata["end_line"] == 1
    assert r.metadata["next_start_line"] == 2


def test_read_lines_to_eof_has_no_next_start_line(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("one\ntwo\n")
    r = run(create_read_tool(tmp_path), {"path": "f.txt", "start_line": 1})
    assert r.text == "one\ntwo"
    assert r.metadata["line_count"] == 2
    assert r.metadata["end_line"] == 2
    assert r.metadata["next_start_line"] is None
    assert not r.truncated


def test_read_lines_without_trailing_newline_keeps_last_line(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("one\ntwo")
    r = run(create_read_tool(tmp_path), {"path": "f.txt", "start_line": 1, "max_lines": 5})
    assert r.text == "one\ntwo"
    assert r.metadata["line_count"] == 2
    assert r.metadata["next_start_line"] is None
    assert not r.truncated


def test_read_lines_trailing_newline_has_no_phantom_line(tmp_path: Path) -> None:
    # EOF 换行不是额外一行;真正的空行("one\n\n")才计一行
    tool = create_read_tool(tmp_path)
    (tmp_path / "one.txt").write_text("one\n")
    r = run(tool, {"path": "one.txt", "start_line": 1})
    assert r.text == "one"
    assert r.metadata["line_count"] == 1
    assert r.metadata["next_start_line"] is None
    (tmp_path / "blank.txt").write_text("one\n\n")
    r = run(tool, {"path": "blank.txt", "start_line": 1})
    assert r.text == "one\n"
    assert r.metadata["line_count"] == 2


def test_read_lines_beyond_eof_is_empty_result(tmp_path: Path) -> None:
    # 超界不是错误:空内容 + next_start_line=None
    (tmp_path / "f.txt").write_text("one\ntwo\n")
    r = run(create_read_tool(tmp_path), {"path": "f.txt", "start_line": 9, "max_lines": 3})
    assert r.status is ResultStatus.OK
    assert not r.is_error
    assert r.text == ""
    assert r.metadata["line_count"] == 0
    assert r.metadata["end_line"] is None
    assert r.metadata["next_start_line"] is None
    assert not r.truncated


def test_read_lines_multibyte_utf8_no_replacement_char(tmp_path: Path) -> None:
    # 每行含 3 字节汉字;行边界翻页不得产生 U+FFFD
    expected = [f"第{i}行中文内容" for i in range(1, 31)]
    (tmp_path / "cn.txt").write_text("\n".join(expected) + "\n", encoding="utf-8")
    tool = create_read_tool(tmp_path)
    r = run(tool, {"path": "cn.txt", "start_line": 5, "max_lines": 10})
    assert "�" not in r.text
    assert r.text.split("\n") == expected[4:14]
    assert r.metadata["next_start_line"] == 15
    # 续页同样不产生替换字符
    nxt = run(tool, {"path": "cn.txt", "start_line": 15, "max_lines": 10})
    assert "�" not in nxt.text
    assert nxt.text.split("\n") == expected[14:24]


def test_read_lines_binary_file_is_result(tmp_path: Path) -> None:
    (tmp_path / "blob.bin").write_bytes(b"\x00\x01\x02\xff" * 16)
    r = run(create_read_tool(tmp_path), {"path": "blob.bin", "start_line": 1})
    assert r.status is ResultStatus.ERROR
    assert r.error_code == "binary_file"
    assert r.metadata["size_bytes"] == 64


def test_read_lines_huge_single_line_is_bounded(tmp_path: Path) -> None:
    # I-1:8MB 单行不得无界阻塞;页首超预算单行 → 有界预览 + 可见提示 + 游标前进。
    (tmp_path / "huge.txt").write_bytes(b"x" * 8_000_000 + b"\n")
    started = time.monotonic()
    r = run(create_read_tool(tmp_path), {"path": "huge.txt", "start_line": 1, "max_lines": 200})
    assert time.monotonic() - started < 3
    assert r.status is ResultStatus.OK
    assert not (r.truncated and r.metadata["next_start_line"] == r.metadata["start_line"] and r.text == "")
    assert r.metadata["line_exceeds_budget"] is True
    assert r.metadata["line_count"] == 1
    assert r.metadata["next_start_line"] == 2
    assert r.truncated
    assert "�" not in r.text
    assert r.text.startswith("x")
    assert "line exceeds the line-mode byte budget" in r.text
    # 预览仍受页预算约束(提示行不计入预算)
    assert len(r.text.encode("utf-8")) < 1_000_000 + 200


def test_read_lines_over_budget_line_advances_cursor(tmp_path: Path) -> None:
    # 复审 fix round 2:超预算单行不是无前进死路——游标推进后同一文件能继续读
    (tmp_path / "f.txt").write_bytes(b"y" * 2_000_000 + b"\nsecond\n")
    tool = create_read_tool(tmp_path)
    first = run(tool, {"path": "f.txt", "start_line": 1, "max_lines": 200})
    assert first.metadata["line_exceeds_budget"] is True
    start = first.metadata["start_line"]
    next_start = first.metadata["next_start_line"]
    assert next_start == start + 1  # 游标真的前进,不是自循环
    assert not (first.truncated and next_start == start and first.text == "")
    second = run(tool, {"path": "f.txt", "start_line": next_start, "max_lines": 200})
    assert second.text == "second"
    assert second.metadata["line_exceeds_budget"] is False
    assert second.metadata["next_start_line"] is None
    assert not second.truncated


def test_read_lines_million_line_file_is_bounded(tmp_path: Path) -> None:
    (tmp_path / "many.txt").write_text("x\n" * 1_000_000)
    started = time.monotonic()
    r = run(create_read_tool(tmp_path), {"path": "many.txt", "start_line": 1, "max_lines": 200})
    assert time.monotonic() - started < 3
    assert r.metadata["line_count"] == 200
    assert r.metadata["next_start_line"] == 201
    assert r.truncated
    # 翻到文件尾部附近也不得无界阻塞
    started = time.monotonic()
    tail = run(create_read_tool(tmp_path), {"path": "many.txt", "start_line": 999_900, "max_lines": 200})
    assert time.monotonic() - started < 3
    assert tail.metadata["line_count"] == 101
    assert tail.metadata["next_start_line"] is None
    assert not tail.truncated


def test_read_lines_byte_budget_stops_on_whole_line(tmp_path: Path) -> None:
    # 内部字节预算与 max_lines 同时生效:在整行边界收口,给出可续读的 next_start_line
    (tmp_path / "wide.txt").write_text(("a" * 1_000 + "\n") * 1_200)
    r = run(create_read_tool(tmp_path), {"path": "wide.txt", "start_line": 1, "max_lines": 2_000})
    assert r.status is ResultStatus.OK
    assert r.truncated
    line_count = r.metadata["line_count"]
    assert 0 < line_count < 1_200
    assert len(r.text.encode("utf-8")) <= 1_000_000
    assert r.metadata["next_start_line"] == line_count + 1
    # 续页从收口处继续,不丢行也不重复
    nxt = run(
        create_read_tool(tmp_path),
        {"path": "wide.txt", "start_line": r.metadata["next_start_line"], "max_lines": 2_000},
    )
    assert nxt.text.split("\n")[0] == "a" * 1_000


def test_read_rejects_mixed_line_and_byte_modes(tmp_path: Path) -> None:
    # 行/字节模式互斥:同时给 → 校验错误(管道归 invalid_input)
    tool = create_read_tool(tmp_path)
    with pytest.raises(ToolValidationError):
        tool.validate_input({"path": "f.txt", "start_line": 1, "offset": 0})
    with pytest.raises(ToolValidationError):
        tool.validate_input({"path": "f.txt", "max_lines": 2, "max_bytes": 10})
    # 单一模式仍然合法
    tool.validate_input({"path": "f.txt", "start_line": 1})
    tool.validate_input({"path": "f.txt", "offset": 0})
    tool.validate_input({"path": "f.txt", "max_bytes": 10})


def test_read_lines_schema_bounds(tmp_path: Path) -> None:
    tool = create_read_tool(tmp_path)
    with pytest.raises(ToolValidationError):
        tool.validate_input({"path": "f.txt", "start_line": 0})
    with pytest.raises(ToolValidationError):
        tool.validate_input({"path": "f.txt", "max_lines": 0})
    with pytest.raises(ToolValidationError):
        tool.validate_input({"path": "f.txt", "max_lines": 2_001})


# ---------------------------------------------------------------------------
# list_files
# ---------------------------------------------------------------------------


def test_list_files_deterministic_sorted_entries(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("xy")
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "inner.txt").write_text("z")
    tool = create_list_files_tool(tmp_path)
    r = run(tool, {"path": "."})
    assert r.status is ResultStatus.OK
    parsed = []
    for line in r.text.split("\n"):
        kind, size, relative = line.split("\t")
        assert kind in {"d", "f", "l"}
        assert int(size) >= 0
        parsed.append((kind, relative))
    # 每层按名字排序 + 前序深搜,目录列出并进入
    assert parsed == [("d", "a"), ("f", "a/inner.txt"), ("f", "b.txt")]
    assert not r.metadata["truncated"]
    assert r.metadata["next_cursor"] is None
    # 同一输入两次调用输出逐字节相同
    assert run(tool, {"path": "."}).text == r.text


def test_list_files_truncated_at_max_entries(tmp_path: Path) -> None:
    for name in ("a.txt", "b.txt", "c.txt", "d.txt", "e.txt"):
        (tmp_path / name).write_text(name)
    r = run(create_list_files_tool(tmp_path), {"max_entries": 3})
    relative = [line.split("\t")[2] for line in r.text.split("\n")]
    assert relative == ["a.txt", "b.txt", "c.txt"]
    assert r.metadata["truncated"] is True
    assert r.metadata["next_cursor"] == "d.txt"


def test_list_files_skips_floor_subtrees(tmp_path: Path) -> None:
    # .git/.praxis/.agent_memory 子树跳过(由 FLOOR_ENTRIES 派生)
    for name in (".git", ".praxis", ".agent_memory"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "inside.txt").write_text("x")
    (tmp_path / "keep.txt").write_text("k")
    r = run(create_list_files_tool(tmp_path), {"path": ".", "include_hidden": True})
    relative = [line.split("\t")[2] for line in r.text.split("\n")]
    assert relative == ["keep.txt"]


def test_list_files_floor_match_is_casefolded(tmp_path: Path) -> None:
    # 地板名匹配与 floor_reason 一致:NFC + casefold(单独目录,避开大小写不敏感 FS 冲突)
    (tmp_path / ".GIT").mkdir()
    (tmp_path / ".GIT" / "inside.txt").write_text("x")
    (tmp_path / "keep.txt").write_text("k")
    r = run(create_list_files_tool(tmp_path), {"path": ".", "include_hidden": True})
    relative = [line.split("\t")[2] for line in r.text.split("\n")]
    assert relative == ["keep.txt"]


def test_list_files_skips_nested_floor_subtree(tmp_path: Path) -> None:
    # 地板目录的子条目同样跳过(不只是工作区顶层)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / ".git").mkdir()
    (tmp_path / "sub" / ".git" / "config").write_text("x")
    (tmp_path / "sub" / "keep.txt").write_text("k")
    r = run(create_list_files_tool(tmp_path), {"path": ".", "include_hidden": True})
    relative = [line.split("\t")[2] for line in r.text.split("\n")]
    assert relative == ["sub", "sub/keep.txt"]


def test_list_files_unreadable_directory_is_fail_open(tmp_path: Path) -> None:
    # 读路径对畸形输入是结果不是异常:不可读子树跳过,其余照常列出
    (tmp_path / "ok.txt").write_text("x")
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "secret.txt").write_text("s")
    os.chmod(blocked, 0)
    try:
        if os.access(blocked, os.R_OK):
            pytest.skip("permission bits not enforced for this user")
        r = run(create_list_files_tool(tmp_path), {"path": "."})
        assert r.status is ResultStatus.OK
        relative = [line.split("\t")[2] for line in r.text.split("\n")]
        assert "blocked" in relative
        assert "blocked/secret.txt" not in relative
        assert "ok.txt" in relative
    finally:
        os.chmod(blocked, 0o755)


def test_list_files_rejects_empty_path(tmp_path: Path) -> None:
    with pytest.raises(ToolValidationError):
        create_list_files_tool(tmp_path).validate_input({"path": ""})


def test_list_files_include_hidden_controls_dotfiles(tmp_path: Path) -> None:
    (tmp_path / ".secret").write_text("s")
    (tmp_path / "visible.txt").write_text("v")
    tool = create_list_files_tool(tmp_path)
    hidden_off = [line.split("\t")[2] for line in run(tool, {"path": "."}).text.split("\n")]
    assert hidden_off == ["visible.txt"]
    hidden_on = [
        line.split("\t")[2]
        for line in run(tool, {"path": ".", "include_hidden": True}).text.split("\n")
    ]
    assert hidden_on == [".secret", "visible.txt"]


def test_list_files_lists_symlinks_without_following(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("s")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "inner").mkdir()
    (workspace / "inner" / "f.txt").write_text("x")
    (workspace / "escape").symlink_to(outside)
    (workspace / "inside_link").symlink_to(workspace / "inner")
    (workspace / "file_link").symlink_to(workspace / "inner" / "f.txt")

    r = run(create_list_files_tool(workspace), {"path": "."})
    lines = [line.split("\t") for line in r.text.split("\n")]
    by_path = {relative: kind for kind, _size, relative in lines}
    # 符号链接一律按 l 列出,指向工作区外也只列出不进入
    assert by_path["escape"] == "l"
    assert by_path["inside_link"] == "l"
    assert by_path["file_link"] == "l"
    # 不跟随任何 symlink:不得出现 link/ 下的子条目
    assert sorted(by_path) == ["escape", "file_link", "inner", "inner/f.txt", "inside_link"]


def test_list_files_missing_directory_is_result(tmp_path: Path) -> None:
    r = run(create_list_files_tool(tmp_path), {"path": "nope"})
    assert r.status is ResultStatus.ERROR
    assert r.error_code == "not_found"
    assert r.metadata["truncated"] is False


def test_list_files_file_target_is_result(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("x")
    r = run(create_list_files_tool(tmp_path), {"path": "f.txt"})
    assert r.status is ResultStatus.ERROR
    assert r.error_code == "not_found"


def test_list_files_rejects_escape_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    tool = create_list_files_tool(workspace)
    with pytest.raises(ToolValidationError):
        run(tool, {"path": "../outside"})
    with pytest.raises(ToolValidationError):
        tool.resolve_use({"path": "../outside"})


def test_list_files_policy_and_resolve_use(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "sub").mkdir()
    tool = create_list_files_tool(workspace)
    assert tool.definition.name == "list_files"
    assert tool.static_effects == frozenset({ToolEffect.READ_WORKSPACE})
    assert tool.idempotent and tool.concurrency_safe
    assert tool.cancellation_mode is CancellationMode.COOPERATIVE
    assert tool.interrupt_behavior is InterruptBehavior.CANCEL
    assert tool.timeout_seconds == 15
    assert tool.max_model_output_bytes == 200_000
    # 默认 "." 合法:target 就是工作区根(目录清单允许根,读文件不允许)
    root_use = tool.resolve_use({})
    assert root_use.effects == frozenset({ToolEffect.READ_WORKSPACE})
    assert root_use.targets[0].value == str(workspace.resolve())
    sub_use = tool.resolve_use({"path": "sub"})
    assert sub_use.targets[0].value == str((workspace / "sub").resolve())


def test_list_files_max_entries_schema_bounds(tmp_path: Path) -> None:
    tool = create_list_files_tool(tmp_path)
    with pytest.raises(ToolValidationError):
        tool.validate_input({"max_entries": 0})
    with pytest.raises(ToolValidationError):
        tool.validate_input({"max_entries": 2_001})
    with pytest.raises(ToolValidationError):
        tool.validate_input({"path": "x" * 4_097})
    tool.validate_input({"max_entries": 2_000})


def test_read_and_write_still_shared_workspace_guards(tmp_path: Path) -> None:
    # 围栏回归:allow_root 只为 list_files 开口,read/write 仍拒绝越界与工作区根
    (tmp_path / "f.txt").write_text("x")
    with pytest.raises(ToolValidationError):
        run(create_read_tool(tmp_path), {"path": "../escape"})
    with pytest.raises(ToolValidationError):
        run(create_write_tool(tmp_path), {"path": "../escape", "content": "x"})
    with pytest.raises(ToolValidationError):
        run(create_read_tool(tmp_path), {"path": "."})
    with pytest.raises(ToolValidationError):
        run(create_write_tool(tmp_path), {"path": ".", "content": "x"})


# ---------------------------------------------------------------------------
# search_text(Task 11:打包 ripgrep + 隔离 worker)
# ---------------------------------------------------------------------------


def test_search_finds_literal_and_regex(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("a.b literal\naxb regex\n", encoding="utf-8")
    tool = create_search_text_tool(tmp_path)
    literal = run(tool, {"pattern": "a.b", "regex": False})
    assert literal.status is ResultStatus.OK
    assert "f.txt:1: a.b literal" in literal.text
    assert "f.txt:2: axb regex" not in literal.text
    regex = run(tool, {"pattern": "a.b", "regex": True})
    assert "f.txt:1: a.b literal" in regex.text
    assert "f.txt:2: axb regex" in regex.text


def test_context_lines_and_max_results(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text(
        "one\nMATCH-a\ntwo\nthree\nMATCH-b\nfour\n", encoding="utf-8"
    )
    r = run(create_search_text_tool(tmp_path), {"pattern": "MATCH", "context_lines": 1, "max_results": 1})
    assert r.status is ResultStatus.OK
    # 上下文行以 "- " 前缀;匹配行是 "<相对路径>:<行号>: <行内容>"
    assert r.text == "- one\nf.txt:2: MATCH-a\n- two"
    assert r.metadata["total_matches"] == 1
    assert r.truncated


def test_line_char_cap_marks_truncated(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("x" * 600 + " needle\n", encoding="utf-8")
    r = run(create_search_text_tool(tmp_path), {"pattern": "needle"})
    assert r.status is ResultStatus.OK
    assert r.truncated
    assert r.metadata["truncated"] is True
    assert r.text == "f.txt:1: " + "x" * LIMITS["max_line_chars"]


def test_symlink_and_agent_memory_skipped(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "o.txt").write_text("needle outside", encoding="utf-8")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "real.txt").write_text("needle real", encoding="utf-8")
    (workspace / "link.txt").symlink_to(outside / "o.txt")
    (workspace / "link_dir").symlink_to(outside)
    (workspace / ".agent_memory").mkdir()
    (workspace / ".agent_memory" / "secret.txt").write_text("needle memory", encoding="utf-8")

    r = run(create_search_text_tool(workspace), {"pattern": "needle"})
    assert "needle real" in r.text
    assert "outside" not in r.text
    assert "memory" not in r.text
    assert r.metadata["total_matches"] == 1


def test_backend_missing_error_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def missing() -> Path:
        raise FileNotFoundError("no packaged rg")

    monkeypatch.setattr(search_module, "rg_path", missing)
    r = run(create_search_text_tool(tmp_path), {"pattern": "x"})
    assert r.status is ResultStatus.ERROR
    assert r.is_error
    assert r.error_code == "search_backend_unavailable"


def test_worker_isolated_env_and_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # worker 必须 -I + 空白 env + cwd="/" + 独立进程组:不继承宿主 PYTHONPATH。
    (tmp_path / "f.txt").write_text("needle\n", encoding="utf-8")
    captured: dict[str, Any] = {}
    real = asyncio.create_subprocess_exec

    async def spy(*args: Any, **kwargs: Any) -> Any:
        captured["args"] = args
        captured["kwargs"] = kwargs
        return await real(*args, **kwargs)

    monkeypatch.setenv("PYTHONPATH", "/tmp/host-only-path")
    monkeypatch.setattr(search_module.asyncio, "create_subprocess_exec", spy)
    r = run(create_search_text_tool(tmp_path), {"pattern": "needle"})
    assert r.status is ResultStatus.OK
    assert captured["args"][1:3] == ("-I", "-c")
    assert captured["kwargs"]["env"] == {"PATH": os.defpath}
    assert "PYTHONPATH" not in captured["kwargs"]["env"]
    assert captured["kwargs"]["cwd"] == "/"
    assert captured["kwargs"]["start_new_session"] is True


def test_invalid_regex_goes_to_metadata_not_content(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("x\n", encoding="utf-8")
    r = run(create_search_text_tool(tmp_path), {"pattern": "(", "regex": True})
    assert r.status is ResultStatus.ERROR
    assert r.error_code == "invalid_input"
    # rg 的 stderr 只进 metadata(前 2000 字符),模型可见内容保持简短
    stderr = r.metadata["stderr"]
    assert isinstance(stderr, str) and stderr
    assert len(stderr) <= 2_000
    assert "(" not in r.text


def test_search_metadata_limits_matrix(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("needle\n", encoding="utf-8")
    r = run(create_search_text_tool(tmp_path), {"pattern": "needle"})
    assert r.metadata["backend"] == "ripgrep-bin"
    assert dict(r.metadata["limits"]) == LIMITS
    assert set(LIMITS) == {
        "max_file_bytes",
        "max_total_bytes",
        "max_files",
        "max_output_bytes",
        "max_backend_bytes",
        "max_line_chars",
    }


def test_search_exact_file_reports_searched_file_path(tmp_path: Path) -> None:
    (tmp_path / "f.txt").write_text("needle\n", encoding="utf-8")
    r = run(create_search_text_tool(tmp_path), {"pattern": "needle", "path": "f.txt"})
    assert r.metadata["searched_file_path"] == "f.txt"


def test_search_policy_and_resolve_use(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    tool = create_search_text_tool(workspace)
    assert tool.definition.name == "search_text"
    assert tool.static_effects == frozenset({ToolEffect.READ_WORKSPACE})
    assert tool.idempotent and tool.concurrency_safe
    assert tool.cancellation_mode is CancellationMode.COOPERATIVE
    assert tool.interrupt_behavior is InterruptBehavior.CANCEL
    assert tool.timeout_seconds == 15
    assert tool.max_model_output_bytes == 2_000_000
    root_use = tool.resolve_use({})
    assert root_use.effects == frozenset({ToolEffect.READ_WORKSPACE})
    assert root_use.targets[0].value == str(workspace.resolve())
    sub = workspace / "sub"
    sub.mkdir()
    assert tool.resolve_use({"path": "sub"}).targets[0].value == str(sub.resolve())


def test_search_schema_bounds(tmp_path: Path) -> None:
    tool = create_search_text_tool(tmp_path)
    with pytest.raises(ToolValidationError):
        tool.validate_input({"pattern": ""})
    with pytest.raises(ToolValidationError):
        tool.validate_input({"pattern": "x" * 2_001})
    with pytest.raises(ToolValidationError):
        tool.validate_input({"pattern": "x", "context_lines": 51})
    with pytest.raises(ToolValidationError):
        tool.validate_input({"pattern": "x", "max_results": 0})
    with pytest.raises(ToolValidationError):
        tool.validate_input({"pattern": "x", "max_results": 201})
    with pytest.raises(ToolValidationError):
        tool.validate_input({"pattern": "x", "glob": "g" * 513})
    with pytest.raises(ToolValidationError):
        tool.validate_input({"pattern": "x", "path": "p" * 4_097})
    with pytest.raises(ToolValidationError):
        tool.validate_input({"pattern": "x", "regex": "yes"})
    tool.validate_input({"pattern": "x", "regex": None, "context_lines": 50, "max_results": 200})


def test_search_rejects_escape_path(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    with pytest.raises(ToolValidationError):
        run(create_search_text_tool(workspace), {"pattern": "x", "path": "../outside"})


# ---------------------------------------------------------------------------
# R-T11-2:目录型读取工具逐条目执行 deny(decide 仍是唯一裁决函数)
# ---------------------------------------------------------------------------


def test_search_skips_denied_file(tmp_path: Path) -> None:
    (tmp_path / "public.txt").write_text("needle public", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("needle secret", encoding="utf-8")
    ctx = _permission_ctx(tmp_path, PathRule(path="secret.txt", access="deny"))
    r = run_with(
        create_search_text_tool(tmp_path),
        {"pattern": "needle"},
        ToolExecContext(permissions=ctx),
    )
    assert "needle public" in r.text
    assert "needle secret" not in r.text
    assert r.metadata["skipped_denied"] >= 1


def test_list_files_denied_subtree_is_invisible(tmp_path: Path) -> None:
    private = tmp_path / "private"
    private.mkdir()
    (private / "secret.txt").write_text("s", encoding="utf-8")
    (tmp_path / "public.txt").write_text("p", encoding="utf-8")
    ctx = _permission_ctx(tmp_path, PathRule(path="private", access="deny"))
    r = run_with(
        create_list_files_tool(tmp_path),
        {"path": "."},
        ToolExecContext(permissions=ctx),
    )
    relative = [line.split("\t")[2] for line in r.text.split("\n")]
    assert relative == ["public.txt"]


def test_denied_file_does_not_leak_and_decide_stays_denied(tmp_path: Path) -> None:
    (tmp_path / "secret.txt").write_text("TOPSECRET", encoding="utf-8")
    (tmp_path / "public.txt").write_text("public", encoding="utf-8")
    ctx = _permission_ctx(tmp_path, PathRule(path="secret.txt", access="deny"))
    exec_ctx = ToolExecContext(permissions=ctx)

    # 执行侧:目录级搜索把 denied 文件排除,内容原文不泄露
    searched = run_with(
        create_search_text_tool(tmp_path), {"pattern": "TOPSECRET"}, exec_ctx
    )
    assert "TOPSECRET" not in searched.text
    assert searched.metadata["skipped_denied"] >= 1

    # decide 仍是唯一裁决函数:read/search 的精确 target 都被 denied——
    # 执行侧过滤没有、也不能把决策翻成 allow。
    read_tool = create_read_tool(tmp_path)
    search_tool = create_search_text_tool(tmp_path)
    assert permission.decide(read_tool, read_tool.resolve_use({"path": "secret.txt"}), ctx).is_deny
    assert permission.decide(search_tool, search_tool.resolve_use({"path": "secret.txt"}), ctx).is_deny

    # 过滤复用同一份规则代数(不是第二套合并器)
    resolved = resolve_policy(ctx.policy, tmp_path)
    assert path_access(resolved, tmp_path / "secret.txt", write=False) is False


def test_pipeline_forwards_permissions_to_search(
    store: Any, bus: Any, workspace: Path
) -> None:
    # R-T11-2 接线:execute 必须把 ctx.permissions 放进 ToolExecContext,
    # 否则目录级搜索在真实管道里仍会读穿 deny 子路径。
    (workspace / "secret.txt").write_text("TOPSECRET", encoding="utf-8")
    (workspace / "public.txt").write_text("needle public", encoding="utf-8")
    policy = PermissionPolicy(filesystem=(PathRule(path="secret.txt", access="deny"),))
    permissions = PermissionContext(workspace=str(workspace), policy=policy, mode="on-request")
    created = create_turn(store, turn_id="turn-search", thread_id="thread-search", binding={})
    ctx = ExecCtx(
        store=store,
        bus=bus,
        workspace=workspace,
        tools={"search_text": create_search_text_tool(workspace)},
        permissions=permissions,
        turn_id=created.turn_id,
        thread_id=created.thread_id,
        policy_revision=policy.revision,
    )
    result = asyncio.run(
        execute(
            ToolCall(
                tool_call_id="c1",
                tool_name="search_text",
                arguments={"pattern": "TOPSECRET"},
            ),
            ctx,
        )
    )
    assert result.status is ResultStatus.OK
    assert "TOPSECRET" not in result.text
    assert result.metadata["skipped_denied"] >= 1
