"""T3:hooks 协议数据与 hooks.json 加载校验(设计 §5.14;3B 笔记 §4)。

钉住五件事:
1. ``HookSpec``/``HookOutcome`` 的字段面、校验与常量逐字(移植旧
   ``LifecycleHook.__post_init__``;但只保留 command hook 一种形态);
2. ``load_host_hooks(workspace, path=None)`` 的路径来源序(显式参数 >
   ``PRAXIS_HOOKS_CONFIG`` > ``host_config_dir()/hooks.json``)与缺失语义
   (R-T3-c:显式/环境路径缺失 ⇒ 带路径 ValueError;仅隐式默认缺失 ⇒ ``()``);
3. workspace 由调用方显式传入并 realpath 规范化(R-T3-a),绝不使用 cwd;
   旧 :110-153 的逐条拒绝面:绝对路径、workspace 之外、64 KiB 读上限哨兵、
   ≤16 项、数组形状、未知键、Stop(阶段 6 文案)、``argv[0]`` 绝对可执行且
   不在 workspace 内、``(event,name)`` 唯一、文件卫生(属主 uid + 非
   group/world 可写);
4. ``HookSpec.directory``(R-T3-b)= 配置文件父目录绝对字符串,同一配置的所有
   spec 相同,并作为 revision identity 的输入;
5. ``revision = sha256(canonical JSON({argv,directory,revision}))[:16]`` 的
   确定性与规范化(唯一实现所在模块是 adapters/hooks/config.py)。
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from praxis.adapters.hooks.config import load_host_hooks
from praxis.protocol.floor import host_config_dir
from praxis.protocol.hooks import (
    MAX_FEEDBACK_BYTES,
    MAX_HOOK_STDOUT_BYTES,
    MAX_HOOKS,
    MAX_HOOKS_CONFIG_BYTES,
    HookOutcome,
    HookSpec,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_SPEC_DIRECTORY = str(Path(sys.executable).resolve().parent)


def _argv() -> list[str]:
    """真实可执行 + 无害参数;加载器只做 resolve(strict=True),不执行它。"""
    return [sys.executable, "-c", "print('{}')"]


def _entry(**overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {"event": "PreToolUse", "name": "policy", "argv": _argv()}
    entry.update(overrides)
    return entry


def _write_config(tmp_path: Path, entries: object, *, mode: int = 0o600) -> Path:
    """在 workspace 之外落一个卫生的配置文件(父目录 0700)。"""
    host = tmp_path / "host"
    host.mkdir(parents=True, exist_ok=True)
    host.chmod(0o700)
    path = host / "hooks.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    path.chmod(mode)
    return path


def _write_raw_config(tmp_path: Path, raw: bytes, *, mode: int = 0o600) -> Path:
    host = tmp_path / "host"
    host.mkdir(parents=True, exist_ok=True)
    host.chmod(0o700)
    path = host / "hooks.json"
    path.write_bytes(raw)
    path.chmod(mode)
    return path


def _canonical_revision(argv: tuple[str, ...], directory: Path, declared: str) -> str:
    payload = json.dumps(
        {"argv": list(argv), "directory": str(directory), "revision": declared},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    """真实 workspace;测试刻意不 chdir,证明加载边界只看显式参数。"""
    root = tmp_path / "workspace"
    root.mkdir()
    return root


# ---------------------------------------------------------------------------
# 常量与 HookSpec
# ---------------------------------------------------------------------------


def test_constants_are_verbatim() -> None:
    assert MAX_HOOKS == 16
    assert MAX_FEEDBACK_BYTES == 2048
    assert MAX_HOOK_STDOUT_BYTES == 16_385
    assert MAX_HOOKS_CONFIG_BYTES == 65_537


def test_hook_spec_defaults_and_normalizes_selectors() -> None:
    spec = HookSpec(
        "policy",
        "1",
        "PreToolUse",
        (sys.executable,),
        _SPEC_DIRECTORY,
        tool_names=["beta", "alpha", "alpha"],
        effects=["write_workspace", "write_workspace"],
    )
    assert spec.timeout_seconds == 5.0
    assert spec.directory == _SPEC_DIRECTORY
    assert spec.tool_names == frozenset({"alpha", "beta"})
    assert spec.effects == frozenset({"write_workspace"})


def test_hook_spec_accepts_empty_selectors_as_no_filter() -> None:
    spec = HookSpec("policy", "1", "PostToolUse", (sys.executable,), _SPEC_DIRECTORY)
    assert spec.tool_names == frozenset()
    assert spec.effects == frozenset()


@pytest.mark.parametrize("event", ["Stop", "Other", "pretooluse", ""])
def test_hook_spec_rejects_unsupported_events(event: str) -> None:
    with pytest.raises(ValueError, match="invalid hook event"):
        HookSpec("policy", "1", event, (sys.executable,), _SPEC_DIRECTORY)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("name", "revision"),
    [
        ("", "1"),
        ("   ", "1"),
        ("policy", ""),
        ("policy", "   "),
        ("x" * 129, "1"),
        ("policy", "x" * 129),
    ],
)
def test_hook_spec_rejects_bad_name_or_revision(name: str, revision: str) -> None:
    with pytest.raises(ValueError, match="1–128 characters"):
        HookSpec(name, revision, "PreToolUse", (sys.executable,), _SPEC_DIRECTORY)


@pytest.mark.parametrize("directory", ["", "relative/dir", ".", "\0bad"])
def test_hook_spec_rejects_non_absolute_directory(directory: str) -> None:
    with pytest.raises(ValueError, match="non-empty absolute path"):
        HookSpec("policy", "1", "PreToolUse", (sys.executable,), directory)


@pytest.mark.parametrize("seconds", [0, 0.0, -1, 30.0001, float("inf"), float("nan"), True, "5"])
def test_hook_spec_rejects_bad_timeout(seconds: object) -> None:
    with pytest.raises(ValueError, match="greater than 0 and at most 30 seconds"):
        HookSpec(  # type: ignore[arg-type]
            "policy", "1", "PreToolUse", (sys.executable,), _SPEC_DIRECTORY, timeout_seconds=seconds
        )


def test_hook_spec_accepts_upper_timeout_boundary() -> None:
    spec = HookSpec(
        "policy", "1", "PreToolUse", (sys.executable,), _SPEC_DIRECTORY, timeout_seconds=30
    )
    assert spec.timeout_seconds == 30.0


@pytest.mark.parametrize("argv", [(), ("python",), ("",), (sys.executable, "bad\0arg")])
def test_hook_spec_rejects_bad_argv(argv: tuple[str, ...]) -> None:
    with pytest.raises(ValueError):
        HookSpec("policy", "1", "PreToolUse", argv, _SPEC_DIRECTORY)


@pytest.mark.parametrize("value", ["", " ", " leading", "trailing ", 1])
def test_hook_spec_rejects_non_exact_tool_names(value: object) -> None:
    with pytest.raises(ValueError, match="exact non-empty names"):
        HookSpec(  # type: ignore[arg-type]
            "policy", "1", "PreToolUse", (sys.executable,), _SPEC_DIRECTORY,
            tool_names=frozenset({value}),
        )


def test_hook_spec_rejects_unknown_effect() -> None:
    with pytest.raises(ValueError, match="valid ToolEffect"):
        HookSpec(
            "policy", "1", "PreToolUse", (sys.executable,), _SPEC_DIRECTORY,
            effects=frozenset({"not_an_effect"}),
        )


# ---------------------------------------------------------------------------
# HookOutcome
# ---------------------------------------------------------------------------


def test_hook_outcome_truncates_feedback_to_2048_bytes() -> None:
    outcome = HookOutcome("blocked", "é" * 2000)
    assert len(outcome.feedback.encode("utf-8")) <= MAX_FEEDBACK_BYTES
    assert outcome.source == ""
    assert outcome.status == "blocked"


def test_hook_outcome_keeps_feedback_at_limit() -> None:
    feedback = "a" * MAX_FEEDBACK_BYTES
    assert HookOutcome("allowed", feedback).feedback == feedback


def test_hook_outcome_rejects_unknown_status() -> None:
    with pytest.raises(ValueError, match="invalid hook status"):
        HookOutcome("permitted")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 良好配置 roundtrip 与 revision
# ---------------------------------------------------------------------------


def test_loads_valid_config_roundtrip(
    tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_config(
        tmp_path,
        [
            _entry(
                name="pre",
                timeout_seconds=2.5,
                effects=["write_workspace", "read_workspace", "write_workspace"],
            ),
            _entry(event="PostToolUse", name="post", tool_names=["read_file"], revision="v2"),
        ],
    )
    monkeypatch.setenv("PRAXIS_HOOKS_CONFIG", str(config))

    hooks = load_host_hooks(workspace)
    assert [hook.name for hook in hooks] == ["pre", "post"]
    assert [hook.event for hook in hooks] == ["PreToolUse", "PostToolUse"]
    assert hooks[0].effects == frozenset({"write_workspace", "read_workspace"})
    assert hooks[0].tool_names == frozenset()
    assert hooks[0].timeout_seconds == 2.5
    assert hooks[1].tool_names == frozenset({"read_file"})
    assert hooks[1].timeout_seconds == 5.0
    assert hooks[0].argv == (str(Path(sys.executable).resolve()), *_argv()[1:])
    # R-T3-b:directory 是配置文件父目录,同一 load 的所有 spec 相同。
    assert hooks[0].directory == hooks[1].directory == str(config.parent)


def test_revision_is_deterministic_canonical_sha256_prefix(
    tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_config(tmp_path, [_entry(revision="declared-v1")])
    monkeypatch.setenv("PRAXIS_HOOKS_CONFIG", str(config))

    first = load_host_hooks(workspace)[0]
    second = load_host_hooks(workspace)[0]
    executable = str(Path(sys.executable).resolve())
    expected = _canonical_revision(tuple([executable, *_argv()[1:]]), config.parent, "declared-v1")

    assert first.revision == second.revision == expected
    assert first.revision == _canonical_revision(first.argv, Path(first.directory), "declared-v1")
    assert len(first.revision) == 16
    assert all(character in "0123456789abcdef" for character in first.revision)


def test_declared_revision_changes_computed_revision(
    tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first_config = _write_config(tmp_path, [_entry(revision="v1")])
    monkeypatch.setenv("PRAXIS_HOOKS_CONFIG", str(first_config))
    first = load_host_hooks(workspace)[0].revision

    second_config = _write_config(tmp_path, [_entry(revision="v2")])
    monkeypatch.setenv("PRAXIS_HOOKS_CONFIG", str(second_config))
    second = load_host_hooks(workspace)[0].revision

    assert first != second


# ---------------------------------------------------------------------------
# 路径来源序 / 缺失语义(R-T3-c)
# ---------------------------------------------------------------------------


def test_missing_default_file_returns_empty(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PRAXIS_HOOKS_CONFIG", raising=False)
    assert load_host_hooks(workspace) == ()


def test_missing_explicit_path_is_rejected_with_path(
    tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PRAXIS_HOOKS_CONFIG", raising=False)
    missing = tmp_path / "host" / "absent.json"
    with pytest.raises(ValueError, match="does not exist") as raised:
        load_host_hooks(workspace, path=missing)
    assert str(missing) in str(raised.value)


def test_missing_env_path_is_rejected_with_path(
    tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "host" / "absent.json"
    monkeypatch.setenv("PRAXIS_HOOKS_CONFIG", str(missing))
    with pytest.raises(ValueError, match="does not exist") as raised:
        load_host_hooks(workspace)
    assert str(missing) in str(raised.value)


def test_defaults_to_host_config_dir(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PRAXIS_HOOKS_CONFIG", raising=False)
    directory = host_config_dir()
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o700)
    config = directory / "hooks.json"
    config.write_text(json.dumps([_entry(name="default")]), encoding="utf-8")
    config.chmod(0o600)

    assert [hook.name for hook in load_host_hooks(workspace)] == ["default"]


def test_env_overrides_default_and_explicit_overrides_env(
    tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    directory = host_config_dir()
    directory.mkdir(parents=True, exist_ok=True)
    directory.chmod(0o700)
    default = directory / "hooks.json"
    default.write_text(json.dumps([_entry(name="default")]), encoding="utf-8")
    default.chmod(0o600)

    env_config = _write_config(tmp_path / "env", [_entry(name="env")])
    monkeypatch.setenv("PRAXIS_HOOKS_CONFIG", str(env_config))
    assert [hook.name for hook in load_host_hooks(workspace)] == ["env"]

    explicit = _write_config(tmp_path / "explicit", [_entry(name="explicit")])
    assert [hook.name for hook in load_host_hooks(workspace, path=explicit)] == ["explicit"]


# ---------------------------------------------------------------------------
# 路径形状 / workspace 边界(R-T3-a)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("configure", ["env", "explicit"])
def test_relative_config_path_is_rejected(
    workspace: Path, monkeypatch: pytest.MonkeyPatch, configure: str
) -> None:
    if configure == "env":
        monkeypatch.setenv("PRAXIS_HOOKS_CONFIG", "relative/hooks.json")
        with pytest.raises(ValueError, match="absolute file outside the workspace"):
            load_host_hooks(workspace)
    else:
        with pytest.raises(ValueError, match="absolute file outside the workspace"):
            load_host_hooks(workspace, path=Path("relative/hooks.json"))


def test_config_inside_workspace_is_rejected_even_when_cwd_is_outside(
    tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    config = workspace / "hooks.json"
    config.write_text(json.dumps([_entry()]), encoding="utf-8")
    config.chmod(0o600)
    monkeypatch.setenv("PRAXIS_HOOKS_CONFIG", str(config))

    with pytest.raises(ValueError, match="outside the workspace"):
        load_host_hooks(workspace)


def test_config_outside_workspace_is_accepted_even_when_cwd_is_inside(
    tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(workspace)
    config = _write_config(tmp_path, [_entry(name="outside")])
    monkeypatch.setenv("PRAXIS_HOOKS_CONFIG", str(config))

    assert [hook.name for hook in load_host_hooks(workspace)] == ["outside"]


def test_workspace_is_realpath_normalized(
    tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    alias = tmp_path / "workspace-alias"
    alias.symlink_to(workspace, target_is_directory=True)
    config = workspace / "hooks.json"
    config.write_text(json.dumps([_entry()]), encoding="utf-8")
    config.chmod(0o600)
    monkeypatch.setenv("PRAXIS_HOOKS_CONFIG", str(config))

    with pytest.raises(ValueError, match="outside the workspace"):
        load_host_hooks(alias)


def test_argv_executable_inside_workspace_is_rejected(
    tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = workspace / "hook.sh"
    executable.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    executable.chmod(0o700)
    config = _write_config(tmp_path, [_entry(argv=[str(executable)])])
    monkeypatch.setenv("PRAXIS_HOOKS_CONFIG", str(config))

    with pytest.raises(ValueError, match="outside the workspace"):
        load_host_hooks(workspace)


def test_relative_argv_is_rejected(
    tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_config(tmp_path, [_entry(argv=["python", "-c", "print(1)"])])
    monkeypatch.setenv("PRAXIS_HOOKS_CONFIG", str(config))

    with pytest.raises(ValueError, match="absolute executable outside the workspace"):
        load_host_hooks(workspace)


def test_missing_argv_executable_is_rejected(
    tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_config(tmp_path, [_entry(argv=[str(tmp_path / "does-not-exist")])])
    monkeypatch.setenv("PRAXIS_HOOKS_CONFIG", str(config))

    with pytest.raises(ValueError, match="absolute executable outside the workspace"):
        load_host_hooks(workspace)


# ---------------------------------------------------------------------------
# 条目形状 / 数量 / 唯一性 / Stop
# ---------------------------------------------------------------------------


def test_unknown_entry_key_is_rejected(
    tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_config(tmp_path, [_entry(handler="callable")])
    monkeypatch.setenv("PRAXIS_HOOKS_CONFIG", str(config))

    with pytest.raises(ValueError, match="event, name and argv are required"):
        load_host_hooks(workspace)


@pytest.mark.parametrize("missing", ["event", "name", "argv"])
def test_missing_required_entry_key_is_rejected(
    tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch, missing: str
) -> None:
    entry = _entry()
    del entry[missing]
    config = _write_config(tmp_path, [entry])
    monkeypatch.setenv("PRAXIS_HOOKS_CONFIG", str(config))

    with pytest.raises(ValueError, match="event, name and argv are required"):
        load_host_hooks(workspace)


def test_stop_event_is_rejected_with_phase_six_message(
    tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_config(tmp_path, [_entry(event="Stop")])
    monkeypatch.setenv("PRAXIS_HOOKS_CONFIG", str(config))

    with pytest.raises(ValueError, match="stop hooks land in phase 6"):
        load_host_hooks(workspace)


def test_unknown_event_is_rejected(
    tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_config(tmp_path, [_entry(event="AfterToolUse")])
    monkeypatch.setenv("PRAXIS_HOOKS_CONFIG", str(config))

    with pytest.raises(ValueError, match="invalid hook event"):
        load_host_hooks(workspace)


def test_duplicate_event_name_pair_is_rejected(
    tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_config(
        tmp_path,
        [_entry(name="same"), _entry(name="same", event="PostToolUse"), _entry(name="same")],
    )
    monkeypatch.setenv("PRAXIS_HOOKS_CONFIG", str(config))

    with pytest.raises(ValueError, match="unique within each event"):
        load_host_hooks(workspace)


def test_same_name_across_events_is_allowed(
    tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_config(
        tmp_path,
        [_entry(name="same"), _entry(name="same", event="PostToolUse")],
    )
    monkeypatch.setenv("PRAXIS_HOOKS_CONFIG", str(config))

    assert [hook.event for hook in load_host_hooks(workspace)] == ["PreToolUse", "PostToolUse"]


def test_empty_array_returns_no_hooks(
    tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_config(tmp_path, [])
    monkeypatch.setenv("PRAXIS_HOOKS_CONFIG", str(config))

    assert load_host_hooks(workspace) == ()


def test_more_than_sixteen_entries_is_rejected(
    tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_config(tmp_path, [_entry(name=f"hook-{index}") for index in range(17)])
    monkeypatch.setenv("PRAXIS_HOOKS_CONFIG", str(config))

    with pytest.raises(ValueError, match="at most 16 entries"):
        load_host_hooks(workspace)


@pytest.mark.parametrize("document", [{}, "not-an-array", 7, None])
def test_non_array_document_is_rejected(
    tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch, document: object
) -> None:
    config = _write_config(tmp_path, document)
    monkeypatch.setenv("PRAXIS_HOOKS_CONFIG", str(config))

    with pytest.raises(ValueError, match="array of at most 16 entries"):
        load_host_hooks(workspace)


def test_non_object_entry_is_rejected(
    tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_config(tmp_path, ["not-an-object"])
    monkeypatch.setenv("PRAXIS_HOOKS_CONFIG", str(config))

    with pytest.raises(ValueError, match="event, name and argv are required"):
        load_host_hooks(workspace)


def test_malformed_json_is_rejected(
    tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_raw_config(tmp_path, b"{not json")
    monkeypatch.setenv("PRAXIS_HOOKS_CONFIG", str(config))

    with pytest.raises(ValueError, match="valid JSON"):
        load_host_hooks(workspace)


def test_invalid_timeout_in_config_is_rejected(
    tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _write_config(tmp_path, [_entry(timeout_seconds=31)])
    monkeypatch.setenv("PRAXIS_HOOKS_CONFIG", str(config))

    with pytest.raises(ValueError, match="greater than 0 and at most 30 seconds"):
        load_host_hooks(workspace)


# ---------------------------------------------------------------------------
# 文件卫生与字节上限
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", [0o664, 0o646, 0o666, 0o602])
def test_group_or_world_writable_config_is_rejected(
    tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch, mode: int
) -> None:
    config = _write_config(tmp_path, [_entry()], mode=mode)
    monkeypatch.setenv("PRAXIS_HOOKS_CONFIG", str(config))

    with pytest.raises(ValueError, match="group/world writable"):
        load_host_hooks(workspace)


def test_config_over_64_kib_is_rejected(
    tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = b" " * MAX_HOOKS_CONFIG_BYTES
    config = _write_raw_config(tmp_path, raw)
    monkeypatch.setenv("PRAXIS_HOOKS_CONFIG", str(config))

    with pytest.raises(ValueError, match="exceeds 64 KiB"):
        load_host_hooks(workspace)


def test_config_at_64_kib_boundary_is_accepted(
    tmp_path: Path, workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = b"[]" + b" " * (MAX_HOOKS_CONFIG_BYTES - 1 - 2)
    assert len(raw) == MAX_HOOKS_CONFIG_BYTES - 1
    config = _write_raw_config(tmp_path, raw)
    monkeypatch.setenv("PRAXIS_HOOKS_CONFIG", str(config))

    assert load_host_hooks(workspace) == ()
