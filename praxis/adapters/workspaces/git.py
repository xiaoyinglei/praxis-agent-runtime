"""Praxis 子工作区适配器:git worktree 快照实现(阶段4 T7;设计 §5.13)。

``GitWorktreeProvisioner`` 是 protocol.workspaces.ChildWorkspaceProvisioner 的
唯一生产实现(composition 绑定)。控制器裁决:

- 落点 ``host_config_dir()/worktrees/<sha256(resolve(parent))[:16]>/<child>``
  ——库外、宿主配置目录、安全地板内(代理不可写);
- ``git -C parent worktree add --detach <path> HEAD``——detach 快照,父工作区
  未提交字节不入快照;integrate 走 diff/patch,不引入 branch merge;
- diff bundle = ``git diff HEAD`` + ``git status --porcelain -z`` 的 untracked
  清单,自有格式(格式常量唯一在本模块);``.praxis``/``.git`` 在 diff 与
  apply 路径集都排除(子 always 策略文件绝不渗回父);
- apply 逐文件 preimage sha256 复检:任一不一致 → conflicts 且整组不写;
  ``git apply --check`` 后 ``git apply``,失败整组拒(``patch_failed`` parity);
- 非 git 工作区 prepare 抛 ``ChildWorkspaceRequiresGit``(Q5)。

子进程纪律(B33):固定 argv 数组、不经 shell;env 基底由 composition 注入
``tools.builtins.child_env``(adapters 不 import tools),本模块只追加最小
GIT_* 键。宿主 ``os.environ`` 从不读取。
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from praxis.protocol.floor import host_config_dir
from praxis.protocol.sandbox import EnvBuilder
from praxis.protocol.workspaces import ApplyResult, ChildWorkspaceRequiresGit

# diff bundle 自有格式:magic + 4-byte header 长度 + JSON header + tracked diff
# + 每个 untracked 文件原始字节(长度/摘要见 header)。格式常量只此一处。
_BUNDLE_MAGIC = b"PRAXIS-WORKSPACE-BUNDLE\x00v1\x00"
_BUNDLE_VERSION = 1

_GIT_TIMEOUT_SECONDS = 60
_MAX_ERROR_CHARS = 512

# floor 任意路径成分命中即排除(.praxis deny / .git read-only);glob 形态覆盖嵌套。
_EXCLUDED_COMPONENTS = frozenset({".praxis", ".git"})
_EXCLUDED_PATHSPECS = (
    ":(exclude).praxis",
    ":(exclude).praxis/**",
    ":(exclude).git",
    ":(exclude).git/**",
    ":(glob,exclude)**/.praxis/**",
    ":(glob,exclude)**/.git/**",
)

# child_env 已注入 GIT_CONFIG_GLOBAL/NOSYSTEM;这里只补交互面最小键(B33:
# 不复制宿主 env,常量白名单之外无路)。
_GIT_ENV_OVERRIDES: Mapping[str, str] = {"GIT_TERMINAL_PROMPT": "0"}


@dataclass(frozen=True, slots=True)
class _UntrackedFile:
    """bundle 中的一个 untracked 文件(数据内联,apply 直接重建)。"""

    path: str
    mode: str  # 100644 / 100755 / 120000(symlink)
    sha256: str
    data: bytes


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _bounded(stderr: bytes) -> str:
    return stderr.decode("utf-8", errors="replace")[:_MAX_ERROR_CHARS].strip()


def _repo_relative(raw: str) -> PurePosixPath:
    """bundle 路径校验:相对、无空/. /.. 成分(apply 绝不越出 parent_root)。"""
    rel = PurePosixPath(raw)
    if rel.is_absolute() or not rel.parts or any(part in ("", ".", "..") for part in rel.parts):
        raise ValueError(f"unsafe workspace bundle path: {raw!r}")
    return rel


def _is_excluded(raw: str) -> bool:
    return any(part.casefold() in _EXCLUDED_COMPONENTS for part in PurePosixPath(raw).parts)


def _validated_child_id(child_thread_id: str) -> str:
    if not child_thread_id or child_thread_id in {".", ".."} or "/" in child_thread_id or "\x00" in child_thread_id:
        raise ValueError(f"invalid child_thread_id for workspace directory: {child_thread_id!r}")
    return child_thread_id


def _file_digest(path: Path) -> str | None:
    """preimage 摘要;缺失/非普通文件 → None(与 preimages 的 None 语义一致)。"""
    try:
        if not path.is_file():
            return None
        return _sha256(path.read_bytes())
    except OSError:
        return None


def _untracked_paths(status: bytes) -> tuple[str, ...]:
    """解析 ``git status --porcelain -z`` 的 ``??`` 记录(跳过 rename 的旧路径字段)。"""
    fields = status.split(b"\0")
    paths: list[str] = []
    index = 0
    while index < len(fields):
        record = fields[index]
        if not record:
            index += 1
            continue
        if len(record) < 4:
            raise ValueError(f"malformed git status record: {record!r}")
        code = record[:2]
        raw_path = record[3:]
        # -z rename/copy: 下一字段是原路径,不是新记录。
        index += 2 if code[:1] in (b"R", b"C") else 1
        if code == b"??":
            paths.append(os.fsdecode(raw_path))
    return tuple(paths)


def _encode_bundle(
    tracked_diff: bytes,
    tracked_paths: Sequence[str],
    untracked: Sequence[_UntrackedFile],
) -> bytes:
    header = {
        "version": _BUNDLE_VERSION,
        "tracked_diff_size": len(tracked_diff),
        "tracked_paths": list(tracked_paths),
        "files": [
            {
                "path": file.path,
                "mode": file.mode,
                "sha256": file.sha256,
                "size": len(file.data),
            }
            for file in untracked
        ],
    }
    # ensure_ascii:路径可能含 surrogateescape 字节,ASCII 转义保证 header 可编码。
    header_bytes = json.dumps(
        header, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    ).encode("ascii")
    parts = [_BUNDLE_MAGIC, struct.pack(">I", len(header_bytes)), header_bytes, tracked_diff]
    parts.extend(file.data for file in untracked)
    return b"".join(parts)


def _decode_bundle(bundle: bytes) -> tuple[bytes, tuple[str, ...], tuple[_UntrackedFile, ...]]:
    if not bundle.startswith(_BUNDLE_MAGIC):
        raise ValueError("invalid workspace bundle: bad magic")
    offset = len(_BUNDLE_MAGIC)
    if len(bundle) < offset + 4:
        raise ValueError("invalid workspace bundle: truncated header size")
    (header_size,) = struct.unpack(">I", bundle[offset : offset + 4])
    offset += 4
    header_end = offset + header_size
    if len(bundle) < header_end:
        raise ValueError("invalid workspace bundle: truncated header")
    try:
        header = json.loads(bundle[offset:header_end].decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid workspace bundle header: {exc}") from exc
    offset = header_end
    if header.get("version") != _BUNDLE_VERSION:
        raise ValueError(f"unsupported workspace bundle version: {header.get('version')!r}")
    tracked_diff_size = header["tracked_diff_size"]
    tracked_diff = bundle[offset : offset + tracked_diff_size]
    if len(tracked_diff) != tracked_diff_size:
        raise ValueError("invalid workspace bundle: truncated tracked diff")
    offset += tracked_diff_size
    untracked: list[_UntrackedFile] = []
    for meta in header["files"]:
        size = meta["size"]
        data = bundle[offset : offset + size]
        offset += size
        if len(data) != size or _sha256(data) != meta["sha256"]:
            raise ValueError(f"invalid workspace bundle: corrupt payload for {meta['path']!r}")
        untracked.append(
            _UntrackedFile(
                path=meta["path"],
                mode=meta["mode"],
                sha256=meta["sha256"],
                data=data,
            )
        )
    if offset != len(bundle):
        raise ValueError("invalid workspace bundle: trailing bytes")
    return tracked_diff, tuple(header["tracked_paths"]), tuple(untracked)


class GitWorktreeProvisioner:
    """git worktree 快照的 write 子工作区 provisioner(protocol 结构匹配)。

    ``env_builder`` 由 composition 绑定 ``tools.builtins.child_env`` 唯一实现
    (R-T15-7 注入纪律:adapters 不 import tools)。
    """

    def __init__(self, env_builder: EnvBuilder) -> None:
        self._env_builder = env_builder

    # -- 子进程唯一出口 ------------------------------------------------------

    def _git(
        self,
        cwd: Path,
        workspace: Path,
        args: Sequence[str],
        *,
        stdin: bytes | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        """固定 argv 数组运行 git(B33:无 shell;env 只来自注入构造器)。"""
        with tempfile.TemporaryDirectory(prefix="praxis-workspaces-git-") as scratch:
            env = dict(self._env_builder(Path(scratch), workspace))
            env.update(_GIT_ENV_OVERRIDES)
            return subprocess.run(
                ["git", "-C", str(cwd), *args],
                input=stdin,
                capture_output=True,
                check=False,
                env=env,
                timeout=_GIT_TIMEOUT_SECONDS,
            )

    # -- prepare -------------------------------------------------------------

    def prepare(self, parent_root: Path, child_thread_id: str, mode: str) -> Path:
        """在库外宿主目录建 detach 快照;非 git → ``ChildWorkspaceRequiresGit``。"""
        parent = Path(parent_root).resolve()
        child_id = _validated_child_id(child_thread_id)
        probe = self._git(parent, parent, ("rev-parse", "--show-toplevel"))
        if probe.returncode != 0:
            detail = _bounded(probe.stderr)
            message = f"write-mode child workspace requires a git repository: {parent}"
            raise ChildWorkspaceRequiresGit(f"{message}: {detail}" if detail else message)
        repository_root = Path(os.fsdecode(probe.stdout.strip())).resolve()
        if repository_root != parent:
            # 与旧快照纪律 parity:子 worktree 以 HEAD 检出整仓;工作区若是仓库
            # 子目录,快照根会与 Session 工作区不一致——明确拒绝而非静默漂移。
            raise RuntimeError(
                "write-mode child workspace requires the git repository root: "
                f"{parent} (repository root is {repository_root})"
            )
        digest = _sha256(os.fsencode(str(parent)))[:16]
        base = Path(os.path.abspath(host_config_dir())) / "worktrees" / digest
        base.mkdir(parents=True, exist_ok=True)
        workspace = base / child_id
        if workspace.exists():
            raise RuntimeError(f"child workspace already exists: {workspace}")
        added = self._git(
            parent, parent, ("worktree", "add", "--detach", str(workspace), "HEAD")
        )
        if added.returncode != 0:
            raise RuntimeError(f"git worktree add failed: {_bounded(added.stderr)}")
        return workspace

    # -- diff ----------------------------------------------------------------

    def diff(self, ws: Path) -> bytes:
        """子工作区相对 HEAD 的 bundle(tracked diff + untracked;排除 .praxis/.git)。"""
        workspace = Path(ws)
        status = self._git(
            workspace, workspace, ("status", "--porcelain", "-z", "--untracked-files=all")
        )
        if status.returncode != 0:
            raise RuntimeError(f"git status failed: {_bounded(status.stderr)}")
        untracked: list[_UntrackedFile] = []
        for raw_path in _untracked_paths(status.stdout):
            if _is_excluded(raw_path):
                continue
            _repo_relative(raw_path)
            target = workspace / Path(*PurePosixPath(raw_path).parts)
            if target.is_symlink():
                data = os.fsencode(os.readlink(target))
                mode = "120000"
            else:
                data = target.read_bytes()
                mode = "100755" if target.stat().st_mode & 0o111 else "100644"
            untracked.append(_UntrackedFile(raw_path, mode, _sha256(data), data))
        untracked.sort(key=lambda file: file.path)

        diff_args = (
            "diff",
            "--no-ext-diff",
            "--binary",
            "--no-renames",
            "HEAD",
            "--",
            ".",
            *_EXCLUDED_PATHSPECS,
        )
        tracked_diff = self._git(workspace, workspace, diff_args)
        if tracked_diff.returncode != 0:
            raise RuntimeError(f"git diff failed: {_bounded(tracked_diff.stderr)}")
        names = self._git(
            workspace,
            workspace,
            (
                "diff",
                "--no-ext-diff",
                "--no-renames",
                "--name-only",
                "-z",
                "HEAD",
                "--",
                ".",
                *_EXCLUDED_PATHSPECS,
            ),
        )
        if names.returncode != 0:
            raise RuntimeError(f"git diff --name-only failed: {_bounded(names.stderr)}")
        tracked_paths = sorted(
            path
            for raw in names.stdout.split(b"\0")
            if raw and not _is_excluded(path := os.fsdecode(raw))
        )
        return _encode_bundle(tracked_diff.stdout, tracked_paths, untracked)

    # -- apply ---------------------------------------------------------------

    def apply(
        self, parent_root: Path, patch: bytes, *, preimages: Mapping[str, str | None]
    ) -> ApplyResult:
        """preimage 逐文件复检 → 整组 apply;冲突或 patch 失败都不半写。"""
        parent = Path(parent_root).resolve()
        tracked_diff, tracked_paths, untracked = _decode_bundle(patch)
        bundle_paths: list[str] = list(tracked_paths)
        bundle_paths.extend(file.path for file in untracked)
        for raw_path in bundle_paths:
            _repo_relative(raw_path)
            if _is_excluded(raw_path):
                raise ValueError(f"workspace bundle must not contain excluded path: {raw_path!r}")
        touched = sorted(set(bundle_paths))

        conflicts: list[str] = []
        for raw_path in touched:
            expected = preimages[raw_path] if raw_path in preimages else None
            actual = _file_digest(parent / Path(*PurePosixPath(raw_path).parts))
            if actual != expected:
                conflicts.append(raw_path)
        if conflicts:
            # 双方字节都在:parent 不动,子快照/bundle 也未被消费。
            return ApplyResult(applied=[], conflicts=conflicts)

        if tracked_diff:
            checked = self._git(
                parent, parent, ("apply", "--check", "--binary"), stdin=tracked_diff
            )
            if checked.returncode != 0:
                raise RuntimeError(
                    f"patch_failed: git apply --check rejected the bundle: {_bounded(checked.stderr)}"
                )
            applied = self._git(parent, parent, ("apply", "--binary"), stdin=tracked_diff)
            if applied.returncode != 0:
                raise RuntimeError(f"patch_failed: git apply failed: {_bounded(applied.stderr)}")

        for file in untracked:
            target = parent / Path(*PurePosixPath(file.path).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            if file.mode == "120000":
                os.symlink(os.fsdecode(file.data), target)
            else:
                target.write_bytes(file.data)
                target.chmod(0o755 if file.mode == "100755" else 0o644)
        return ApplyResult(applied=touched, conflicts=[])

    # -- discard -------------------------------------------------------------

    def discard(self, ws: Path) -> None:
        """移除快照 worktree;路径已消失 = 幂等成功。"""
        workspace = Path(ws)
        if not workspace.exists():
            return
        removed = self._git(
            workspace, workspace, ("worktree", "remove", "--force", str(workspace))
        )
        if removed.returncode != 0 and workspace.exists():
            raise RuntimeError(f"git worktree remove failed: {_bounded(removed.stderr)}")