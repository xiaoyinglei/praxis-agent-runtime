"""Praxis 子工作区适配器:git worktree 快照实现(阶段4 T7;设计 §5.13)。

``GitWorktreeProvisioner`` 是 protocol.workspaces.ChildWorkspaceProvisioner 的
唯一生产实现(composition 绑定)。控制器裁决:

- 落点 ``host_worktrees_dir()/<sha256(resolve(parent))[:16]>/<child>``
  (R-T8-1 重裁 T7 落点)——库外、父工作区外、宿主私有(根 0o700)、
  **非地板私有集**:子 Session 可经生产工具写自身工作区;父工具/沙箱
  不可达(host_private_paths 逐字不变,S-3);
- ``git -C parent worktree add --detach <path> HEAD``——detach 快照,父工作区
  未提交字节不入快照;integrate 走 diff/patch,不引入 branch merge;
- diff bundle = ``git diff HEAD`` + ``git status --porcelain -z`` 的 untracked
  清单,自有格式(格式常量唯一在本模块);``.praxis``/``.git`` 在 diff 与
  apply 路径集都排除(子 always 策略文件绝不渗回父);
- apply 逐文件 preimage sha256 复检:任一不一致 → conflicts 且整组不写;
  ``git apply --check`` 后 ``git apply``,失败整组拒(``patch_failed`` parity);
- R-T7-3(T8 收口):untracked 写入前对目标路径**祖先链**做 symlink
  confinement——悬空 symlink 祖先计 conflict;解析后越出 parent_root 的
  symlink 祖先计 conflict(否则 ENOENT→None 假匹配会半写/写出父仓);
- T8 inspect:路径集 + HEAD/子两侧 sha256 表 + ≤200_000 字节 diff 预览
  (截断旗标 parity 旧 diff_limit);diff/inspect 支持 paths 子集;
- 非 git 工作区 prepare 抛 ``ChildWorkspaceRequiresGit``(Q5)。

子进程纪律(B33):固定 argv 数组、不经 shell;env 基底由 composition 注入
``tools.builtins.child_env``(adapters 不 import tools),本模块只追加最小
GIT_* 键。宿主 ``os.environ`` 从不读取。
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import struct
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from praxis.protocol.floor import host_worktrees_dir
from praxis.protocol.sandbox import EnvBuilder
from praxis.protocol.workspaces import (
    DIFF_PREVIEW_LIMIT_BYTES,
    ApplyResult,
    ChildWorkspaceRequiresGit,
    WorkspaceInspection,
)

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


def _validated_subset(paths: Sequence[str]) -> tuple[str, ...]:
    """integrate paths 子集校验:相对、无越界成分、不含排除路径。"""
    subset: list[str] = []
    for raw in paths:
        rel = _repo_relative(raw)
        if _is_excluded(raw):
            raise ValueError(f"workspace subset must not contain excluded path: {raw!r}")
        subset.append(rel.as_posix())
    return tuple(subset)


def _in_subset(raw: str, subset: Sequence[str]) -> bool:
    return any(raw == entry or raw.startswith(entry + "/") for entry in subset)


def _ancestor_symlink_escape(parent: Path, rel: PurePosixPath) -> bool:
    """R-T7-3:目标路径祖先链的 symlink confinement(承重)。

    任一祖先是 symlink 时:悬空/环链(resolve 失败)→ True;解析后越出
    parent_root → True。解析后仍在父仓内的 symlink 祖先放行(controller
    裁决口径)。不查目标本身——``_file_digest`` 的 lstat 已把非普通文件
    目标计为冲突。
    """
    current = parent
    for part in rel.parts[:-1]:
        current = current / part
        if not current.is_symlink():
            continue
        try:
            resolved = current.resolve(strict=True)
        except OSError:
            return True  # 悬空 symlink/环链:ENOENT 假匹配的病灶
        if not resolved.is_relative_to(parent):
            return True
    return False


def _symlink_digest(path: Path) -> str | None:
    """tracked symlink 的父侧 preimage 摘要(R-T8-2)。

    HEAD mode 120000 的 blob 内容 = 链接目标字节,``_head_preimages`` 据此
    返回 sha256(目标);父侧复检必须同口径(readlink sha),否则子改/删
    tracked symlink 在父未漂移时永久假冲突。缺失 = None(父删了 → 漂移
    conflict);已存在但非 symlink = ""(父类型漂移 → conflict)。
    untracked 写目标仍走 ``_file_digest``(非普通文件一律拒写,语义不变)。
    """
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return None
    except OSError:
        return ""
    if not stat.S_ISLNK(mode):
        return ""
    try:
        return _sha256(os.fsencode(os.readlink(path)))
    except OSError:
        return ""


def _worktree_digest(path: Path) -> str | None:
    """子工作区侧 postimage 摘要:普通文件 sha256;symlink = 链接目标字节
    sha256(与 bundle 的 120000 口径一致);缺失 = None;其余 = ""。"""
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return None
    except OSError:
        return ""
    if stat.S_ISLNK(mode):
        try:
            return _sha256(os.fsencode(os.readlink(path)))
        except OSError:
            return ""
    if not stat.S_ISREG(mode):
        return ""
    try:
        return _sha256(path.read_bytes())
    except OSError:
        return ""


def _diff_preview(tracked_diff: bytes) -> tuple[str, bool]:
    """≤200_000 字节预览(UTF-8 边界安全;旧 diff_limit 分页口径 parity)。"""
    end = min(len(tracked_diff), DIFF_PREVIEW_LIMIT_BYTES)
    while end < len(tracked_diff) and tracked_diff[end] & 0xC0 == 0x80:
        end += 1
    return tracked_diff[:end].decode("utf-8", errors="backslashreplace"), end < len(tracked_diff)


def _validated_child_id(child_thread_id: str) -> str:
    if not child_thread_id or child_thread_id in {".", ".."} or "/" in child_thread_id or "\x00" in child_thread_id:
        raise ValueError(f"invalid child_thread_id for workspace directory: {child_thread_id!r}")
    return child_thread_id


def _file_digest(path: Path) -> str | None:
    """preimage 摘要;区分"真缺失"与"已存在但非普通文件"。

    None 只在 lstat ENOENT 时返回;目录/悬空 symlink/fifo/无权限等已存在但
    不可作为普通文件写的目标返回空串——空串与任何 sha256/None 都不相等,
    因此 apply 会把它计入 conflicts(整体零写),不会先 apply tracked patch
    再在 untracked 写入时炸出半写。

    口径分工(R-T8-2):本函数服务 regular 路径与 untracked 写目标;
    HEAD 为 symlink(120000)的 tracked 路径父侧复检走 ``_symlink_digest``
    (readlink sha,与 _head_preimages 的 blob 内容口径配对)。
    """
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError:
        return None
    except OSError:
        return ""
    if not stat.S_ISREG(mode):
        return ""
    try:
        return _sha256(path.read_bytes())
    except OSError:
        return ""


_DIFF_ESCAPES = {
    ord("\\"): 0x5C,
    ord('"'): 0x22,
    ord("t"): 0x09,
    ord("n"): 0x0A,
    ord("r"): 0x0D,
    ord("f"): 0x0C,
    ord("v"): 0x0B,
    ord("b"): 0x08,
    ord("a"): 0x07,
}


def _unquote_diff_path(token: bytes) -> bytes:
    """git C 引号路径解码(``"a/\\303\\251.txt"`` → 原始字节)。"""
    if not (len(token) >= 2 and token.startswith(b'"') and token.endswith(b'"')):
        return token
    body = token[1:-1]
    out = bytearray()
    index = 0
    while index < len(body):
        byte = body[index]
        index += 1
        if byte != 0x5C:
            out.append(byte)
            continue
        if index >= len(body):
            raise ValueError("malformed quoted path in workspace bundle")
        escaped = body[index]
        index += 1
        if escaped in _DIFF_ESCAPES:
            out.append(_DIFF_ESCAPES[escaped])
            continue
        if 0x30 <= escaped <= 0x37:
            digits = bytes((escaped,)) + body[index : index + 2]
            if len(digits) != 3 or any(digit < 0x30 or digit > 0x37 for digit in digits):
                raise ValueError("malformed octal escape in workspace bundle path")
            out.append(int(digits, 8))
            index += 2
            continue
        raise ValueError("unsupported escape in workspace bundle path")
    return bytes(out)


def _decode_repo_path(token: bytes) -> str | None:
    """``---``/``+++`` 或 ``diff --git`` 的单个路径 token → 仓内相对路径。"""
    if token == b"/dev/null":
        return None
    decoded = _unquote_diff_path(token)
    for prefix in (b"a/", b"b/"):
        if decoded.startswith(prefix):
            return os.fsdecode(decoded[len(prefix) :])
    raise ValueError(f"workspace bundle diff path lacks a/ or b/ prefix: {os.fsdecode(decoded)!r}")


def _header_token(raw: bytes) -> bytes:
    """``---``/``+++`` 行路径:带空格时 git 在行尾补 tab 作分隔。"""
    return raw[:-1] if raw.endswith(b"\t") else raw


def _split_git_diff_header(remainder: bytes) -> tuple[bytes, bytes]:
    """``diff --git a/old b/new`` 的双路径拆分(引号名 + 空格名启发式)。

    二进制 patch 没有 ``---``/``+++`` 行,名字只在这个头里;引号名按 C 转义
    扫描,未引号含空格时用「同一路径两前缀」启发式拆分(git 自身同形态)。
    """
    tokens: list[bytes] = []
    index = 0
    length = len(remainder)
    while index < length:
        if remainder[index] == 0x22:
            end = index + 1
            while end < length:
                if remainder[end] == 0x5C:
                    end += 2
                    continue
                if remainder[end] == 0x22:
                    break
                end += 1
            if end >= length:
                raise ValueError("unterminated quoted path in workspace bundle header")
            tokens.append(remainder[index : end + 1])
            index = end + 1
            if index < length and remainder[index] == 0x20:
                index += 1
        else:
            end = remainder.find(b" ", index)
            if end == -1:
                end = length
            tokens.append(remainder[index:end])
            index = end + 1
    if len(tokens) == 2:
        return tokens[0], tokens[1]
    marker = b" b/"
    start = 0
    while True:
        split = remainder.find(marker, start)
        if split == -1:
            break
        first = remainder[:split]
        second = remainder[split + 1 :]
        if first.startswith(b"a/") and second.startswith(b"b/") and first[2:] == second[2:]:
            return first, second
        start = split + 1
    raise ValueError("ambiguous diff --git header in workspace bundle")


def _diff_declared_paths(tracked_diff: bytes) -> tuple[str, ...]:
    """解析 patch 声明的**两端口径**路径(旧路径删除向量承重)。

    numstat 只报 ``+++`` 新路径;``---``/``+++`` 名字不同的段会被 git apply
    同时删除旧路径。这里对 old≠new 的段直接整组拒绝(本模块 diff 固定
    ``--no-renames``,只产 old==new / /dev/null 形态);旧路径也进入排除校验。
    """
    paths: list[str] = []
    in_header = False
    git_header: bytes | None = None
    old_token: bytes | None = None
    for line in tracked_diff.split(b"\n"):
        if line.startswith(b"diff --git "):
            in_header = True
            git_header = line[len(b"diff --git ") :]
            old_token = None
            continue
        if not in_header:
            continue
        if line.startswith(b"GIT binary patch"):
            if git_header is None:
                raise ValueError("workspace bundle binary section lacks diff --git header")
            old_token, new_token = _split_git_diff_header(git_header)
            old_path = _decode_repo_path(old_token)
            new_path = _decode_repo_path(new_token)
            if old_path is not None and new_path is not None and old_path != new_path:
                raise ValueError("workspace bundle must not contain differing patch paths")
            paths.extend(path for path in (old_path, new_path) if path is not None)
            in_header = False
            continue
        if line.startswith(b"@@"):
            in_header = False
            old_token = None
            continue
        if line.startswith(b"--- "):
            old_token = _header_token(line[4:])
            continue
        if line.startswith(b"+++ "):
            new_token = _header_token(line[4:])
            if old_token is not None:
                old_path = _decode_repo_path(old_token)
                new_path = _decode_repo_path(new_token)
                if old_path is not None and new_path is not None and old_path != new_path:
                    raise ValueError("workspace bundle must not contain differing patch paths")
                paths.extend(path for path in (old_path, new_path) if path is not None)
            old_token = None
    return tuple(paths)


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

    # -- tracked patch 载荷对账 ----------------------------------------------

    def _tracked_payload_paths(self, parent: Path, tracked_diff: bytes) -> tuple[str, ...]:
        """解析 tracked diff 载荷的真实路径(``git apply --numstat -z``)。

        header 的 tracked_paths 是自报清单;真正写父仓的是 diff 载荷。apply 前
        必须逐条校验载荷路径(排除 .praxis/.git、不越界)并与 header 对账——
        否则构造 header 合法、载荷改 ``.praxis`` 的 bundle 会绕过排除。
        """
        if not tracked_diff:
            return ()
        numstat = self._git(parent, parent, ("apply", "--numstat", "-z"), stdin=tracked_diff)
        if numstat.returncode != 0:
            raise RuntimeError(
                f"patch_failed: git apply --numstat rejected the bundle: {_bounded(numstat.stderr)}"
            )
        summary = self._git(parent, parent, ("apply", "--summary"), stdin=tracked_diff)
        if summary.returncode != 0:
            raise RuntimeError(
                f"patch_failed: git apply --summary rejected the bundle: {_bounded(summary.stderr)}"
            )
        for line in summary.stdout.decode("utf-8", errors="replace").splitlines():
            # 本模块生成 diff 时固定 --no-renames;rename/copy 载荷的旧路径
            # 不在 numstat 清单里,无法对账——整组拒绝(不引入 branch/rename 面)。
            if line.strip().startswith(("rename ", "copy ")):
                raise ValueError("workspace bundle must not contain rename/copy payloads")
        paths: list[str] = []
        for record in numstat.stdout.split(b"\0"):
            if not record:
                continue
            fields = record.split(b"\t", 2)
            if len(fields) != 3:
                raise RuntimeError(f"patch_failed: malformed numstat record: {record!r}")
            paths.append(os.fsdecode(fields[2]))
        return tuple(paths)

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
        root = Path(os.path.abspath(host_worktrees_dir()))
        # 宿主私有根 0o700(command_jobs_dir 同惯例);chmod 幂等收口,
        # 防外部预创建的宽松模式目录。
        root.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(root, 0o700)
        base = root / digest
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

    def diff(self, ws: Path, *, paths: Sequence[str] | None = None) -> bytes:
        """子工作区相对 HEAD 的 bundle(tracked diff + untracked;排除 .praxis/.git)。

        ``paths`` 给定 = 子集 bundle:tracked diff 走 literal pathspec,
        untracked 按精确文件/目录前缀过滤(T8 integrate paths 参数)。
        """
        workspace = Path(ws)
        subset = _validated_subset(paths) if paths is not None else None
        status = self._git(
            workspace, workspace, ("status", "--porcelain", "-z", "--untracked-files=all")
        )
        if status.returncode != 0:
            raise RuntimeError(f"git status failed: {_bounded(status.stderr)}")
        untracked: list[_UntrackedFile] = []
        for raw_path in _untracked_paths(status.stdout):
            if _is_excluded(raw_path):
                continue
            if subset is not None and not _in_subset(raw_path, subset):
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

        pathspecs = (".",) if subset is None else tuple(f":(literal){entry}" for entry in subset)
        diff_args = (
            "diff",
            "--no-ext-diff",
            "--binary",
            "--no-renames",
            "--no-color",
            "--src-prefix=a/",
            "--dst-prefix=b/",
            "HEAD",
            "--",
            *pathspecs,
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
                "--no-color",
                "--name-only",
                "-z",
                "HEAD",
                "--",
                *pathspecs,
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

    # -- inspect -------------------------------------------------------------

    def _head_preimages(self, workspace: Path, paths: Sequence[str]) -> dict[str, str | None]:
        """HEAD(快照基线)侧逐路径 sha256;基线不存在 = None。

        ls-tree -r -z 取 blob oid(路径经 -z 原始字节,无引号歧义),
        cat-file --batch 按 oid 取内容(oid 是 hex,stdin 无路径注入面)。
        """
        preimages: dict[str, str | None] = {path: None for path in paths}
        if not paths:
            return preimages
        listing = self._git(workspace, workspace, ("ls-tree", "-r", "-z", "HEAD"))
        if listing.returncode != 0:
            raise RuntimeError(f"git ls-tree failed: {_bounded(listing.stderr)}")
        wanted = set(paths)
        oids: dict[str, str] = {}
        for record in listing.stdout.split(b"\0"):
            if not record:
                continue
            meta, _, raw_path = record.partition(b"\t")
            fields = meta.split(b" ")
            if len(fields) != 3:
                raise RuntimeError(f"malformed git ls-tree record: {record!r}")
            path = os.fsdecode(raw_path)
            if path in wanted:
                oids[path] = fields[2].decode("ascii")
        if not oids:
            return preimages
        ordered = sorted(oids.items())
        stdin = "".join(f"{oid}\n" for _, oid in ordered).encode("ascii")
        batch = self._git(workspace, workspace, ("cat-file", "--batch"), stdin=stdin)
        if batch.returncode != 0:
            raise RuntimeError(f"git cat-file --batch failed: {_bounded(batch.stderr)}")
        out = batch.stdout
        offset = 0
        for path, oid in ordered:
            newline = out.find(b"\n", offset)
            if newline == -1:
                raise RuntimeError("truncated git cat-file batch output")
            header = out[offset:newline].decode("ascii").split()
            offset = newline + 1
            if len(header) != 3 or header[0] != oid or header[1] != "blob":
                raise RuntimeError(f"unexpected git cat-file header for {path!r}: {header!r}")
            size = int(header[2])
            data = out[offset : offset + size]
            if len(data) != size:
                raise RuntimeError(f"truncated git cat-file blob payload for {path!r}")
            offset += size + 1  # blob 后的分隔 LF
            preimages[path] = _sha256(data)
        return preimages

    def _head_symlink_paths(self, parent: Path, paths: Sequence[str]) -> frozenset[str]:
        """HEAD mode 120000 的 touched 路径集(R-T8-2 父侧口径判据)。

        以父仓 HEAD 为模式事实源:preimages 期望值来自子快照基线,父 HEAD
        未漂移时两者同源;父 HEAD 漂移致模式/内容变化时 preimage 复检本就
        conflict(fail-closed),``git apply --check`` 仍是类型错配的最终兜底。
        """
        if not paths:
            return frozenset()
        listing = self._git(parent, parent, ("ls-tree", "-r", "-z", "HEAD"))
        if listing.returncode != 0:
            raise RuntimeError(f"git ls-tree failed: {_bounded(listing.stderr)}")
        wanted = set(paths)
        symlinks: set[str] = set()
        for record in listing.stdout.split(b"\0"):
            if not record:
                continue
            meta, _, raw_path = record.partition(b"\t")
            fields = meta.split(b" ")
            if len(fields) != 3:
                raise RuntimeError(f"malformed git ls-tree record: {record!r}")
            if fields[0] == b"120000":
                path = os.fsdecode(raw_path)
                if path in wanted:
                    symlinks.add(path)
        return frozenset(symlinks)

    def inspect(self, ws: Path, *, paths: Sequence[str] | None = None) -> WorkspaceInspection:
        """T8 读面:变更路径集 + HEAD/子两侧 sha256 表 + 有界 diff 预览。"""
        workspace = Path(ws)
        tracked_diff, tracked_paths, untracked = _decode_bundle(self.diff(workspace, paths=paths))
        touched = sorted(set(tracked_paths) | {file.path for file in untracked})
        preimages = self._head_preimages(workspace, touched)
        postimages = {
            path: _worktree_digest(workspace / Path(*PurePosixPath(path).parts))
            for path in touched
        }
        preview, truncated = _diff_preview(tracked_diff)
        return WorkspaceInspection(
            paths=tuple(touched),
            preimages=preimages,
            postimages=postimages,
            diff_preview=preview,
            diff_truncated=truncated,
            diff_bytes=len(tracked_diff),
        )

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
        # 安全面承重:header 自报之外,逐条校验 tracked diff 载荷路径并对账。
        # declared = ---/+++ 与二进制 diff --git 的两端口径(旧路径删除向量);
        # payload = numstat 的新路径清单,保留作交叉核对。
        declared_paths = _diff_declared_paths(tracked_diff)
        payload_paths = self._tracked_payload_paths(parent, tracked_diff)
        for raw_path in (*declared_paths, *payload_paths):
            _repo_relative(raw_path)
            if _is_excluded(raw_path):
                raise ValueError(
                    f"workspace bundle patch must not touch excluded path: {raw_path!r}"
                )
        undeclared = sorted((set(declared_paths) | set(payload_paths)) - set(tracked_paths))
        if undeclared:
            raise ValueError(
                "workspace bundle tracked diff paths do not match bundle header: "
                f"undeclared={undeclared}"
            )
        touched = sorted(set(bundle_paths))

        # R-T8-2:HEAD 为 symlink 的路径,父侧按 readlink sha 复检(与
        # _head_preimages 的 120000 blob 口径配对);其余路径仍 _file_digest
        # (untracked 写目标的非普通文件拒语义不变)。
        head_symlinks = self._head_symlink_paths(parent, touched)
        conflicts: list[str] = []
        for raw_path in touched:
            expected = preimages[raw_path] if raw_path in preimages else None
            rel = PurePosixPath(raw_path)
            target = parent / Path(*rel.parts)
            actual = _symlink_digest(target) if raw_path in head_symlinks else _file_digest(target)
            # R-T7-3:祖先链 symlink confinement 先于任何写入(悬空/越界
            # 祖先都计 conflict,整组零写——ENOENT→None 假匹配不再可达)。
            if actual != expected or _ancestor_symlink_escape(parent, rel):
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