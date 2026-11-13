"""Seatbelt 后端:macOS sandbox-exec profile 的唯一渲染实现(设计 §5.3/§4.2;S1/S2;D4)。

移植自旧 agent_runtime/tools/builtins/shell.py(只读参照,不 import):
- _build_command_sandbox_profile(1806-1932)→ render_profile 骨架;
- _seatbelt_permission_rules(1155-1200)→ _permission_regions 不相交子树;
- _escape_seatbelt_string(2214)→ _escape_seatbelt_string;
- _trusted_toolchain_roots(1935)/_linked_git_metadata_roots(2160)→ 只读工具链区
  (uv 管理 CPython 与 git worktree 元数据,使 bash/python/git 在沙箱内可用)。

刻意差异(report 留档):
- 地板段由 protocol/floor.py FLOOR_ENTRIES 循环生成(D4 单一来源)——旧
  .venv/node_modules/.agents/.codex/.aws 第二份清单不移植;地板 deny 全域生效
  (与 floor_reason 的任意路径成分匹配一致,不排除 scratch);
- private_paths 双形态(词法+realpath)subpath deny 取代旧 command_state_directory()
  deny(host_private_paths 单一来源;/var ↔ /private/var 内核解析形态都覆盖);
- 旧 sources 祖先 file-write-unlink 保护不移植(ResolvedPolicy 无 sources;
  配置命名空间由地板 .praxis read+write deny 覆盖,file-write* 含 unlink);
- 旧 ValueError 一律改 SandboxRefusal(protocol 层有界异常,管道转 denied)。
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Literal

from praxis.protocol.floor import FLOOR_ENTRIES
from praxis.protocol.policy import ResolvedPolicy
from praxis.protocol.sandbox import SandboxRefusal, SandboxSpec

_SANDBOX_EXEC = "/usr/bin/sandbox-exec"


def _escape_seatbelt_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def render_profile(
    resolved: ResolvedPolicy,
    *,
    workspace: Path,
    scratch: Path,
    allow_network: bool,
    allow_write: bool,
    private_paths: tuple[str, ...],
) -> str:
    """把解析后的策略渲染为 seatbelt profile 原文(唯一渲染实现)。

    内核按解析后路径裁决:workspace/scratch 一律 realpath 渲染;policy 区域
    由 _permission_regions 不相交子树生成;地板段消费 FLOOR_ENTRIES(D4);
    网络段 iff allow_network(D9 裁决在 compile 层,本函数只执行布尔)。
    """
    workspace_real = _escape_seatbelt_string(str(os.path.realpath(workspace)))
    temporary = _escape_seatbelt_string(str(os.path.realpath(scratch)))
    trusted_roots = {path.resolve() for path in _trusted_toolchain_roots(workspace)}
    trusted_roots.update(_linked_git_metadata_roots(workspace))
    trusted_toolchains = tuple(_escape_seatbelt_string(str(path)) for path in sorted(trusted_roots, key=str))
    trusted_toolchain_reads = "".join(f'  (subpath "{path}")\n' for path in trusted_toolchains)
    trusted_toolchain_ancestors = "".join(f'  (path-ancestors "{path}")\n' for path in trusted_toolchains)
    policy_reads, policy_rules = _permission_regions(resolved, temporary_root=scratch, allow_write=allow_write)
    network_policy = (
        # DNS is the only approved AF_UNIX endpoint; local service and Docker
        # sockets remain denied even when IP networking is approved.
        '(allow network-outbound (remote ip "*:*"))\n'
        "(allow system-socket\n"
        "  (require-all\n"
        "    (socket-domain AF_SYSTEM)\n"
        "    (socket-protocol 2)))\n"
        "(allow system-socket (socket-domain AF_UNIX))\n"
        "(allow network-outbound\n"
        "  (remote unix-socket\n"
        '    (literal "/private/var/run/mDNSResponder"))\n'
        "  (remote unix-socket\n"
        '    (literal "/var/run/mDNSResponder")))\n'
        "(allow mach-lookup\n"
        '  (global-name "com.apple.bsd.dirhelper")\n'
        '  (global-name "com.apple.dnssd.service")\n'
        '  (global-name "com.apple.system.opendirectoryd.membership")\n'
        '  (global-name "com.apple.networkd")\n'
        '  (global-name "com.apple.ocspd")\n'
        '  (global-name "com.apple.trustd.agent")\n'
        '  (global-name "com.apple.SystemConfiguration.DNSConfiguration")\n'
        '  (global-name "com.apple.SystemConfiguration.configd"))\n'
        '(allow sysctl-read (sysctl-name-regex #"^net.routetable"))'
        if allow_network
        else ""
    )
    return (
        "(version 1)\n"
        "(deny default)\n"
        "(allow process-exec)\n"
        "(allow process-fork)\n"
        "(allow signal (target same-sandbox))\n"
        "(allow process-info* (target same-sandbox))\n"
        "(allow sysctl-read)\n"
        "(allow mach-lookup\n"
        '  (global-name "com.apple.system.opendirectoryd.libinfo"))\n'
        "(allow file-read* file-test-existence\n"
        f"{policy_reads}"
        f'  (subpath "{temporary}")\n'
        f"{trusted_toolchain_reads}"
        '  (literal "/")\n'
        '  (subpath "/bin")\n'
        '  (subpath "/sbin")\n'
        '  (subpath "/usr/bin")\n'
        '  (subpath "/usr/sbin")\n'
        '  (subpath "/usr/lib")\n'
        '  (subpath "/usr/libexec")\n'
        '  (subpath "/usr/share")\n'
        '  (subpath "/System/Library")\n'
        '  (subpath "/Library/Apple")\n'
        '  (subpath "/Library/Developer/CommandLineTools")\n'
        '  (subpath "/opt/homebrew")\n'
        '  (subpath "/usr/local")\n'
        '  (literal "/private/var/select/sh")\n'
        '  (literal "/private/etc/hosts")\n'
        '  (literal "/private/etc/protocols")\n'
        '  (literal "/private/etc/resolv.conf")\n'
        '  (literal "/private/etc/services")\n'
        '  (literal "/private/etc/ssl/cert.pem")\n'
        '  (literal "/private/etc/ssl/openssl.cnf")\n'
        '  (literal "/Library/Preferences/com.apple.networkd.plist")\n'
        '  (subpath "/private/var/db/timezone/zoneinfo")\n'
        '  (literal "/dev/null")\n'
        '  (literal "/dev/random")\n'
        '  (literal "/dev/urandom"))\n'
        "(allow file-read-metadata file-test-existence\n"
        f'  (path-ancestors "{workspace_real}")\n'
        f'  (path-ancestors "{temporary}")\n'
        f"{trusted_toolchain_ancestors}"
        '  (path-ancestors "/Library/Apple")\n'
        '  (path-ancestors "/Library/Developer/CommandLineTools")\n'
        '  (path-ancestors "/Library/Preferences/com.apple.networkd.plist")\n'
        '  (path-ancestors "/etc/hosts")\n'
        '  (path-ancestors "/opt/homebrew")\n'
        '  (path-ancestors "/usr/local")\n'
        '  (path-ancestors "/var/run/mDNSResponder"))\n'
        "(allow file-write*\n"
        f'  (subpath "{temporary}")\n'
        '  (literal "/dev/null"))\n'
        f"{policy_rules}"
        f"{_floor_entry_rules()}"
        f"{network_policy}\n"
        f"{_private_path_rules(private_paths)}"
    )


def compile_seatbelt(
    resolved: ResolvedPolicy,
    *,
    workspace: Path,
    scratch: Path,
    allow_network: bool,
    allow_write: bool,
    private_paths: tuple[str, ...],
) -> SandboxSpec:
    """渲染 profile 并产出 seatbelt 执行形态(argv_prefix + 审计原文)。"""
    profile = render_profile(
        resolved,
        workspace=workspace,
        scratch=scratch,
        allow_network=allow_network,
        allow_write=allow_write,
        private_paths=private_paths,
    )
    return SandboxSpec(
        backend="seatbelt",
        argv_prefix=(_SANDBOX_EXEC, "-p", profile),
        scratch_dir=str(scratch),
        profile_text=profile,
    )


# ---------------------------------------------------------------------------
# policy 区域:不相交子树(移植旧 _seatbelt_permission_rules)
# ---------------------------------------------------------------------------


def _permission_regions(
    resolved: ResolvedPolicy, *, temporary_root: Path, allow_write: bool,
) -> tuple[str, str]:
    """Render disjoint subtree regions so a parent deny cannot shadow child grants."""
    rules = resolved.filesystem
    broad_root_read = any(rule.path == "/" and rule.access != "deny" for rule in rules)
    if broad_root_read and allow_write:
        raise SandboxRefusal(
            "A broad host-root read grant cannot safely combine with subprocess writes: "
            "Seatbelt resolves host symlink aliases. Use path tools or scoped filesystem grants for edits."
        )
    temporary = _escape_seatbelt_string(str(os.path.realpath(temporary_root)))
    reads: list[str] = []
    policies: list[str] = []
    for rule in rules:
        path = Path(rule.path)
        try:
            resolved_path = path.resolve()
        except (OSError, RuntimeError, ValueError) as error:
            raise SandboxRefusal(f"cannot resolve sandbox policy boundary: {rule.path}") from error
        if resolved_path != path:
            raise SandboxRefusal(f"Seatbelt cannot represent a symlink policy boundary: {rule.path}")
        escaped = _escape_seatbelt_string(rule.path)
        children = [child for child in rules if child.path != rule.path and Path(child.path).is_relative_to(path)]
        exclusions = "".join(
            f'    (require-not (subpath "{_escape_seatbelt_string(child.path)}"))\n' for child in children
        )
        region = f'(require-all (subpath "{escaped}")\n{exclusions}'
        region += f'    (require-not (subpath "{temporary}")))'
        if rule.access != "deny":
            reads.append(f"  {region}\n")
            if path != Path("/"):
                policies.append(f'(allow file-read-metadata file-test-existence (path-ancestors "{escaped}"))\n')
        elif path != Path("/"):
            policies.append(f"(deny file-read* {region})\n")
        if allow_write and rule.access == "write":
            policies.append(f"(allow file-write* {region})\n")
        elif path != Path("/"):
            policies.append(f"(deny file-write* {region})\n")
    return "".join(reads), "".join(policies)


# ---------------------------------------------------------------------------
# 地板段:FLOOR_ENTRIES 单一来源(D4)+ private_paths 双形态
# ---------------------------------------------------------------------------


def _floor_regex(name: str) -> str:
    """目录名 → 全路径 case-insensitive regex(移植旧 #"/[.][gG][iI][tT]($|/)" 风格)。"""
    parts: list[str] = []
    for char in name:
        if char.isalpha():
            parts.append(f"[{char.lower()}{char.upper()}]")
        elif char == ".":
            parts.append("[.]")
        else:
            parts.append(re.escape(char))
    return "/" + "".join(parts) + "($|/)"


def _floor_entry_rules() -> str:
    """地板段由 FLOOR_ENTRIES 循环生成——不得有第二份地板清单(D4)。

    access=read(.git)→ 全域 write-deny;access=deny(.praxis/.agent_memory)
    → 全域 read+write deny。与 floor_reason 的任意路径成分匹配一致,不排除
    scratch(地板 bypass 免疫,沙箱内同样绝对)。注释行供 profile 审计。
    """
    lines: list[str] = []
    for entry in FLOOR_ENTRIES:
        regex = _floor_regex(entry.name)
        lines.append(f"; floor entry: {entry.name} (access={entry.access})\n")
        if entry.access == "read":
            lines.append(f'(deny file-write* (regex #"{regex}"))\n')
        else:
            lines.append(f'(deny file-read* file-write* (regex #"{regex}"))\n')
    return "".join(lines)


def _private_path_rules(private_paths: tuple[str, ...]) -> str:
    """宿主私有目录 subpath deny:词法与 realpath 双形态都渲染(ruling 3)。

    macOS 的 /var/folders ↔ /private/var/folders:内核按解析后路径裁决,
    只渲染词法形态会漏掉解析形态。
    """
    lines: list[str] = []
    for private in private_paths:
        try:
            lexical = os.path.abspath(os.path.expanduser(private))
            forms = dict.fromkeys((lexical, os.path.realpath(lexical)))
        except (OSError, ValueError) as error:
            raise SandboxRefusal(f"cannot normalize sandbox private path: {private}") from error
        for form in forms:
            lines.append(f'(deny file-read* file-write* (subpath "{_escape_seatbelt_string(form)}"))\n')
    return "".join(lines)


# ---------------------------------------------------------------------------
# 只读工具链区(移植旧 _trusted_toolchain_roots / _linked_git_metadata_roots)
# ---------------------------------------------------------------------------


def _trusted_toolchain_roots(workspace_root: Path) -> tuple[Path, ...]:
    """Resolve recognized read-only toolchains used by a workspace venv.

    A uv-created virtual environment commonly links ``.venv/bin/python`` to
    the managed CPython installation outside the workspace. Seatbelt resolves
    that symlink before applying path rules, so the interpreter must be
    readable explicitly. Only the one referenced CPython distribution is
    admitted, and only when it lives below a standard uv-managed Python root.
    """
    workspace = Path(workspace_root).resolve()
    uv_python_roots = tuple(
        path.resolve()
        for path in (
            Path.home() / "Library" / "Application Support" / "uv" / "python",
            Path.home() / ".local" / "share" / "uv" / "python",
        )
        if path.is_dir()
    )
    trusted: set[Path] = set()
    for name in ("python", "python3"):
        executable = workspace / ".venv" / "bin" / name
        if not executable.exists():
            continue
        resolved = executable.resolve()
        for root in uv_python_roots:
            if not resolved.is_relative_to(root):
                continue
            relative = resolved.relative_to(root)
            if relative.parts:
                trusted.add(root / relative.parts[0])
            break
    return tuple(sorted(trusted, key=str))


def _linked_git_metadata_roots(workspace_root: Path) -> tuple[Path, ...]:
    """Read registered worktree metadata without exposing the source checkout.

    Validate Git's bidirectional administrative pointers before admitting the
    common Git directory. An arbitrary workspace .git pointer is not a grant.
    These roots enter the same read-only OS boundary as runtime toolchains.
    """
    pointer = Path(workspace_root).resolve() / ".git"
    if not pointer.is_file() or pointer.is_symlink():
        return ()
    try:
        targets = _git_metadata_pointer_targets(pointer, kind="gitdir")
        if not targets:
            return ()
        administrative = targets[-1]
        common_targets = _git_metadata_pointer_targets(administrative / "commondir", kind="commondir")
        backlink_targets = _git_metadata_pointer_targets(administrative / "gitdir", kind="commondir")
        if not common_targets or not backlink_targets:
            return ()
        common = common_targets[-1]
        if (administrative.parent != common / "worktrees" or backlink_targets[-1] != pointer
                or not (common / "HEAD").is_file() or not (common / "objects").is_dir()):
            return ()
        return (common,)
    except (OSError, UnicodeError, RuntimeError):
        return ()


def _git_metadata_pointer_targets(path: Path, *, kind: Literal["gitdir", "commondir"]) -> tuple[Path, ...]:
    with path.open(encoding="utf-8") as stream:
        content = stream.read(4097)
    if len(content) > 4096:
        raise OSError("Git metadata pointer is unexpectedly large")
    first_line = content.splitlines()[0].strip() if content else ""
    if kind == "gitdir":
        marker = "gitdir:"
        if not first_line.casefold().startswith(marker):
            return ()
        value = first_line[len(marker):].strip()
    else:
        value = first_line
    if not value:
        return ()
    target = Path(value)
    if not target.is_absolute():
        target = path.parent / target
    lexical_target = Path(os.path.abspath(target))
    return tuple(dict.fromkeys((lexical_target, lexical_target.resolve())))