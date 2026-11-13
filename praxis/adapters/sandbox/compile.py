"""沙箱编译:唯一 profile 构造入口(设计 §5.3/§4.2;S1 结构消亡;D7/D9)。

流程(controller ruling 2):
resolve_policy(policy + 逐一 ceilings,private_paths=host_private_paths())
→ validate_aliases(S9 preflight,拒绝即 SandboxRefusal)
→ allow_write = path_access(policy, workspace, write=True) 且全部 ceilings 通过
→ allow_network = command_network_decision == "allow"(D9,与 decide 同一函数)
  且全部 ceilings allow
→ probe 分派:seatbelt 编译;其他平台 SandboxRefusal(bwrap 延后,D7)。

use 参数是冻结契约(SandboxCompiler 结构签名,Task 7 接线);本阶段编译按
workspace 粒度,target 级细化留待后续任务。ceiling 只否决、永不授权:
写/网络授权取 policy 与全部 ceilings 的交集,fail-closed。
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path

from praxis.adapters.sandbox.base import probe_backend
from praxis.adapters.sandbox.seatbelt import compile_seatbelt
from praxis.protocol.floor import host_private_paths
from praxis.protocol.permission import PermissionContext
from praxis.protocol.policy import (
    ResolvedPolicy,
    command_network_decision,
    path_access,
    resolve_policy,
)
from praxis.protocol.sandbox import SandboxRefusal, SandboxSpec
from praxis.protocol.tools import ResolvedUse

logger = logging.getLogger(__name__)


def compile_sandbox(permissions: PermissionContext, use: ResolvedUse, workspace: Path) -> SandboxSpec:
    """唯一 profile 构造入口:policy→SandboxSpec,任何不可表示/不可强制即拒绝。

    签名与 core 注入的 SandboxCompiler 结构一致(Task 7)。拒绝一律
    SandboxRefusal(有界 reason,管道转 denied);scratch 仅在分派成功后创建,
    编译失败即清理,不泄漏宿主临时目录。
    """
    # 与 resolve_policy 同一规范化(expanduser+resolve):macOS 的 /var/folders 等
    # 词法形态含 symlink 成分,不规范化会让 path_access 的词法候选落空、
    # allow_write 误判为 False(workspace 写授权被影子化)。
    root = workspace.expanduser().resolve()
    private_paths = host_private_paths()
    resolved = resolve_policy(permissions.policy, root, private_paths=private_paths)
    ceilings = tuple(
        resolve_policy(ceiling, root, private_paths=private_paths) for ceiling in permissions.ceilings
    )
    allow_write = path_access(resolved, root, write=True) and all(
        path_access(ceiling, root, write=True) for ceiling in ceilings
    )
    for policy in (resolved, *ceilings):
        validate_aliases(policy, root, allow_write=allow_write)
    allow_network = command_network_decision(resolved) == "allow" and all(
        command_network_decision(ceiling) == "allow" for ceiling in ceilings
    )
    availability = probe_backend()
    if not availability.available or availability.backend != "seatbelt":
        raise SandboxRefusal("no sandbox backend on this platform")
    scratch = Path(tempfile.mkdtemp(prefix="praxis-scratch-"))
    try:
        return compile_seatbelt(
            resolved,
            workspace=root,
            scratch=scratch,
            allow_network=allow_network,
            allow_write=allow_write,
            private_paths=private_paths,
        )
    except Exception:
        shutil.rmtree(scratch, ignore_errors=True)
        raise


def finalize_sandbox(spec: SandboxSpec, workspace: Path) -> None:
    """清理编译产物:seatbelt 删 scratch(bwrap 的 commit_view 随 Task 6 延后)。

    幂等(重复调用 no-op),异常不抛——清理失败只记日志,绝不炸穿管道。
    """
    if spec.backend != "seatbelt":
        return
    try:
        shutil.rmtree(spec.scratch_dir)
    except FileNotFoundError:
        pass  # 幂等:已清理
    except Exception:
        logger.warning("sandbox scratch cleanup failed: %s", spec.scratch_dir, exc_info=True)


def validate_aliases(resolved: ResolvedPolicy, workspace: Path, *, allow_write: bool) -> None:
    """S9 preflight(移植旧 _validate_permission_aliases + 边界检查,ruling 5)。

    四类拒绝(fail-closed,一律 SandboxRefusal):
    1. symlink 策略边界:seatbelt 按解析后路径裁决,边界含 symlink 不可表示;
    2. 硬链接授权文件:st_nlink>1 的授权文件,inode 共享使词法限制不可强制;
    3. 授权根 symlink 逃逸:alias 无权限但 target 有权限 → 词法限制被擦除;
    4. 扫描失败(OSError/不可解析):无法验证即拒绝。

    根选择逻辑忠实移植旧实现:broad-root-read 时选 deny 根,否则选授权根
    (含被授权父覆盖的 deny 根);去掉 symlink/非目录根与嵌套根。

    workspace 是冻结契约参数(brief ACI);旧实现用它跳过外部根的
    _workspace_sandbox_preflight_issue——该重型扫描不在本任务移植范围
    (brief 四类检查),参数保留以对齐签名。
    """
    rules = resolved.filesystem
    for rule in rules:
        path = Path(rule.path)
        try:
            resolved_path = path.resolve()
        except (OSError, RuntimeError, ValueError) as error:
            raise SandboxRefusal(f"cannot resolve sandbox policy boundary: {rule.path}") from error
        if resolved_path != path:
            raise SandboxRefusal(f"seatbelt cannot represent a symlink policy boundary: {rule.path}")
    for rule in rules:
        if rule.access == "deny":
            continue
        path = Path(rule.path)
        try:
            if path.is_file() and path.stat().st_nlink > 1:
                raise SandboxRefusal(f"sandbox cannot safely expose a hardlinked file grant: {path}")
        except OSError as error:
            raise SandboxRefusal(f"cannot validate sandbox permission aliases: {path}") from error
    broad_root_read = any(rule.path == "/" and rule.access != "deny" for rule in rules)
    try:
        roots = {
            Path(rule.path) for rule in rules if Path(rule.path) != Path("/") and (
                (rule.access == "deny" if broad_root_read else rule.access != "deny") or (not broad_root_read and any(
                    parent.access != "deny" and Path(rule.path).is_relative_to(parent.path) for parent in rules
                ))
            )
        }
        roots = {root for root in roots if root.is_dir() and not root.is_symlink()}
        roots = {root for root in roots if not any(root != parent and root.is_relative_to(parent) for parent in roots)}
    except OSError as error:
        raise SandboxRefusal(f"cannot validate sandbox permission aliases: {error.filename}") from error

    def scan_error(error: OSError) -> None:
        raise SandboxRefusal(f"cannot validate sandbox permission aliases: {error.filename}") from error

    for root in sorted(roots):
        for directory, directories, files in os.walk(root, followlinks=False, onerror=scan_error):
            for name in (*directories, *files):
                alias = Path(directory) / name
                if not alias.is_symlink():
                    continue
                try:
                    target = alias.resolve()
                except (OSError, RuntimeError) as error:
                    raise SandboxRefusal("cannot resolve a sandbox permission alias") from error
                for write in (False, True) if allow_write else (False,):
                    if not path_access(resolved, alias, write=write) and path_access(resolved, target, write=write):
                        raise SandboxRefusal(
                            f"sandbox cannot enforce lexical permission restriction on symlink: {alias}"
                        )