"""目录/文件级读取工具的执行侧 deny 过滤(R-T11-2)。

背景(S1 家族的真实口子):``decide`` 只看 ``resolve_use`` 声明的 target。
目录型读取工具(search_text / list_files)的 target 是**被搜/被列目录**,因此
工作区内部被 ``PathRule(deny)`` 覆盖的子路径,仍可能被一次目录级遍历把内容
原文读出来。本模块把"已由 decide 授权的目录 target"细化为"逐条目检查",
确保 deny 规则在遍历内部同样生效。

**这不是第二套决策入口**:``decide``(core/permission.py)仍是唯一裁决函数,
审批/地板/沙箱等全部步骤不变。这里只复用 ``protocol.policy`` 的同一份规则
代数(``resolve_policy`` + ``path_access``,CI 规则 9 的唯一实现),对每个具体
路径做"是否被任一策略来源拒绝"的合取——拒绝即跳过,绝不反向授权。

规则来源与 ``decide`` 的 path 裁决保持一致:声明 policy + ceilings,再加
session/once 两个覆盖层(``protocol.policy.overlay_policy`` 构造,与
``core.permission._rule_layers`` 共用同一实现);逐一 ``resolve_policy`` 后
最严者胜——任一来源 deny 即 deny。注意 ``session``/``always`` 的规则本身已被
``apply_update`` 折进 ``permissions.policy``,覆盖层来源是额外的最严者检查;
``once`` 只存在于 ``once_rules``,覆盖层是它唯一进入目录过滤的通道。

覆盖层只收紧、不放宽(allow 被 ``/`` write 基线吸收),因此把它们纳入过滤
是安全方向;绝不把覆盖层合并进 base policy——合并会让更具体的 allow 反向
覆盖较宽的 deny。

``permissions=None``(单测直跑、无快照)返回空来源元组 ⇒ 不做策略过滤,
阶段 1/2 既有工具行为不变。**但地板与私有目录始终生效**(``floor_reason``
按路径判定,S-2):环境派生的私有根(PRAXIS_CONFIG_DIR / jobs 目录)落在
工作区内时,目录级遍历也必须跳过,而不是只按目录名剪枝。
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from praxis.protocol.floor import floor_reason
from praxis.protocol.permission import PermissionContext
from praxis.protocol.policy import (
    PermissionPolicy,
    ResolvedPolicy,
    overlay_policy,
    path_access,
    resolve_policy,
)


def permission_declarations(
    permissions: PermissionContext | None,
) -> tuple[PermissionPolicy, ...]:
    """权限快照 → 需逐一 resolve 的策略声明(policy + ceilings + 覆盖层)。

    search 工具把同一组声明序列化给 worker,worker 侧再 ``resolve_policy``,
    保证父子进程看到完全相同的来源集合。
    """
    if permissions is None:
        return ()
    declarations = [permissions.policy, *permissions.ceilings]
    for rules in (permissions.session_rules, permissions.once_rules):
        if rules:
            declarations.append(overlay_policy(rules))
    return tuple(declarations)


def permission_sources(
    permissions: PermissionContext | None,
    workspace: Path,
    *,
    private_paths: tuple[str, ...] = (),
) -> tuple[ResolvedPolicy, ...]:
    """把权限快照解析为路径裁决来源(policy + ceilings + session/once 覆盖层)。

    ``private_paths`` 由调用方传 ``host_private_paths()``:与 ``decide`` 的
    path 裁决同一来源集合,私有目录在工作区内时也能被逐条目过滤(S-2)。
    """
    return tuple(
        resolve_policy(policy, workspace, private_paths=private_paths)
        for policy in permission_declarations(permissions)
    )


def is_read_denied(
    sources: Iterable[ResolvedPolicy],
    path: Path,
    *,
    private_paths: tuple[str, ...] = (),
) -> bool:
    """该路径的读访问是否被拒:地板/私有目录按路径判 + 任一策略来源拒绝。

    地板先判(S-2):私有目录与 deny 地板条目不依赖策略声明——即使没有权限
    快照(单测直跑),它们也不得被目录级遍历读出。空来源 + 无地板命中 =
    不拒绝(无快照直跑路径)。
    """
    if floor_reason(path, write=False, private_paths=private_paths):
        return True
    return any(not path_access(source, path, write=False) for source in sources)
