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

``permissions=None``(单测直跑、无快照)返回空来源元组 ⇒ 不做逐条目过滤,
阶段 1/2 既有工具行为不变。
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

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
    permissions: PermissionContext | None, workspace: Path
) -> tuple[ResolvedPolicy, ...]:
    """把权限快照解析为路径裁决来源(policy + ceilings + session/once 覆盖层)。"""
    return tuple(
        resolve_policy(policy, workspace) for policy in permission_declarations(permissions)
    )


def is_read_denied(sources: Iterable[ResolvedPolicy], path: Path) -> bool:
    """任一来源拒绝该路径(读访问)→ True。空来源 = 不拒绝(无快照直跑)。"""
    return any(not path_access(source, path, write=False) for source in sources)
