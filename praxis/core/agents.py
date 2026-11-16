"""Praxis 子代理原语(设计 §5.13 四条已拍板语义;阶段4 T5;B2/B14 主战场)。

四条语义与本模块的对应:

1. **同库独立 thread**:子代理有自己的 thread_id/turn_id,items 写自己的
   thread,但库是父库(子工作区是独立目录时 Store 仍指向父库文件——
   composition.build_child_session 经 ``Store.db_path`` 绑定)。执行锁按子
   thread 独立持有(read_only 共享父工作区时锁文件同目录不同名;write 子
   工作区内独立锁文件族),sweep 覆盖全库。
2. **显式级联**:`cascade_for` 是级联的**唯一实现**(R-T5-2)——Session 终态
   路径(CANCELLED/FAILED)与 recovery sweep 路径(INTERRUPTED)都调它;
   级联是代码逻辑不是状态机语义(TRANSITIONS 零父子规则)。幂等:子已终态/
   已有未处理 cancel_request → 跳过;只到直接子,孙代逐跳自发收敛。
3. **resume 不连带子**:子 Session 是独立实例,父 resume(confirm=True) 只
   驱动本 turn;子有自己的 interrupted 标记需各自确认(结构成立,测试钉)。
4. **预算树结算**:grant = ``limits_merge_grant(requested, parent.remaining)``
   (B2:份额上限是算法,默认永不吸干父);任一有界维 grant ≤ 0 →
   ``ChildGrantError`` 拒 spawn。settle 是推导不是写:子根 turn 终态事实 →
   build_tree 折叠归还残额。interrupted/paused/running 子继续占用份额
   (R-T5-1:可被 resume 的子不能提前归还预算)。

spawn 序(brief 钉死):guard(父非终态/深度/总数/并发)→ build_tree(父) →
grant → provisioner.prepare(write;失败 → allocation 不写)→ 子 Session 构造
(先于事务:其 sweep 看不到尚未存在的子根 turn)→ **同一 BEGIN IMMEDIATE**
写 child_allocation item + create_turn_tx(R-T5-3)→ asyncio.create_task 驱动
子 session.submit(task, turn_id=子根 turn)。driver task 持引用;wait 用
``asyncio.wait``——过期**不取消** driver(R-T5-4,旧 subagents.py:199-201
先例逐条移植)。

children 注册表 = child_allocation items 折叠(无表列);TurnResult 尾部注记
字段 ``children_summary`` 由 Session._execute 完成后经 ``children_summary``
折叠并入(零既有构造破坏)。

core 纯度:子 Session 工厂经 composition 注入(Session.child_session_factory),
本模块不 import composition/adapters;provisioner 只认识 protocol 结构签名。

阶段4 T6 收口:
- 数据形状(ChildHandle/ChildStatus/ChildSummary/ChildGrantError/
  ChildLimitError/AGENT_TOOL_NAMES)迁居 protocol.agents(工具面协议签名
  需要它们,tools 不 import core);本模块原样 re-export,既有调用方零改动。
- R-T5-7:并发/总数 guard 按 **session 全家族**计(旧 check_child_limits
  parity):折叠父 turn 根血统下所有 child_allocation(跨 turn/跨 thread),
  不是 per-turn;depth 仍按血统参数。
- ㉒:driver 收尾 shield——teardown 窗口内再来 cancel 不打断子 Session
  关闭(Store 写者线程必须收敛);manager.aclose 经 composition 挂进
  Session 收尾链(integrations)。
"""

from __future__ import annotations

import asyncio
import logging
import math
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Protocol
from uuid import uuid4

from praxis.core.budget import build_tree, fold_turn_facts, remaining
from praxis.core.statemachine import TRANSITIONS, TurnStatus
from praxis.core.turn import pending_approval_summaries
from praxis.protocol.agents import (
    AGENT_TOOL_NAMES,
    ChildActiveError,
    ChildGrantError,
    ChildHandle,
    ChildLimitError,
    ChildStateLabel,
    ChildStatus,
    ChildSummary,
)
from praxis.protocol.budget import (
    ALLOCATION_SOURCES,
    SUBAGENT_LIMITS_DEFAULT,
    Limits,
    allocation_payload,
    cancel_request_payload,
    limits_merge_grant,
    limits_payload,
)
from praxis.protocol.items import Item, ItemKind
from praxis.protocol.json import JsonValue
from praxis.protocol.tools import Tool
from praxis.protocol.workspaces import (
    ApplyResult,
    ChildWorkspaceProvisioner,
    WorkspaceInspection,
    WorkspaceMode,
)
from praxis.store.log import ItemEntry, Store, append_items_tx
from praxis.store.turns import create_turn_tx, get_turn, list_turns

if TYPE_CHECKING:
    from praxis.core.session import Session

logger = logging.getLogger(__name__)

_STATUS_TO_LABEL: Final[Mapping[str, ChildStateLabel]] = {
    "running": "running",
    "paused": "paused",
    "interrupted": "interrupted",
    "completed": "done",
    "failed": "failed",
    "cancelled": "cancelled",
}


class ChildSessionFactory(Protocol):
    """composition 注入的子 Session 工厂结构签名(parent 由组合根闭包绑定)。

    内部仍调 build_session——Session 构造唯一点不破(CI 规则 7)。
    """

    def __call__(
        self,
        *,
        workspace: Path,
        thread_id: str,
        limits: Limits,
        tool_subset: Sequence[Tool],
        workspace_mode: WorkspaceMode,
    ) -> Session: ...


# ---------------------------------------------------------------------------
# children 注册表折叠(child_allocation items,无表列;读端 fail-open)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _AllocationRef:
    """一条 child_allocation 的身份读形(预算折叠在 core.budget,这里只取键)。"""

    allocation_id: str
    child_thread_id: str
    child_root_turn_id: str
    parent_turn_id: str
    workspace: str
    workspace_mode: str


def _parse_ref(item: Item) -> _AllocationRef | None:
    payload = item.payload
    allocation_id = payload.get("allocation_id")
    child_thread_id = payload.get("child_thread_id")
    child_root_turn_id = payload.get("child_root_turn_id")
    if (
        not isinstance(allocation_id, str)
        or not allocation_id
        or not isinstance(child_thread_id, str)
        or not child_thread_id
        or not isinstance(child_root_turn_id, str)
        or not child_root_turn_id
    ):
        return None
    parent_turn_id = payload.get("parent_turn_id")
    workspace = payload.get("workspace")
    workspace_mode = payload.get("workspace_mode")
    return _AllocationRef(
        allocation_id=allocation_id,
        child_thread_id=child_thread_id,
        child_root_turn_id=child_root_turn_id,
        parent_turn_id=parent_turn_id if isinstance(parent_turn_id, str) else (item.turn_id or ""),
        workspace=workspace if isinstance(workspace, str) else "",
        workspace_mode=workspace_mode if isinstance(workspace_mode, str) else "read_only",
    )


def _allocation_refs(items: Sequence[Item], turn_id: str) -> tuple[_AllocationRef, ...]:
    """本 turn 的 child_allocation 折叠(注册表 = items,seq 升序;畸形跳过)。"""
    refs: list[_AllocationRef] = []
    for item in sorted(items, key=lambda entry: entry.seq):
        if item.kind != ItemKind.CHILD_ALLOCATION or item.turn_id != turn_id:
            continue
        ref = _parse_ref(item)
        if ref is not None:
            refs.append(ref)
    return tuple(refs)


def _child_is_active(store: Store, ref: _AllocationRef) -> bool:
    """子根 turn 非终态 = active(running/paused/interrupted 都继续占用份额)。

    读不到子 turn → 保守 active(与 build_tree 的 fail-open 同向)。
    """
    turn = get_turn(store, ref.child_root_turn_id)
    if turn is None:
        return True
    return bool(TRANSITIONS[TurnStatus(turn.status)])


def _family_allocations(store: Store, parent_thread_id: str) -> tuple[_AllocationRef, ...]:
    """本 session 根血统下的全部 child_allocation(跨 turn/跨 thread;R-T5-7)。

    旧 ``check_child_limits``(subagents.py:41-72)的家族语义 + 控制器裁决的
    session 全家族口径:全库 allocation 折叠成 child_root_turn → parent_turn
    边表;以父 thread 的**全部 turn**为种子(顶层 session 的每个 turn 都是
    家族根——跨 turn spawn 不重置份额),各自上溯到根血统顶端(父自身是子
    时到达根 session,兄弟同样占份额),再逐层展开家族后代。max_total 计
    家族全部后代(累计,含已终态),max_concurrent 计其中 active。畸形数据
    成环时上溯 visited 守卫终止(fail-safe,不 infinite loop)。
    """
    edges: dict[str, str] = {}
    refs: dict[str, _AllocationRef] = {}
    for thread_id in store.list_threads():
        for item in store.read_items(thread_id):
            if item.kind != ItemKind.CHILD_ALLOCATION:
                continue
            ref = _parse_ref(item)
            if ref is None:
                continue
            edges[ref.child_root_turn_id] = ref.parent_turn_id
            refs[ref.child_root_turn_id] = ref
    roots: set[str] = set()
    for turn in list_turns(store, thread_id=parent_thread_id):
        root = turn.turn_id
        visited = {root}
        while root in edges:
            root = edges[root]
            if root in visited:
                break
            visited.add(root)
        roots.add(root)
    family = set(roots)
    while True:
        expanded = family | {child for child, parent in edges.items() if parent in family}
        if expanded == family:
            break
        family = expanded
    return tuple(ref for turn_id, ref in refs.items() if turn_id in family and turn_id not in roots)


# ---------------------------------------------------------------------------
# 级联唯一实现(R-T5-2)+ TurnResult 注记折叠
# ---------------------------------------------------------------------------


def cascade_for(store: Store, parent_thread_id: str, parent_turn_id: str) -> int:
    """父终态/被打断时显式逐个取消直接子 turn;返回本次级联数。

    Session 终态路径(CANCELLED/FAILED)与 recovery sweep 路径(INTERRUPTED)
    都调这一个函数——级联是代码逻辑,TRANSITIONS 不含父子规则。幂等:
    - 子已终态(COMPLETED/FAILED/CANCELLED)→ 跳过;
    - 子已有未处理 cancel_request item → 跳过(收敛归子执行者步边界);
    - allocation 指向不存在的子 turn → fail-open 跳过。
    只到直接子:孙代由子自己这一跳的终态/sweep 级联收敛(逐跳)。
    interrupted 子收到 cancel_request 后仍 interrupted(份额继续占用,
    R-T5-1),直到各自 resume(confirm=True) 步边界收敛为 CANCELLED 才 settle。
    """
    cascaded = 0
    for ref in _allocation_refs(store.read_items(parent_thread_id), parent_turn_id):
        turn = get_turn(store, ref.child_root_turn_id)
        if turn is None:
            continue
        if not TRANSITIONS[TurnStatus(turn.status)]:
            continue
        child_items = store.read_turn_items(ref.child_root_turn_id)
        if any(item.kind == ItemKind.CANCEL_REQUEST for item in child_items):
            continue
        store.append_items(
            ref.child_thread_id,
            ref.child_root_turn_id,
            [
                ItemEntry(
                    kind=ItemKind.CANCEL_REQUEST,
                    payload=cancel_request_payload(source="parent", parent_turn_id=parent_turn_id),
                )
            ],
        )
        cascaded += 1
    return cascaded


def children_summary(store: Store, thread_id: str, turn_id: str) -> tuple[ChildSummary, ...]:
    """父 turn 的 children 状态注记(Session._execute 完成后并入 TurnResult)。"""
    summaries: list[ChildSummary] = []
    for ref in _allocation_refs(store.read_items(thread_id), turn_id):
        turn = get_turn(store, ref.child_root_turn_id)
        state: ChildStateLabel = (
            _STATUS_TO_LABEL.get(turn.status, "running") if turn is not None else "running"
        )
        summaries.append(
            ChildSummary(
                child_thread_id=ref.child_thread_id,
                child_root_turn_id=ref.child_root_turn_id,
                state=state,
            )
        )
    return tuple(summaries)


def _latest_unknown_calls(items: Sequence[Item]) -> tuple[str, ...]:
    """最新一条 tool_result 仍为 outcome_unknown 的 call_id(最后结果胜)。

    与 Session.result 的展示口径同向;wait 对已关闭子 Session 的跨进程重建
    也走这一份读侧折叠(不依赖子 Session 实例存活)。
    """
    latest: dict[str, tuple[int, str]] = {}
    for item in items:
        if item.kind != ItemKind.TOOL_RESULT:
            continue
        call_id = item.payload.get("call_id")
        status = item.payload.get("status")
        if not isinstance(call_id, str) or not isinstance(status, str):
            continue
        current = latest.get(call_id)
        if current is None or item.seq >= current[0]:
            latest[call_id] = (item.seq, status)
    return tuple(cid for cid, (_seq, status) in latest.items() if status == "outcome_unknown")


# ---------------------------------------------------------------------------
# SubagentManager
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _Driver:
    """一个子的本进程驱动登记:task 持引用(wait 不取消),session 供收尾。"""

    session: Session
    task: asyncio.Task[Any]
    root_turn_id: str
    workspace: Path
    workspace_mode: WorkspaceMode
    policy_revision: str


class SubagentManager:
    """一个父 Session 的子代理编排面(T6 挂工具后生效;构造仅当工具表含 agent 工具)。

    ``limits_cfg`` 取用 ``dict(SUBAGENT_LIMITS_DEFAULT)`` 合并覆盖(不可变
    映射没有 .copy());未知键/负值/bool 边界拒绝(旧 child_limits 纪律)。
    """

    def __init__(
        self,
        *,
        parent: Session,
        store: Store,
        limits_cfg: Mapping[str, int] = SUBAGENT_LIMITS_DEFAULT,
        provisioner: ChildWorkspaceProvisioner | None = None,
    ) -> None:
        self.parent = parent
        self._store = store
        self._provisioner = provisioner
        limits = dict(SUBAGENT_LIMITS_DEFAULT)
        for name, value in limits_cfg.items():
            if name not in limits:
                raise ValueError(f"unknown subagent limit: {name!r}")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"subagent limit {name!r} must be a non-negative integer")
            limits[name] = value
        self._limits = limits
        self._drivers: dict[str, _Driver] = {}

    # -- spawn ----------------------------------------------------------------

    async def spawn(
        self,
        task: str,
        *,
        context_summary: str = "",
        requested: Limits | None = None,
        workspace_mode: WorkspaceMode = "read_only",
        depth: int = 0,
        source: str = "model",
        parent_turn_id: str | None = None,
        maintenance_purpose: str | None = None,
    ) -> ChildHandle:
        """分配份额并启动一个子代理(序见模块 docstring;guard 先于任何写入)。

        T11:``source="memory"`` = memory maintenance 子分配(唯一 spawn 原语,
        不开第二条路径)。差异仅三点:锚 turn 经 ``parent_turn_id`` 显式给定
        且**允许终态**(Q6:维护在源 turn 完成后学习);子工具表为空(旧
        bounded tool-free maintenance turn parity);binding 记
        ``memory_maintenance``/``maintenance_purpose``(stage 答案崩溃复用
        与选源跳过消费)。份额仍走 ``limits_merge_grant`` **无例外规则**
        (R-PF-5):父无限 → grant=requested parity;父有界 → 份额截断,
        不足 ChildGrantError 由调用方 retry_after 重试。
        """
        if source not in ALLOCATION_SOURCES:
            raise ValueError(f"unknown allocation source: {source!r} (expected model/memory)")
        if source == "memory":
            if parent_turn_id is None:
                raise ValueError("source='memory' requires parent_turn_id (the terminal anchor turn)")
            if not maintenance_purpose:
                raise ValueError("source='memory' requires a non-empty maintenance_purpose")
            if workspace_mode != "read_only":
                raise ValueError("memory maintenance children must be read_only")
            anchor = get_turn(self._store, parent_turn_id)
            if anchor is None or anchor.thread_id != self.parent.thread_id:
                raise ValueError(
                    f"anchor turn {parent_turn_id!r} not found in parent thread {self.parent.thread_id!r}"
                )
            # 终态锚合法:maintenance 学的就是已完成的 turn(非终态 guard 只约束 model 子)。
        else:
            if parent_turn_id is not None:
                raise ValueError("parent_turn_id override is only valid for source='memory'")
            parent_turn_id = self.parent.active_turn_id
            if parent_turn_id is None:
                raise RuntimeError("spawn requires an executing parent turn (called outside turn execution)")
            parent_turn = get_turn(self._store, parent_turn_id)
            if parent_turn is None or not TRANSITIONS[TurnStatus(parent_turn.status)]:
                raise ChildLimitError(f"parent turn {parent_turn_id!r} is terminal; cannot spawn children")
        if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
            raise ValueError("depth must be a non-negative integer")
        if depth + 1 > self._limits["max_depth"]:
            raise ChildLimitError(
                f"subagent depth limit exceeded (max_depth={self._limits['max_depth']})"
            )
        # R-T5-7:全家族口径(跨 turn)——父逐 turn spawn 仍受同一 max_total/
        # max_concurrent 约束;per-turn 计数会让每 turn 重置份额(旧 parity 病灶)。
        family = _family_allocations(self._store, self.parent.thread_id)
        if len(family) >= self._limits["max_total"]:
            raise ChildLimitError(
                f"subagent total count limit exceeded (max_total={self._limits['max_total']})"
            )
        active = sum(1 for ref in family if _child_is_active(self._store, ref))
        if active >= self._limits["max_concurrent"]:
            raise ChildLimitError(
                f"subagent concurrent limit exceeded (max_concurrent={self._limits['max_concurrent']})"
            )
        # 份额:父 remaining 来自树折叠(含已存在的 active 子占用)——B2 上限
        # 是算法不是特判,多个子逐次 spawn 也吸不干父。
        node = build_tree(
            self._store,
            thread_id=self.parent.thread_id,
            turn_id=parent_turn_id,
            root_limits=self.parent.turn_budget_limits(parent_turn_id),
            unit_costs=self.parent.unit_costs,
        )
        grant = limits_merge_grant(requested, remaining(node))
        for dimension in fields(grant):
            value = getattr(grant, dimension.name)
            if value is not None and value <= 0:
                raise ChildGrantError(
                    f"parent budget exhausted in dimension {dimension.name!r}; no child share available"
                )
        child_thread_id = uuid4().hex
        child_root_turn_id = uuid4().hex
        allocation_id = uuid4().hex
        if workspace_mode == "write":
            if self._provisioner is None:
                raise ValueError(
                    "workspace_mode='write' requires a ChildWorkspaceProvisioner "
                    "(not wired before T7); use read_only or inject a provisioner"
                )
            # prepare 失败 → 异常传播,allocation 不写(T7 序)。
            child_workspace = self._provisioner.prepare(
                self.parent.workspace, child_thread_id, workspace_mode
            )
        else:
            # read_only 共享父工作区;权限收窄(intersection + 拒 WRITE/
            # DESTRUCTIVE)在 build_child_session 构造期完成。
            child_workspace = self.parent.workspace
        factory = self.parent.child_session_factory
        if factory is None:
            raise RuntimeError("child session factory not wired (composition.build_session)")
        # 结构深度门:子工具表 = 父表减 agent 五件(T6 挂工具后生效)。
        # T11:maintenance 子工具表为空(旧 tool-free bounded turn parity)。
        tool_subset = (
            ()
            if source == "memory"
            else tuple(tool for name, tool in self.parent.tools.items() if name not in AGENT_TOOL_NAMES)
        )
        # 子 Session 先于事务构造:其 build_session sweep 看不到尚未存在的
        # 子根 turn(不会把自己的 running 根 turn 误标 interrupted)。
        child_session = factory(
            workspace=child_workspace,
            thread_id=child_thread_id,
            limits=grant,
            tool_subset=tool_subset,
            workspace_mode=workspace_mode,
        )
        # T16 ㉓:payload 拼装唯一助手(protocol/budget.py 键常量同模块)。
        payload = allocation_payload(
            allocation_id=allocation_id,
            parent_turn_id=parent_turn_id,
            child_thread_id=child_thread_id,
            child_root_turn_id=child_root_turn_id,
            workspace_mode=workspace_mode,
            workspace=str(child_workspace),
            grant=grant,
            requested=requested,
            source=source,
        )
        binding: dict[str, JsonValue] = {
            "budget": limits_payload(grant),
            "parent_turn_id": parent_turn_id,
            "root_turn_id": child_root_turn_id,
            "depth": depth + 1,
        }
        if source == "memory":
            binding["memory_maintenance"] = True
            binding["maintenance_purpose"] = maintenance_purpose

        def _tx(conn: sqlite3.Connection) -> None:
            append_items_tx(
                conn,
                self.parent.thread_id,
                parent_turn_id,
                [ItemEntry(kind=ItemKind.CHILD_ALLOCATION, payload=payload)],
            )
            create_turn_tx(
                conn, turn_id=child_root_turn_id, thread_id=child_thread_id, binding=binding
            )

        try:
            self._store.ensure_thread(child_thread_id)
            # 同一 BEGIN IMMEDIATE:allocation 与子根 turn 要么都在要么都不在
            # (R-T5-3;kill 矩阵的真 SIGKILL 验收归 T9)。
            self._store.transact(_tx)
        except BaseException:
            try:
                await child_session.aclose()
            except Exception:  # noqa: BLE001 — 收尾容忍,不掩盖原始异常
                logger.warning("child session teardown after failed spawn failed", exc_info=True)
            raise
        text = f"{context_summary}\n\n{task}" if context_summary else task
        driver_task = asyncio.create_task(
            self._drive(child_session, child_root_turn_id, text),
            name=f"child:{child_root_turn_id}",
        )
        self._drivers[child_thread_id] = _Driver(
            session=child_session,
            task=driver_task,
            root_turn_id=child_root_turn_id,
            workspace=child_workspace,
            workspace_mode=workspace_mode,
            policy_revision=child_session.permissions.policy.revision,
        )
        # 旧 subagents.py:133 先例:父故意不 wait 时也要取回 driver 异常,
        # 避免 "Task exception was never retrieved" 噪声。
        driver_task.add_done_callback(lambda t: None if t.cancelled() else t.exception())
        # 让行一步:driver 起步即持子执行锁(或已终态),后续兄弟 spawn 的
        # build_session sweep 不会把本子误判为 running 孤儿。
        await asyncio.sleep(0)
        return ChildHandle(
            child_thread_id=child_thread_id,
            child_root_turn_id=child_root_turn_id,
            allocation_id=allocation_id,
            workspace=child_workspace,
            workspace_mode=workspace_mode,
            grant=grant,
        )

    async def _drive(self, session: Session, root_turn_id: str, text: str) -> None:
        """driver task 本体:驱动子 session 执行预创建的根 turn,收尾关子 Session。

        turn 级失败已由 run_turn 错误边界收敛为 FAILED;这里只兜装配级异常。
        子 Session 的 Store 是独立实例(同库文件),driver 结束即关闭——
        不留写者线程孤儿;wait/send 经 manager 的 store 重建事实。

        ㉒ 收尾竞态承重:teardown 经 shield 独立 task 执行——request_cancel/
        aclose 读 RUNNING 后子恰终结时,后续 cancel 落在 teardown 窗口也不
        打断子 Session 关闭(否则 Store 写者线程成孤儿)。driver 在 teardown
        完成前绝不结束;期间收到的 cancel 记下来,收尾完成后重抛保持 task
        的 cancelled 结局。manager.aclose 的 gather 因此等到完整收敛。
        """
        try:
            await session.submit(text, turn_id=root_turn_id)
        except Exception:  # noqa: BLE001 — driver 边界:不上抛,只记日志
            logger.exception("child driver for turn %s failed", root_turn_id)
        finally:
            closer = asyncio.ensure_future(session.aclose())
            cancelled = False
            while not closer.done():
                try:
                    await asyncio.shield(closer)
                except asyncio.CancelledError:
                    cancelled = True
                except Exception:  # noqa: BLE001 — 收尾容忍(与 Session._close_runtime 同语义)
                    logger.warning("child session teardown failed", exc_info=True)
                    break
            if cancelled:
                raise asyncio.CancelledError

    # -- wait/send/request_cancel ----------------------------------------------

    async def wait(self, child_thread_id: str, *, timeout: float = 10.0) -> ChildStatus:
        """等子终结(或超时如实返回当前状态);timeout 钳到 0..60。

        旧 subagents.py:199-201 先例逐条移植:asyncio.wait 过期**不取消**
        driver——子继续跑,父只是不再等(R-T5-4)。
        """
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout):
            raise ValueError("wait timeout must be a finite number")
        clamped = max(0.0, min(60.0, float(timeout)))
        ref = self._ref_for(child_thread_id)
        driver = self._drivers.get(child_thread_id)
        if driver is not None and not driver.task.done() and clamped:
            # asyncio.wait never cancels the driver when a wait expires.
            await asyncio.wait({driver.task}, timeout=clamped)
        return self._child_status(ref)

    async def send(self, child_thread_id: str, message: str) -> None:
        """向子 thread append steering item;子执行者步边界读(T4 缝,零新机制)。"""
        self._ref_for(child_thread_id)
        self._store.append_items(
            child_thread_id, None, [ItemEntry(kind=ItemKind.STEERING, payload={"text": message})]
        )

    def request_cancel(self, child_thread_id: str) -> str:
        """请求取消子 turn:落 cancel_request(source=user),返回收敛路径。

        与 Session.request_cancel 同语义:本进程 driver 正在执行该子 →
        注入取消("signalled");否则仅落 item("queued",子执行者步边界收敛)。
        终态子抛 ValueError;不取执行锁、不写 turn 状态。

        ㉒ 窄竞态(读 RUNNING 后子恰终结):cancel 可能落在 driver 的
        teardown 窗口——_drive 的 shield 收尾保证子 Session 关闭不被打断,
        这里无需取锁复查(复查本身也是 TOCTOU)。
        """
        ref = self._ref_for(child_thread_id)
        turn = get_turn(self._store, ref.child_root_turn_id)
        if turn is None:
            raise ValueError(f"child root turn {ref.child_root_turn_id!r} not found")
        status = TurnStatus(turn.status)
        if not TRANSITIONS[status]:
            raise ValueError(
                f"child turn {ref.child_root_turn_id!r} is {status.value}; terminal turns cannot be cancelled"
            )
        self._store.append_items(
            ref.child_thread_id,
            ref.child_root_turn_id,
            [
                ItemEntry(
                    kind=ItemKind.CANCEL_REQUEST,
                    payload=cancel_request_payload(source="user", parent_turn_id=ref.parent_turn_id),
                )
            ],
        )
        driver = self._drivers.get(child_thread_id)
        if status is TurnStatus.RUNNING and driver is not None and not driver.task.done():
            driver.task.cancel()
            return "signalled"
        return "queued"

    async def aclose(self) -> None:
        """收尾:取消仍在跑的 driver 并等其收敛(子 Session 由 driver finally 关闭)。

        取消经 run_turn 的 CancelledError 分支全组清算收敛 CANCELLED(B14),
        不留孤儿 task。T6 把本面挂进 Session 收尾链。
        """
        tasks = [driver.task for driver in self._drivers.values() if not driver.task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    # -- integrate 交付面(T8)-------------------------------------------------

    def _integration_target(
        self, child_thread_id: str
    ) -> tuple[_AllocationRef, ChildWorkspaceProvisioner]:
        """integrate 三动作共用前置:write 子 + provisioner 接线 + 子终态。

        子须终态(completed/failed/cancelled;旧 child_workspaces parity):
        running/paused/interrupted 都拒——子工作区还在被消费时绝不 diff/apply/
        discard。读不到子 turn 保守拒绝(fail-open 同 _child_is_active 方向)。
        read_only 子/provisioner 未接线抛 ValueError(工具面如实上抛,管道
        唯一归一 tool_error,不发明第二套错误码)。
        """
        ref = self._ref_for(child_thread_id)
        if ref.workspace_mode != "write":
            raise ValueError(
                f"child {child_thread_id!r} is read_only; only write-mode children "
                "have an isolated workspace to integrate"
            )
        provisioner = self._provisioner
        if provisioner is None:
            raise ValueError(
                "write-mode child workspace provisioner is not wired; cannot integrate"
            )
        turn = get_turn(self._store, ref.child_root_turn_id)
        if turn is None or TRANSITIONS[TurnStatus(turn.status)]:
            raise ChildActiveError(
                f"child {child_thread_id!r} is still active; integrate requires a "
                "terminal child (completed/failed/cancelled)"
            )
        return ref, provisioner

    def integrate_inspect(
        self, child_thread_id: str, *, paths: Sequence[str] | None = None
    ) -> WorkspaceInspection:
        """子工作区变更的有界读面(路径集/双方 sha 表/≤200_000 预览)。"""
        ref, provisioner = self._integration_target(child_thread_id)
        return provisioner.inspect(Path(ref.workspace), paths=paths)

    def integrate_apply(
        self, child_thread_id: str, *, paths: Sequence[str] | None = None
    ) -> ApplyResult:
        """preimage 复检 + 整组 apply;conflicts 非空 = 双方字节保留零写。

        preimages 取 HEAD 基线(inspect 同源)——父仓快照后的漂移必然
        conflict,绝不静默覆盖。apply 是普通工具调用:事实 = tool_call/
        tool_result,kill 在 apply 内 → outcome_unknown → reconcile 走 T4
        既有矩阵(不写 durable intent,brief 钉死)。
        """
        ref, provisioner = self._integration_target(child_thread_id)
        workspace = Path(ref.workspace)
        inspection = provisioner.inspect(workspace, paths=paths)
        bundle = provisioner.diff(workspace, paths=paths)
        return provisioner.apply(self.parent.workspace, bundle, preimages=inspection.preimages)

    def integrate_discard(self, child_thread_id: str) -> None:
        """移除子工作区快照;provisioner.discard 路径已消失 = 幂等成功。"""
        ref, provisioner = self._integration_target(child_thread_id)
        provisioner.discard(Path(ref.workspace))

    # -- 内部重建 --------------------------------------------------------------

    def _ref_for(self, child_thread_id: str) -> _AllocationRef:
        """按 child_thread_id 在父 thread 全量 allocation 中查身份(任意父 turn)。"""
        for item in self._store.read_items(self.parent.thread_id):
            if item.kind != ItemKind.CHILD_ALLOCATION:
                continue
            if item.payload.get("child_thread_id") != child_thread_id:
                continue
            ref = _parse_ref(item)
            if ref is not None:
                return ref
        raise ValueError(f"unknown child thread {child_thread_id!r} (no allocation in parent thread)")

    def _child_status(self, ref: _AllocationRef) -> ChildStatus:
        """从 append-only 事实重建子的展示面(不依赖子 Session 实例存活)。"""
        turn = get_turn(self._store, ref.child_root_turn_id)
        state: ChildStateLabel = (
            _STATUS_TO_LABEL.get(turn.status, "running") if turn is not None else "running"
        )
        items = self._store.read_turn_items(ref.child_root_turn_id)
        conclusion = ""
        for item in reversed(items):
            if item.kind == ItemKind.MODEL_RESPONSE:
                content = item.payload.get("content")
                if isinstance(content, str):
                    conclusion = content
                break
        driver = self._drivers.get(ref.child_thread_id)
        # 跨进程/重启后无 driver:revision 未知 → 空串(无决策匹配,待批全量
        # 呈现——展示面 fail-open,不藏待批项)。
        revision = driver.policy_revision if driver is not None else ""
        facts = fold_turn_facts(items, unit_costs=self.parent.unit_costs)
        return ChildStatus(
            state=state,
            conclusion=conclusion,
            pending_approvals=pending_approval_summaries(items, policy_revision=revision),
            unknown_calls=_latest_unknown_calls(items),
            steps_used=facts.steps,
            cost_used=facts.cost_micros,
            workspace=ref.workspace,
        )
