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
"""

from __future__ import annotations

import asyncio
import logging
import math
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol
from uuid import uuid4

from praxis.core.budget import build_tree, fold_turn_facts, remaining
from praxis.core.statemachine import TRANSITIONS, TurnStatus
from praxis.core.turn import ApprovalSummary, pending_approval_summaries
from praxis.protocol.budget import (
    SUBAGENT_LIMITS_DEFAULT,
    Limits,
    limits_merge_grant,
    limits_payload,
)
from praxis.protocol.items import Item, ItemKind
from praxis.protocol.json import JsonValue
from praxis.protocol.tools import Tool
from praxis.protocol.workspaces import ChildWorkspaceProvisioner, WorkspaceMode
from praxis.store.log import ItemEntry, Store, append_items_tx
from praxis.store.turns import create_turn_tx, get_turn

if TYPE_CHECKING:
    from praxis.core.session import Session

logger = logging.getLogger(__name__)

# 子代理状态标签(wait/children_summary 共用):completed/failed 都表达
# "不再消耗",但 failed 保留独立标签供上层区分结局(与 ChildState 的 done
# 折叠不同——展示面不合并失败)。
type ChildStateLabel = Literal["running", "done", "paused", "interrupted", "cancelled", "failed"]

_STATUS_TO_LABEL: Final[Mapping[str, ChildStateLabel]] = {
    "running": "running",
    "paused": "paused",
    "interrupted": "interrupted",
    "completed": "done",
    "failed": "failed",
    "cancelled": "cancelled",
}

# T6 模型工具面五件套的名字常量(本任务先行定义:子工具表 = 父表减这五件,
# depth 结构门——孙代即使谎报 depth 也拿不到 spawn 工具)。
AGENT_TOOL_NAMES: Final[frozenset[str]] = frozenset(
    {
        "spawn_subagent",
        "wait_subagent",
        "send_to_subagent",
        "cancel_subagent",
        "integrate_child_workspace",
    }
)


class ChildGrantError(RuntimeError):  # noqa: N818 — 计划钉死的名字
    """份额公式算出的 grant 在有界维 ≤ 0:父该维已耗尽,拒绝 spawn(B2)。"""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class ChildLimitError(RuntimeError):  # noqa: N818 — 与 ChildGrantError 同族
    """并发/总数/深度/父终态 guard 拒绝(T6 映射 child_limit_reached 错误码)。"""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


@dataclass(frozen=True, slots=True)
class ChildHandle:
    """spawn 的返回身份:注册表 item 与子根 turn 的互指键 + 工作区事实。"""

    child_thread_id: str
    child_root_turn_id: str
    allocation_id: str
    workspace: Path
    workspace_mode: WorkspaceMode
    grant: Limits


@dataclass(frozen=True, slots=True)
class ChildStatus:
    """wait 的如实展示面:子的状态/结论/待批/未知调用/已用步数与成本。

    paused/interrupted 可见但不代批(Q3 直面裁决:人类对子 session 各自
    resume/confirm,父不替子审批)。
    """

    state: ChildStateLabel
    conclusion: str
    pending_approvals: tuple[ApprovalSummary, ...]
    unknown_calls: tuple[str, ...]
    steps_used: int
    cost_used: int
    workspace: str


@dataclass(frozen=True, slots=True)
class ChildSummary:
    """TurnResult.children_summary 的一条注记:父 turn 结束时子的状态快照。"""

    child_thread_id: str
    child_root_turn_id: str
    state: ChildStateLabel


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
                    payload={"source": "parent", "parent_turn_id": parent_turn_id},
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
    ) -> ChildHandle:
        """分配份额并启动一个子代理(序见模块 docstring;guard 先于任何写入)。"""
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
        refs = _allocation_refs(self._store.read_items(self.parent.thread_id), parent_turn_id)
        if len(refs) >= self._limits["max_total"]:
            raise ChildLimitError(
                f"subagent total count limit exceeded (max_total={self._limits['max_total']})"
            )
        active = sum(1 for ref in refs if _child_is_active(self._store, ref))
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
        tool_subset = tuple(
            tool for name, tool in self.parent.tools.items() if name not in AGENT_TOOL_NAMES
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
        payload: dict[str, JsonValue] = {
            "allocation_id": allocation_id,
            "parent_turn_id": parent_turn_id,
            "child_thread_id": child_thread_id,
            "child_root_turn_id": child_root_turn_id,
            "workspace_mode": workspace_mode,
            "workspace": str(child_workspace),
            "grant": limits_payload(grant),
            "requested": limits_payload(requested) if requested is not None else {},
            "source": "model",
        }
        binding: dict[str, JsonValue] = {
            "budget": limits_payload(grant),
            "parent_turn_id": parent_turn_id,
            "root_turn_id": child_root_turn_id,
            "depth": depth + 1,
        }

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
        """
        try:
            await session.submit(text, turn_id=root_turn_id)
        except Exception:  # noqa: BLE001 — driver 边界:不上抛,只记日志
            logger.exception("child driver for turn %s failed", root_turn_id)
        finally:
            try:
                await session.aclose()
            except Exception:  # noqa: BLE001 — 收尾容忍(与 Session._close_runtime 同语义)
                logger.warning("child session teardown failed", exc_info=True)

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
                    payload={"source": "user", "parent_turn_id": ref.parent_turn_id},
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