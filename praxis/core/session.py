"""Praxis Session:thread 生命周期、人工确认续跑协议与 steering(设计 §4.1/§5.2/§5.8)。

Session 是 turn 循环之上的会话门面:它不执行任何东西,只负责"什么时候、以什么
身份、拿不拿锁"地调用 run_turn,并兑现 §5.8 的三条硬语义——

1. **执行锁 = 执行者身份**:submit/resume 在 run_turn 外围持 thread 执行锁,
   锁被占(ThreadBusy)一律换成公开的 ``ConversationBusy``;steer/approval/
   reconcile/confirm 只是 append item,永不取锁(§5.8:它们不是执行者)。
2. **人工确认守卫**(§5.2:interrupted 只能由人重启):INTERRUPTED turn 必须
   ``confirm=True``,否则 ``ResumeNotConfirmed``;PAUSED turn 必须给**非空**的
   approval(审批)或 reconcile(结果对账)映射——空映射不是决策,同样拒绝;
   RUNNING 与终态 turn 不可 resume。
3. **classify_pending 决策矩阵**(§5.9,recovery 提供分类,本模块提供动作):
   未 started 的待决调用写 not_executed 结果(它确实没跑);started+幂等自动
   重跑(经唯一管道);started+非幂等写 outcome_unknown 并停在 PAUSED 等人对账
   ——续跑的代价不对称,绝不自动重放未知副作用。

resume 的处理序(与代码一一对应):守卫 → 映射校验(目标存在、verdict/outcome
合法——在**任何** transition 之前,失败即零写入,绝不把 turn 留在无执行者的
RUNNING 等 sweep 误标 interrupted)→ transition(RUNNING,真实 reason) →
reconcile 写结果(先) → approval 写决策(摘要从 approval_request item 取,绑定
digest;policy_revision stamp 当前策略——R5(b) 批准时刻绑定)→ classify_pending
分类(**先 started 后 decision**:未 started 且有
matching decision 的调用进重放队列承接审批重入;已 started 的决策一律失效于
安全矩阵——非幂等 outcome_unknown,见 _classify_and_prepare)→ 若有 outcome_unknown 则
transition(PAUSED) 直接返回 → 否则带 resume_calls 重入 run_turn。

已知边界(Phase 1,如实):
- steps 不跨 pause/resume 累计(每次 run_turn 从 0 计)——预算归 §5.5 的
  Phase 4 budget 树,此处不建预算;
- 分类写入在锁外完成(与 brief 钉死的序一致):并发 resume 同一 turn 时,第二条
  ``transition(→RUNNING)`` 会因 RUNNING→RUNNING 非法而 IllegalTransition 失败,
  这是天然的单执行者守卫;
- 本模块定义的 ``ConversationBusy``/``ResumeNotConfirmed`` 由 interfaces.sdk
  re-export(sdk 是 CLI 唯一入口,R2):core 不得 import interfaces
  (import-linter praxis_core_purity),异常类只能定义在这里、从 sdk 出口。

Session 构造只允许 praxis/composition.py(CI 规则 7,设计 §4.1 唯一组合根)。
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Literal, Protocol
from uuid import uuid4

from praxis.core.compaction import SUMMARY_MAX_OUTPUT_TOKENS, SUMMARY_PROMPT_MAX_CHARS
from praxis.core.events import EventBus, Subscription
from praxis.core.locks import ThreadBusy, thread_execution_lock
from praxis.core.permission import apply_update, command_network_verdict, extend_policy
from praxis.core.pipeline import SandboxCompiler, SandboxFinalizer
from praxis.core.recovery import classify_pending
from praxis.core.statemachine import TurnStatus, transition
from praxis.core.turn import (
    ApprovalSummary,
    TurnDeps,
    TurnResult,
    TurnState,
    pending_approval_summaries,
    run_turn,
)
from praxis.protocol.hooks import HookSetup
from praxis.protocol.items import Item, ItemKind
from praxis.protocol.json import JsonValue
from praxis.protocol.models import ModelAdapter, ModelSettings, ModelToolCall, WireRequest
from praxis.protocol.permission import (
    PermissionContext,
    PermissionNotice,
    PermissionUpdate,
)
from praxis.protocol.policy import NetworkRule
from praxis.protocol.retry import RetryingStream
from praxis.protocol.skills import SkillCatalog
from praxis.protocol.tools import ResultStatus, Tool
from praxis.store.artifacts import ArtifactStore
from praxis.store.log import ItemEntry, Store
from praxis.store.turns import Turn, create_turn, get_turn, list_turns
from praxis.store.verify import VerifyReport, verify_store

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT_TEMPLATE = (
    "You are Praxis, a workspace coding agent. Workspace: {workspace}. "
    "Use tools to inspect and modify files. Answer concisely."
)

_NOT_EXECUTED_TEXT = "[not executed: turn interrupted before dispatch]"
_OUTCOME_UNKNOWN_TEXT = "[outcome unknown: process died during execution]"

_APPROVAL_VERDICTS = frozenset({"allow", "deny"})
_RECONCILE_OUTCOMES = frozenset({"ok", "failed"})

# 阶段 2 留档②:D9 授权披露的逐字文案(旧代码静默是运维陷阱)。
_D9_NOTICE_KIND = "command_network_blocked"
_D9_NOTICE_TEXT = (
    "granting this network rule makes bash command network fail-closed (D9); "
    "approvals for bash will be required as whole-tool asks"
)
# R-T11-2:once 授权只在当前执行批生效(批末由 _execute 清除),披露必须如实标范围。
_D9_ONCE_SUFFIX = " — this effect lasts only for the current execution batch (once grant)."


class ConversationBusy(RuntimeError):  # noqa: N818 — 设计 §5.8/plan 钉死的名字
    """该 thread 的执行锁被另一执行者持有:submit/resume 拒绝排队(§5.8)。"""


class ResumeNotConfirmed(RuntimeError):  # noqa: N818 — 设计 §5.2 守卫钉死的名字
    """interrupted turn 未经人工确认(confirm=True)不得续跑(§5.2 守卫)。"""


class AsyncCloser(Protocol):
    """composition 注入的异步收尾面(Task 15/T19;core 不 import 具体实现)。

    ``CommandSessions.aclose()`` 与 T19 ``McpBundle.aclose()`` 均结构匹配。
    Session 只经这个协议触达后台命令会话与 MCP integrations:取消在跑命令、
    删除 marker、关闭 MCP owner task/沙箱产物——不留活口。
    """

    async def aclose(self) -> None: ...


# Task 15 名字保留为同一协议的别名(不新增第二 closer 协议;T19 ruling)。
CommandSessionsCloser = AsyncCloser


# ---------------------------------------------------------------------------
# item 读取助手(全部面向"本 turn 的完整 item 列表",纯函数)
# ---------------------------------------------------------------------------


def _call_names(items: Sequence[Item]) -> dict[str, str]:
    """call_id → 工具名:从最后出现的 model_response/tool_call/tool_result 取。

    B31 判断(last-wins 对 reconcile 无害,交档):reconcile 映射的键本身就是裸
    call_id,同 turn 跨轮复用 id 时 API 无法区分两条调用;可成为对账目标的必然
    是尚无结果的那一轮,而"最后出现的同名调用"正是最近的声明窗口及其
    tool_call——更早窗口已有结果,不会是需要命名的对账目标。裸 id 语义不变,
    窗口化 reconcile API 属阶段 4。
    """
    names: dict[str, str] = {}
    for item in items:
        if item.kind == ItemKind.MODEL_RESPONSE:
            raw_calls = item.payload.get("tool_calls")
            if isinstance(raw_calls, (list, tuple)):
                for entry in raw_calls:
                    if not isinstance(entry, Mapping):
                        continue
                    call_id, name = entry.get("call_id"), entry.get("name")
                    if isinstance(call_id, str) and isinstance(name, str):
                        names[call_id] = name
        elif item.kind in (ItemKind.TOOL_CALL, ItemKind.TOOL_RESULT, ItemKind.APPROVAL_REQUEST):
            call_id, name = item.payload.get("call_id"), item.payload.get("tool_name")
            if isinstance(call_id, str) and isinstance(name, str):
                names[call_id] = name
    return names


def _latest_payloads(items: Sequence[Item], kind: str) -> dict[str, Mapping[str, JsonValue]]:
    """某 kind 下 call_id → 最后一条 payload(同 call_id 多次请求/决策取最新)。

    B31 判断(交档):
    - 审批**选择/落笔**面(last-wins 无害):``_validate_approval`` 与
      ``_write_approval_decisions`` 取最近一条 approval_request 的 digest——裸
      call_id 重入只可能指向最近声明窗口,与管道 ``_latest_approval_decision``
      的窗口无关绑定同口径;取最新正是重入要绑定/执行的动作。
    - ``_pending_approvals`` 的哈希键仍是裸 call_id,但抑制判据由
      ``turn.pending_approval_summaries`` 单点实现(C-1 + K-2):
      ``(call_id, args_digest)`` 且决策 revision 等于当前 revision。裸 id
      审批映射在"同 id 两轮同时待批"时仍只能呈现一条(取最新),这是 resume
      API 的固有边界,窗口化属阶段 4。
    """
    latest: dict[str, Mapping[str, JsonValue]] = {}
    for item in items:
        if item.kind != kind:
            continue
        call_id = item.payload.get("call_id")
        if isinstance(call_id, str):
            latest[call_id] = item.payload
    return latest


def _pending_approvals(
    items: Sequence[Item], *, policy_revision: str
) -> tuple[ApprovalSummary, ...]:
    """本 turn 内没有**当前 revision 有效**决策的 approval_request(镜像 turn 层口径)。

    K-2:判据集中在 ``turn.pending_approval_summaries``——两处不再各写一份。
    C-1:抑制条件 = 同 ``(call_id, args_digest)`` **且决策 revision 等于当前
    revision**。否则同 turn 跨轮复用 call_id、第二轮不同参数 re-ask 时(或
    批准后策略修订使旧批准作废、管道 re-ask 时),旧决策会把新一轮请求藏掉,
    PAUSED 的 pending_approvals 为空 → 人类无可批项(B32 同族)。
    """
    return pending_approval_summaries(items, policy_revision=policy_revision)


def _last_model_response_text(items: Sequence[Item]) -> str:
    for item in reversed(items):
        if item.kind == ItemKind.MODEL_RESPONSE:
            content = item.payload.get("content")
            if isinstance(content, str):
                return content
    return ""


def _last_status_reason(items: Sequence[Item]) -> str | None:
    for item in reversed(items):
        if item.kind == ItemKind.TURN_STATUS:
            reason = item.payload.get("reason")
            return reason if isinstance(reason, str) else None
    return None


def _unresolved_unknown_calls(items: Sequence[Item]) -> tuple[str, ...]:
    """最新一条 tool_result 仍为 outcome_unknown 的 call_id(R10 口径:最后结果胜)。"""
    latest: dict[str, tuple[int, int, str]] = {}
    for position, item in enumerate(items):
        if item.kind != ItemKind.TOOL_RESULT:
            continue
        call_id = item.payload.get("call_id")
        status = item.payload.get("status")
        if not isinstance(call_id, str) or not isinstance(status, str):
            continue
        key = (item.seq, position)
        current = latest.get(call_id)
        if current is None or key >= current[:2]:
            latest[call_id] = (key[0], key[1], status)
    return tuple(cid for cid, (_seq, _pos, status) in latest.items() if status == "outcome_unknown")


def _text_result_entry(
    call_id: str,
    tool_name: str,
    status: ResultStatus,
    text: str,
    metadata: Mapping[str, JsonValue],
) -> ItemEntry:
    """会话层合成的 tool_result item:与管道落库字段形状一致(pipeline 约定,勿漂移)。"""
    payload: dict[str, JsonValue] = {
        "call_id": call_id,
        "tool_name": tool_name,
        "status": status.value,
        "content": ({"type": "text", "data": {"text": text}},),
        "is_error": status is ResultStatus.ERROR,
        "error_code": "user_reconciled_failed" if status is ResultStatus.ERROR else None,
        "error_message": text if status is ResultStatus.ERROR else None,
        "truncated": False,
        "metadata": dict(metadata),
    }
    return ItemEntry(kind=ItemKind.TOOL_RESULT, payload=payload)


# ---------------------------------------------------------------------------
# resume 映射校验(纯函数;在 transition(RUNNING) **之前**调用,见 Session.resume)
# ---------------------------------------------------------------------------


def _validate_reconcile(
    turn_id: str, items: Sequence[Item], reconcile: Mapping[str, str]
) -> Mapping[str, str]:
    """reconcile:outcome 必须 ok/failed、目标必须出现在本 turn items;返回 call_id→工具名。"""
    names = _call_names(items)
    for call_id, outcome in reconcile.items():
        if outcome not in _RECONCILE_OUTCOMES:
            raise ValueError(f"reconcile outcome must be ok/failed, got {outcome!r}")
        if call_id not in names:
            raise ValueError(f"reconcile target {call_id!r} not found in turn {turn_id!r} items")
    return names


def _validate_approval(
    turn_id: str, items: Sequence[Item], approval: Mapping[str, str]
) -> Mapping[str, Mapping[str, JsonValue]]:
    """approval:verdict 必须 allow/deny、目标必须有 matching approval_request;返回请求表。"""
    requests = _latest_payloads(items, ItemKind.APPROVAL_REQUEST)
    for call_id, verdict in approval.items():
        if verdict not in _APPROVAL_VERDICTS:
            raise ValueError(f"approval verdict must be allow/deny, got {verdict!r}")
        if call_id not in requests:
            raise ValueError(f"no approval_request for call {call_id!r} in turn {turn_id!r}")
    return requests


# ---------------------------------------------------------------------------
# 策略声明落盘(原子写:同目录 tmp + os.replace,与 filesystem 内置工具同语义;
# 不跨层借 tools.builtins 的私有助手——ACI 各层自有实现,R6-4)
# ---------------------------------------------------------------------------


def _atomic_write_text(path: Path, content: str) -> None:
    tmp = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        tmp.write_text(content, encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------


class Session:
    """一个 thread 的会话门面:提交/续跑/转向/查询。构造只允许 composition.py。"""

    def __init__(
        self,
        store: Store,
        bus: EventBus,
        workspace: Path,
        adapter: ModelAdapter,
        tools: Mapping[str, Tool],
        permissions: PermissionContext,
        thread_id: str,
        max_steps: int = 50,
        system_prompt: str | None = None,
        compile_sandbox: SandboxCompiler | None = None,
        finalize_sandbox: SandboxFinalizer | None = None,
        retry_stream: RetryingStream | None = None,
        artifacts: ArtifactStore | None = None,
        compact_limit_tokens: int | None = None,
        measure_tokens: Callable[[WireRequest], int] | None = None,
        summary_max_output_tokens: int = SUMMARY_MAX_OUTPUT_TOKENS,
        summary_prompt_max_chars: int = SUMMARY_PROMPT_MAX_CHARS,
        command_sessions: AsyncCloser | None = None,
        hooks: HookSetup | None = None,
        skills_catalog: SkillCatalog | None = None,
        integrations: Sequence[AsyncCloser] = (),
    ) -> None:
        self._store = store
        self._bus = bus
        self._workspace = Path(workspace)
        self._adapter = adapter
        self._tools = dict(tools)
        self._permissions = permissions
        # R13:base = composition 载入/构造的策略声明(live policy 之前)。
        # session/once 更新会扩展 live policy(仅内存生效)但绝不进 base;
        # always 落盘 = base ⊕ always_rules 折叠,杜绝 session 授权静默升级。
        self._base_policy = permissions.policy
        self._thread_id = thread_id
        self._max_steps = max_steps
        self._system_prompt = (
            system_prompt
            if system_prompt is not None
            else DEFAULT_SYSTEM_PROMPT_TEMPLATE.format(workspace=self._workspace)
        )
        self._compile_sandbox = compile_sandbox
        self._finalize_sandbox = finalize_sandbox
        self._retry_stream = retry_stream
        self._artifacts = artifacts
        self._compact_limit_tokens = compact_limit_tokens
        self._measure_tokens = measure_tokens
        self._summary_max_output_tokens = summary_max_output_tokens
        self._summary_prompt_max_chars = summary_prompt_max_chars
        self._command_sessions = command_sessions
        # T19:宿主装配的异步收尾面(MCP bundle 等);close/aclose 统一收尾。
        self._integrations = tuple(integrations)
        self._hooks = hooks
        # None = 不注入 skill 目录(既有测试/单测路径);composition 注入 loader 产出。
        self._skills_catalog = skills_catalog

    # -- 身份与生命周期 -------------------------------------------------------

    @property
    def thread_id(self) -> str:
        return self._thread_id

    @property
    def permissions(self) -> PermissionContext:
        """当前 live 权限上下文的只读快照。

        composition 的 web authorizer 经它读取**实时**策略(每次调用重新取,
        绝不缓存快照):session 中途授权后,下一跳/下一次请求立刻按新策略裁决。
        """
        return self._permissions

    def subscribe(self) -> Subscription:
        """订阅本会话的事件总线(UI 提示通道,§5.6;事件非事实,item 日志才是)。"""
        return self._bus.subscribe()

    async def aclose(self) -> None:
        """异步收尾:**command_sessions → integrations → store**(顺序钉死)。

        MCP owner task 活在连接它的 loop 里,必须在 loop 关闭前 await 收尾;
        CLI 的 ``finally: await session.aclose()`` 走这里。单条收尾失败只记日志
        (mirror command_sessions tolerance),绝不让一个坏 integration 挡住其余
        收尾或 store 关闭。
        """
        try:
            await self._close_runtime()
        finally:
            self._store.close()

    async def _close_runtime(self) -> None:
        sessions = self._command_sessions
        if sessions is not None:
            try:
                await sessions.aclose()
            except Exception:  # noqa: BLE001 — 收尾容忍(与既有 close 语义一致)
                logger.warning("command sessions teardown failed", exc_info=True)
        for integration in self._integrations:
            try:
                await integration.aclose()
            except Exception:  # noqa: BLE001 — 一个 integration 不得阻塞其余收尾
                logger.warning("session integration teardown failed", exc_info=True)

    def close(self) -> None:
        """同步收尾;EventBus 无持久资源,不另设关闭。

        Task 15/T19:先走后台命令会话与 integrations 收尾(取消在跑会话、删
        marker、关闭 MCP 连接),最后关 store。同步 API 在无运行中事件循环时用
        ``asyncio.run``;若被异步宿主调用,交给当前循环(store 仍立即关闭——
        集成本身不依赖 store,T19 测试钉住这一顺序)。
        """
        if self._command_sessions is None and not self._integrations:
            self._store.close()
            return
        try:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(self._close_runtime())
            else:
                asyncio.ensure_future(self._close_runtime())
        finally:
            # 修复轮 #5:closer 抛 BaseException(如 KeyboardInterrupt)时也绝不
            # 让写者线程成为孤儿——store.close 必须执行。
            self._store.close()

    # -- 权限更新(Task 8:once/session/always;设计 §5.3/§5.12;裁决 R6)-----

    def update_permission(self, update: PermissionUpdate) -> PermissionNotice | None:
        """权限更新的会话级唯一入口:经 apply_update 纯代数落地,always 同批落盘。

        once/session 只改内存(once 由 _execute 在下一批结束时清除;session
        永不落盘)。always 额外落盘,R13:写的是 base 声明 ⊕ always_rules
        折叠(非 live policy——后者还含 session 扩展,直 dump 会静默升级
        session→always),原子写 ``.praxis/permissions.json``——该文件即策略
        声明,composition 下次打开时按"显式参数 > 文件 > 默认"加载。

        D9 披露(阶段 2 留档② + R-T11-2):NetworkRule 以 session/always 落地后,
        若 live 的 ``command_network_verdict`` 为 deny(任何来源存在域过滤即
        fail closed),返回 ``PermissionNotice``;once 授权同理,但仅当它**首次**
        把命令网络从非 deny 翻成 deny(已 deny 时无行为变化、不重复告警),且
        文案带"仅当前执行批"的范围后缀。其余路径返回 None——零行为变化。
        """
        once_flips_deny = False
        if isinstance(update.rule, NetworkRule) and update.destination == "once":
            once_flips_deny = command_network_verdict(self._permissions) != "deny"
        self._permissions = apply_update(self._permissions, update)
        if update.destination == "always":
            self._persist_policy()
        if isinstance(update.rule, NetworkRule) and command_network_verdict(self._permissions) == "deny":
            if update.destination in ("session", "always"):
                return PermissionNotice(kind=_D9_NOTICE_KIND, text=_D9_NOTICE_TEXT)
            if update.destination == "once" and once_flips_deny:
                return PermissionNotice(kind=_D9_NOTICE_KIND, text=_D9_NOTICE_TEXT + _D9_ONCE_SUFFIX)
        return None

    def _persist_policy(self) -> None:
        """R13 落盘真值 = base 声明 ⊕ always_rules 顺序折叠;session/once 绝不上盘。

        直接 dump live policy 会把 session 授权静默升级为 always(R6 的
        session 扩展只该活在内存)——违反 D10。base(composition 载入的声明)
        叠加 always_rules 记录层(存在正是为此)才是文件该写的内容;每条折叠
        经 extend_policy 重验(R11),mode 不进 policy(C5,approval 域只有
        on-request/never)。原子写:同目录 tmp + os.replace。
        """
        persisted = self._base_policy
        for rule in self._permissions.always_rules:
            persisted = extend_policy(persisted, rule)
        praxis_dir = self._workspace / ".praxis"
        praxis_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_text(praxis_dir / "permissions.json", persisted.model_dump_json(indent=2))

    # -- 审批流携带的权限更新(resume 的可选面)------------------------------

    def _validate_permission_updates(
        self, updates: Sequence[PermissionUpdate]
    ) -> None:
        """对全部更新做一次纯代数折叠预验:任何非法 grant 在 transition 前抛出。

        与 ``update_permission`` 走同一 ``apply_update`` 代数(R11 完整重验),
        只算不提交——要么整批合法,要么零写入。
        """
        probe = self._permissions
        for update in updates:
            probe = apply_update(probe, update)

    def _apply_permission_updates(
        self, updates: Sequence[PermissionUpdate]
    ) -> tuple[PermissionNotice, ...]:
        """逐条提交(always 落盘)并收集 D9 披露;空输入 = 空元组。"""
        notices: list[PermissionNotice] = []
        for update in updates:
            notice = self.update_permission(update)
            if notice is not None:
                notices.append(notice)
        return tuple(notices)

    # -- submit ---------------------------------------------------------------

    async def submit(self, text: str) -> TurnResult:
        """新 turn:append user_message → create_turn(INSERT running) → 持锁执行。

        turn_id 先生成:user_message 以本 turn 归因落库(先 item 后 turns 行,
        顺序由 brief 钉死;锁被占时留下的 running 孤儿 turn 由下次 sweep 收编,
        这正是"崩溃即 interrupted"同一条收敛路径)。
        """
        turn_id = uuid4().hex
        self._store.append_items(
            self._thread_id, turn_id, [ItemEntry(kind=ItemKind.USER_MESSAGE, payload={"text": text})]
        )
        create_turn(
            self._store,
            turn_id=turn_id,
            thread_id=self._thread_id,
            binding={"model": self._model_id()},
        )
        return await self._execute(turn_id, ())

    # -- resume ---------------------------------------------------------------

    async def resume(
        self,
        turn_id: str,
        *,
        confirm: bool = False,
        approval: Mapping[str, Literal["allow", "deny"]] | None = None,
        reconcile: Mapping[str, Literal["ok", "failed"]] | None = None,
        permission_updates: Sequence[PermissionUpdate] = (),
    ) -> TurnResult:
        """续跑 interrupted/paused turn——审批决策以 item 为事实显式落库,
        授权建议经 ``permission_updates`` update 通道落地(两者不混写)。

        ``permission_updates`` 是审批流里人类**显式选择**的建议落地(例如 URL
        ask 的域级 NetworkRule):先纯代数预验(零写入),再逐条经
        ``update_permission`` 提交(always 同批落盘),产生的 D9 披露收集进
        TurnResult.notices 供 CLI 打印。审批本身仍是 once 决策——绝不因批准
        一次调用而自动放宽权限。
        """
        turn = get_turn(self._store, turn_id)
        if turn is None:
            raise ValueError(f"unknown turn_id: {turn_id!r}")
        if turn.thread_id != self._thread_id:
            raise ValueError(
                f"turn {turn_id!r} belongs to thread {turn.thread_id!r}, not this session's "
                f"thread {self._thread_id!r} (different thread)"
            )
        status = TurnStatus(turn.status)
        if status is TurnStatus.INTERRUPTED:
            if not confirm:
                raise ResumeNotConfirmed(
                    f"turn {turn_id!r} is interrupted; resume requires confirm=True (design §5.2)"
                )
        elif status is TurnStatus.PAUSED:
            if not approval and not reconcile:  # 空映射不是决策:不得静默丢弃待决审批/对账
                raise ValueError("paused turn needs approval or reconcile")
        elif status is TurnStatus.RUNNING:
            raise ValueError(f"turn {turn_id!r} is still running; not resumable")
        else:
            raise ValueError(f"turn {turn_id!r} is {status.value}; terminal turns are not resumable")

        # 映射校验在 transition 之前:非法目标/verdict/outcome 绝不把 turn 转成
        # 无执行者的 RUNNING——那会被下次 open_session 的 sweep 误标 interrupted,
        # 伪造"进程死亡"的恢复现场。校验失败 = 零 transition、零 item 写入。
        if reconcile or approval:
            probe_items = self._store.read_turn_items(turn_id)
            if reconcile:
                _validate_reconcile(turn_id, probe_items, reconcile)
            if approval:
                _validate_approval(turn_id, probe_items, approval)
        # 建议落地先纯代数预验(R11:非法 grant 在 transition 之前抛,零写入)。
        self._validate_permission_updates(permission_updates)

        transition(
            self._store,
            self._bus,
            turn_id,
            TurnStatus.RUNNING,
            reason=self._resume_reason(status, approval, reconcile),
        )
        # 授权先于 approval_decision 落笔:decision 必须 stamp 更新后的 revision,
        # 否则重入按 revision 绑定判旧批准作废,刚批准的动作会被 re-ask。
        notices = self._apply_permission_updates(permission_updates)
        # reconcile 先写(对账是给"已有 tool_call 无结果"的调用一个结论);
        # approval 写决策,digest 一律取自匹配的 approval_request item(摘要绑定)。
        if reconcile:
            self._write_reconcile(turn_id, reconcile)
        if approval:
            self._write_approval_decisions(turn_id, approval)

        resume_calls, unknown_ids = self._classify_and_prepare(turn_id)
        if unknown_ids:
            transition(self._store, self._bus, turn_id, TurnStatus.PAUSED, reason="outcome_unknown")
            items = self._store.read_turn_items(turn_id)
            return TurnResult(
                turn_id=turn_id,
                status=TurnStatus.PAUSED,
                output=_last_model_response_text(items),
                reason="outcome_unknown",
                pending_approvals=_pending_approvals(
                    items, policy_revision=self._permissions.policy.revision
                ),
                unknown_calls=tuple(unknown_ids),
                notices=notices,
            )
        return replace(await self._execute(turn_id, resume_calls), notices=notices)

    @staticmethod
    def _resume_reason(
        status: TurnStatus,
        approval: Mapping[str, Literal["allow", "deny"]] | None,
        reconcile: Mapping[str, Literal["ok", "failed"]] | None,
    ) -> str:
        if status is TurnStatus.INTERRUPTED:
            return "resume_confirmed"
        if reconcile:
            return "reconciled"
        return "approval_decided"

    def _write_reconcile(self, turn_id: str, reconcile: Mapping[str, Literal["ok", "failed"]]) -> None:
        items = self._store.read_turn_items(turn_id)
        names = _validate_reconcile(turn_id, items, reconcile)  # resume 已前置校验;此处幂等复核
        entries: list[ItemEntry] = []
        for call_id, outcome in reconcile.items():
            status = ResultStatus.OK if outcome == "ok" else ResultStatus.ERROR
            text = f"reconciled by user: {outcome}"
            entries.append(_text_result_entry(call_id, names[call_id], status, text, {"reconciled": True}))
        self._store.append_items(self._thread_id, turn_id, entries)

    def _write_approval_decisions(
        self, turn_id: str, approval: Mapping[str, Literal["allow", "deny"]]
    ) -> None:
        items = self._store.read_turn_items(turn_id)
        requests = _validate_approval(turn_id, items, approval)  # 同上:幂等复核
        entries: list[ItemEntry] = []
        for call_id, verdict in approval.items():
            request = requests[call_id]
            payload: dict[str, JsonValue] = {
                "call_id": call_id,
                "verdict": verdict,
                "args_digest": request.get("args_digest"),
                "destination": "once",
                # R5(b):批准绑定批准时刻的 policy revision(当前 ctx 策略,而非
                # request 的旧 revision)——管道重入按 decision item 校验,
                # 策略变更使旧批准作废(fail-closed re-ask)。
                "policy_revision": self._permissions.policy.revision,
            }
            entries.append(ItemEntry(kind=ItemKind.APPROVAL_DECISION, payload=payload))
        self._store.append_items(self._thread_id, turn_id, entries)

    def _classify_and_prepare(self, turn_id: str) -> tuple[list[ModelToolCall], list[str]]:
        """classify_pending 决策矩阵 → (重放队列, outcome_unknown 清单)。

        优先级(R12 评审修复):**先 started,后 decision**。未 started 且持
        matching approval_decision 的调用进重放队列——正常"ask→批准→重入"里
        调用从未 dispatch(request 先于 tool_call),审批重入由管道内 digest 绑定
        落地(Task 12/13 交接语义)。而 approve→dispatch→crash 的调用一旦
        started,决策就绝不能再开自动重放的安全通道:非幂等一律 outcome_unknown
        (未知副作用只能人工对账),幂等照常自动重试。其余按中断矩阵:
        not_started→not_executed / started+幂等→重放 / started+非幂等→unknown。
        """
        items = self._store.read_turn_items(turn_id)
        decided = set(_latest_payloads(items, ItemKind.APPROVAL_DECISION))
        resume_calls: list[ModelToolCall] = []
        unknown_ids: list[str] = []
        entries: list[ItemEntry] = []
        for pending in classify_pending(items, self._tools):
            call = pending.call
            if not pending.started:
                if call.call_id in decided:
                    resume_calls.append(call)  # 审批重入:未 dispatch,无未知副作用
                else:
                    entries.append(
                        _text_result_entry(
                            call.call_id, call.name, ResultStatus.NOT_EXECUTED, _NOT_EXECUTED_TEXT, {}
                        )
                    )
            elif pending.idempotent:
                resume_calls.append(call)
            else:
                entries.append(
                    _text_result_entry(
                        call.call_id, call.name, ResultStatus.OUTCOME_UNKNOWN, _OUTCOME_UNKNOWN_TEXT, {}
                    )
                )
                unknown_ids.append(call.call_id)
        if entries:
            self._store.append_items(self._thread_id, turn_id, entries)
        return resume_calls, unknown_ids

    # -- steering(§5.8:append-only,不取执行锁)-----------------------------

    def steer(self, text: str) -> None:
        """向日志 append 一条 steering:任何进程任何时刻可写;执行者在步边界读到。"""
        self._store.append_items(
            self._thread_id, None, [ItemEntry(kind=ItemKind.STEERING, payload={"text": text})]
        )

    # -- 查询 ------------------------------------------------------------------

    def status(self, turn_id: str) -> Turn:
        turn = get_turn(self._store, turn_id)
        if turn is None:
            raise ValueError(f"unknown turn_id: {turn_id!r}")
        return turn

    def result(self, turn_id: str) -> TurnResult:
        """从 items 重建 TurnResult:最后 model_response 文本、最后 turn_status
        reason、未决 approval_request、仍为 outcome_unknown 的调用清单。"""
        turn = self.status(turn_id)
        items = self._store.read_turn_items(turn_id)
        return TurnResult(
            turn_id=turn_id,
            status=TurnStatus(turn.status),
            output=_last_model_response_text(items),
            reason=_last_status_reason(items),
            pending_approvals=_pending_approvals(
                items, policy_revision=self._permissions.policy.revision
            ),
            unknown_calls=_unresolved_unknown_calls(items),
        )

    # -- 只读日志/turn 面(RULING R14:CLI 的 log/verify/status 经 sdk 触达)------

    def items(self, *, after_seq: int = 0) -> list[Item]:
        """本 thread 全量 item 日志(seq > after_seq,升序);纯读,不取任何锁。"""
        return self._store.read_items(self._thread_id, after_seq=after_seq)

    def turns(self) -> list[Turn]:
        """本 thread 的 turn 快照列表(创建序);纯读。"""
        return list_turns(self._store, thread_id=self._thread_id)

    def verify(self) -> VerifyReport:
        """对本会话所属 store 做只读校验扫描(offline verify,永不修复)。"""
        return verify_store(self._store)

    # -- 内部执行入口 ----------------------------------------------------------

    async def _execute(self, turn_id: str, resume_calls: Sequence[ModelToolCall]) -> TurnResult:
        """持 thread 执行锁跑 run_turn;锁忙 → ConversationBusy(不排队,§5.8)。"""
        deps = TurnDeps(
            store=self._store,
            bus=self._bus,
            adapter=self._adapter,
            tools=self._tools,
            permissions=self._permissions,
            workspace=self._workspace,
            system_prompt=self._system_prompt,
            max_steps=self._max_steps,
            compile_sandbox=self._compile_sandbox,
            finalize_sandbox=self._finalize_sandbox,
            retry_stream=self._retry_stream,
            artifacts=self._artifacts,
            compact_limit_tokens=self._compact_limit_tokens,
            measure_tokens=self._measure_tokens,
            summary_max_output_tokens=self._summary_max_output_tokens,
            summary_prompt_max_chars=self._summary_prompt_max_chars,
            hooks=self._hooks,
            skills_catalog=self._skills_catalog,
        )
        try:
            with thread_execution_lock(self._workspace, self._thread_id):
                return await run_turn(deps, TurnState(turn_id, self._thread_id), resume_calls=resume_calls)
        except ThreadBusy as exc:
            raise ConversationBusy(
                f"thread {self._thread_id!r} is executing elsewhere; try again later"
            ) from exc
        finally:
            # R6-4:once 授权是调用域——本执行批结束即清除,绝不下渗后续批。
            if self._permissions.once_rules:
                self._permissions = replace(self._permissions, once_rules=())

    def _model_id(self) -> str:
        """turn binding 的 model 标识:经公开 build_request 探针取 WireRequest.model
        (纯构建零副作用;探不到/为空 → "unknown",不触网不耗剧本)。"""
        try:
            model = self._adapter.build_request((), (), ModelSettings()).model
        except Exception:  # noqa: BLE001 — binding 标注尽力而为,失败绝不阻断提交
            return "unknown"
        return model or "unknown"
