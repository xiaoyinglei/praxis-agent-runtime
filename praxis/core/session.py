"""Praxis Session:thread 生命周期、人工确认续跑协议与 steering(设计 §4.1/§5.2/§5.8)。

Session 是 turn 循环之上的会话门面:它不执行任何东西,只负责"什么时候、以什么
身份、拿不拿锁"地调用 run_turn,并兑现 §5.8 的三条硬语义——

1. **执行锁 = 执行者身份**:submit/resume 在 run_turn 外围持 thread 执行锁,
   锁被占(ThreadBusy)一律换成公开的 ``ConversationBusy``;steer/approval/
   reconcile/confirm 只是 append item,永不取锁(§5.8:它们不是执行者)。
2. **人工确认守卫**(§5.2:interrupted 只能由人重启):INTERRUPTED turn 必须
   ``confirm=True``,否则 ``ResumeNotConfirmed``;PAUSED turn 必须给 approval
   (审批)或 reconcile(结果对账)之一;RUNNING 与终态 turn 不可 resume。
3. **classify_pending 决策矩阵**(§5.9,recovery 提供分类,本模块提供动作):
   未 started 的待决调用写 not_executed 结果(它确实没跑);started+幂等自动
   重跑(经唯一管道);started+非幂等写 outcome_unknown 并停在 PAUSED 等人对账
   ——续跑的代价不对称,绝不自动重放未知副作用。

resume 的处理序(与代码一一对应):守卫 → transition(RUNNING,真实 reason) →
reconcile 写结果(先) → approval 写决策(摘要从 approval_request item 取,绑定
digest)→ classify_pending 分类(有 matching decision 的调用进重放队列——审批
重入靠管道内的 digest 绑定裁决,其余才走中断矩阵)→ 若有 outcome_unknown 则
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

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal
from uuid import uuid4

from praxis.core.events import EventBus, Subscription
from praxis.core.locks import ThreadBusy, thread_execution_lock
from praxis.core.recovery import classify_pending
from praxis.core.statemachine import TurnStatus, transition
from praxis.core.turn import ApprovalSummary, TurnDeps, TurnResult, TurnState, run_turn
from praxis.protocol.items import Item, ItemKind
from praxis.protocol.json import JsonValue
from praxis.protocol.models import ModelAdapter, ModelSettings, ModelToolCall
from praxis.protocol.permission import PermissionContext
from praxis.protocol.tools import ResultStatus, Tool
from praxis.store.log import ItemEntry, Store
from praxis.store.turns import Turn, create_turn, get_turn

DEFAULT_SYSTEM_PROMPT_TEMPLATE = (
    "You are Praxis, a workspace coding agent. Workspace: {workspace}. "
    "Use tools to inspect and modify files. Answer concisely."
)

_NOT_EXECUTED_TEXT = "[not executed: turn interrupted before dispatch]"
_OUTCOME_UNKNOWN_TEXT = "[outcome unknown: process died during execution]"

_APPROVAL_VERDICTS = frozenset({"allow", "deny"})
_RECONCILE_OUTCOMES = frozenset({"ok", "failed"})


class ConversationBusy(RuntimeError):  # noqa: N818 — 设计 §5.8/plan 钉死的名字
    """该 thread 的执行锁被另一执行者持有:submit/resume 拒绝排队(§5.8)。"""


class ResumeNotConfirmed(RuntimeError):  # noqa: N818 — 设计 §5.2 守卫钉死的名字
    """interrupted turn 未经人工确认(confirm=True)不得续跑(§5.2 守卫)。"""


# ---------------------------------------------------------------------------
# item 读取助手(全部面向"本 turn 的完整 item 列表",纯函数)
# ---------------------------------------------------------------------------


def _call_names(items: Sequence[Item]) -> dict[str, str]:
    """call_id → 工具名:从最后出现的 model_response/tool_call/tool_result 取。"""
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
    """某 kind 下 call_id → 最后一条 payload(同 call_id 多次请求/决策取最新)。"""
    latest: dict[str, Mapping[str, JsonValue]] = {}
    for item in items:
        if item.kind != kind:
            continue
        call_id = item.payload.get("call_id")
        if isinstance(call_id, str):
            latest[call_id] = item.payload
    return latest


def _pending_approvals(items: Sequence[Item]) -> tuple[ApprovalSummary, ...]:
    """本 turn 内没有 matching approval_decision 的 approval_request(镜像 turn 层口径)。"""
    decided = set(_latest_payloads(items, ItemKind.APPROVAL_DECISION))
    pending: dict[str, ApprovalSummary] = {}
    for call_id, payload in _latest_payloads(items, ItemKind.APPROVAL_REQUEST).items():
        if call_id in decided:
            continue
        tool_name = payload.get("tool_name")
        digest = payload.get("args_digest")
        if not isinstance(tool_name, str) or not isinstance(digest, str):
            continue
        effects: tuple[str, ...] = ()
        raw_effects = payload.get("effects")
        if isinstance(raw_effects, (list, tuple)):
            effects = tuple(e for e in raw_effects if isinstance(e, str))
        pending[call_id] = ApprovalSummary(
            call_id=call_id, tool_name=tool_name, args_digest=digest, effects=effects
        )
    return tuple(pending.values())


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
    ) -> None:
        self._store = store
        self._bus = bus
        self._workspace = Path(workspace)
        self._adapter = adapter
        self._tools = dict(tools)
        self._permissions = permissions
        self._thread_id = thread_id
        self._max_steps = max_steps
        self._system_prompt = (
            system_prompt
            if system_prompt is not None
            else DEFAULT_SYSTEM_PROMPT_TEMPLATE.format(workspace=self._workspace)
        )

    # -- 身份与生命周期 -------------------------------------------------------

    @property
    def thread_id(self) -> str:
        return self._thread_id

    def subscribe(self) -> Subscription:
        """订阅本会话的事件总线(UI 提示通道,§5.6;事件非事实,item 日志才是)。"""
        return self._bus.subscribe()

    def close(self) -> None:
        """关闭 store(写者线程收尾);EventBus 无持久资源,不另设关闭。"""
        self._store.close()

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
    ) -> TurnResult:
        """续跑 interrupted/paused turn——每一步都以 item 为事实,决策全部显式落库。"""
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
            if approval is None and reconcile is None:
                raise ValueError("paused turn needs approval or reconcile")
        elif status is TurnStatus.RUNNING:
            raise ValueError(f"turn {turn_id!r} is still running; not resumable")
        else:
            raise ValueError(f"turn {turn_id!r} is {status.value}; terminal turns are not resumable")

        transition(
            self._store,
            self._bus,
            turn_id,
            TurnStatus.RUNNING,
            reason=self._resume_reason(status, approval, reconcile),
        )
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
                pending_approvals=_pending_approvals(items),
                unknown_calls=tuple(unknown_ids),
            )
        return await self._execute(turn_id, resume_calls)

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
        names = _call_names(items)
        entries: list[ItemEntry] = []
        for call_id, outcome in reconcile.items():
            if outcome not in _RECONCILE_OUTCOMES:
                raise ValueError(f"reconcile outcome must be ok/failed, got {outcome!r}")
            tool_name = names.get(call_id)
            if tool_name is None:
                raise ValueError(f"reconcile target {call_id!r} not found in turn {turn_id!r} items")
            status = ResultStatus.OK if outcome == "ok" else ResultStatus.ERROR
            text = f"reconciled by user: {outcome}"
            entries.append(_text_result_entry(call_id, tool_name, status, text, {"reconciled": True}))
        self._store.append_items(self._thread_id, turn_id, entries)

    def _write_approval_decisions(
        self, turn_id: str, approval: Mapping[str, Literal["allow", "deny"]]
    ) -> None:
        items = self._store.read_turn_items(turn_id)
        requests = _latest_payloads(items, ItemKind.APPROVAL_REQUEST)
        entries: list[ItemEntry] = []
        for call_id, verdict in approval.items():
            if verdict not in _APPROVAL_VERDICTS:
                raise ValueError(f"approval verdict must be allow/deny, got {verdict!r}")
            request = requests.get(call_id)
            if request is None:
                raise ValueError(f"no approval_request for call {call_id!r} in turn {turn_id!r}")
            payload: dict[str, JsonValue] = {
                "call_id": call_id,
                "verdict": verdict,
                "args_digest": request.get("args_digest"),
                "destination": "once",
            }
            entries.append(ItemEntry(kind=ItemKind.APPROVAL_DECISION, payload=payload))
        self._store.append_items(self._thread_id, turn_id, entries)

    def _classify_and_prepare(self, turn_id: str) -> tuple[list[ModelToolCall], list[str]]:
        """classify_pending 决策矩阵 → (重放队列, outcome_unknown 清单)。

        matching approval_decision(allow/deny)的待决调用直接进重放队列——审批
        重入由管道内的 digest 绑定裁决落地(Task 12/13 交接语义);其余才按中断
        矩阵写 not_executed / 自动重放幂等 / 标记 outcome_unknown。
        """
        items = self._store.read_turn_items(turn_id)
        decided = set(_latest_payloads(items, ItemKind.APPROVAL_DECISION))
        resume_calls: list[ModelToolCall] = []
        unknown_ids: list[str] = []
        entries: list[ItemEntry] = []
        for pending in classify_pending(items, self._tools):
            call = pending.call
            if call.call_id in decided:
                resume_calls.append(call)
            elif not pending.started:
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
            pending_approvals=_pending_approvals(items),
            unknown_calls=_unresolved_unknown_calls(items),
        )

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
        )
        try:
            with thread_execution_lock(self._workspace, self._thread_id):
                return await run_turn(deps, TurnState(turn_id, self._thread_id), resume_calls=resume_calls)
        except ThreadBusy as exc:
            raise ConversationBusy(
                f"thread {self._thread_id!r} is executing elsewhere; try again later"
            ) from exc

    def _model_id(self) -> str:
        """turn binding 的 model 标识:经公开 build_request 探针取 WireRequest.model
        (纯构建零副作用;探不到/为空 → "unknown",不触网不耗剧本)。"""
        try:
            model = self._adapter.build_request((), (), ModelSettings()).model
        except Exception:  # noqa: BLE001 — binding 标注尽力而为,失败绝不阻断提交
            return "unknown"
        return model or "unknown"
