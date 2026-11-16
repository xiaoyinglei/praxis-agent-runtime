"""预算树:事实折叠、子份额占用与 admission 判据(设计 §5.5;阶段4 T2)。

纪律(唯一账本):预算树不是第二本账,而是对既有 append-only 事实的**纯函数折叠**:
- tokens/cost 来自 ``model_response.payload.usage`` 与 ``summary.payload.usage``;
- steps 是声明轮计数(见 ``fold_turn_facts`` docstring,控制器裁决 R-PF-4);
- web/browser 请求数来自 ``tool_result`` 与未闭合的 web ``tool_call``;
- 父 ``budget_extension`` 增量折叠进 limits;
- 子 ``child_allocation`` 按 ``child_reader(child_thread_id)`` 读子线程事实,子根
  turn 的终态 turn_status 决定份额是否归还——**settle 是推导不是写**(无 settle
  item,守恒由"事实只增"保证)。

模型调用不做预扣:超额诚实暴露在 exposure,下一次 admission 拒绝;唯一的 reserve
是子分配 grant 的未用部分。单价只在 ``CostMicros`` 注入(§5.15):cost 限额生效而
单价缺失 ⇒ dispatch 前 fail-closed 拒绝(旧 ``PricingUnavailableError`` 的 praxis 形),
绝不按 0 计。

core 纯度:本模块零 adapter 依赖;``store`` 只在 ``build_tree``/``BudgetTracker``
经参数注入读取。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal

from praxis.protocol.budget import Limits, limits_from_payload
from praxis.protocol.items import Item, ItemKind
from praxis.protocol.json import JsonValue
from praxis.protocol.models import CostMicros, Usage
from praxis.store.log import Store

# 子分配的生命周期标签。终态只表达"份额结算完毕",失败原因仍看子 turn 的
# turn_status.reason(本模块不复制该信息)。
type ChildState = Literal["active", "done", "cancelled", "interrupted"]

_DIMENSIONS: Final[tuple[str, ...]] = (
    "tokens",
    "cost_micros",
    "steps",
    "web_requests",
    "browser_requests",
)

# 阶段3 的 web 工具族(旧 agent_runtime/tools/builtins/web.py:41 WEB_TOOL_NAMES,
# 禁止 import)。T15 的 dispatch gate 复用同一集合。
WEB_TOOL_NAMES: Final[frozenset[str]] = frozenset({"web_fetch", "web_search"})

# 一条 web tool_result 证明请求已完成的 status 集合(T2 brief 原文)。denied /
# paused / not_executed 都发生在 dispatch 之前,不占请求配额。
_BILLABLE_WEB_STATUSES: Final[frozenset[str]] = frozenset(
    {"ok", "error", "timeout", "cancelled"}
)

_MICROS_PER_MILLION: Final[int] = 1_000_000
_PROTECTED_TAIL_CAP: Final[int] = 8_192

# 子根 turn 的终态 → 份额结算标签。failed 与 completed 都表示"不再消耗",
# 结算语义相同;cancelled 保留独立标签供调用方区分(设计 §5.13 级联取消)。
_CHILD_TERMINAL_STATES: Final[Mapping[str, ChildState]] = {
    "completed": "done",
    "failed": "done",
    "cancelled": "cancelled",
}


# ---------------------------------------------------------------------------
# 数据形状
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class UsageFacts:
    """五维已用事实;每维都是非负计数(超额由 exposure 诚实暴露)。"""

    tokens: int = 0
    cost_micros: int = 0
    steps: int = 0
    web_requests: int = 0
    browser_requests: int = 0

    def __add__(self, other: UsageFacts) -> UsageFacts:
        return UsageFacts(
            tokens=self.tokens + other.tokens,
            cost_micros=self.cost_micros + other.cost_micros,
            steps=self.steps + other.steps,
            web_requests=self.web_requests + other.web_requests,
            browser_requests=self.browser_requests + other.browser_requests,
        )


@dataclass(slots=True)
class ChildAllocation:
    """一条 ``child_allocation`` 事实的折叠结果。

    ``grant`` 是 spawn 时按份额公式定下的上限(写端事实);``used`` 不是 payload
    字段,而是本模块按 child_reader 读到的子树事实算出的**推导量**。state 由子根
    turn 的最后 turn_status 决定;读不到子事实时保守 ``active``(份额全额占用)。
    """

    allocation_id: str
    child_thread_id: str
    child_root_turn_id: str
    grant: Limits
    source: str
    state: ChildState
    used: UsageFacts = field(default_factory=UsageFacts)


@dataclass
class BudgetNode:
    """一个 turn 的预算节点。

    ``used`` 含本人事实与全部子分配子树的已用事实(守恒:``exposure = used +
    reserved``);``reserved`` 只含 state=active 子分配的未用份额。``unit_costs``
    是 admission 的 fail-closed 判据所需元数据,不是事实账的一部分。
    """

    turn_id: str
    limits: Limits
    used: UsageFacts
    children: dict[str, ChildAllocation] = field(default_factory=dict)
    under_accounted: bool = False
    unit_costs: CostMicros | None = None

    @property
    def reserved(self) -> UsageFacts:
        """Σ active 子分配的 grant−已用(逐维下钳 ≥0)。"""
        return _reserved(self.children)[0]


@dataclass(frozen=True, slots=True)
class Admission:
    """一次 dispatch 的准入结论;reason 只在拒绝时非空。"""

    allowed: bool
    reason: str = ""


@dataclass(frozen=True, slots=True)
class _AllocationSpec:
    """通过 fail-open 校验的 child_allocation payload 读形。"""

    allocation_id: str
    child_thread_id: str
    child_root_turn_id: str
    grant: Limits
    source: str


# ---------------------------------------------------------------------------
# 单价:旧 agent_runtime/budget/pricing.py ``_ceil_cost`` 语义移植
# ---------------------------------------------------------------------------


def estimate_cost(usage: Usage, cost: CostMicros) -> int:
    """按每百万 token 单价算 micros,整个和**向上取整一次**(旧 ``_ceil_cost``)。

    - ``Usage.input_tokens`` 含 cached 子集:未缓存部分按 input 单价、cached 部分
      按 cache_read 单价;cache_read 未配置(0)时回落 input 单价(旧 pricing 的
      ``None → input_rate`` 语义,CostMicros 用 0 表示未配置);
    - reasoning 是 output 的子集(models.Usage 契约),不重复计价;
    - ``Usage`` 没有 cache-write 计数字段,该分量天然为 0。
    """
    cached = max(usage.cached_input_tokens, 0)
    uncached = max(usage.input_tokens - cached, 0)
    cache_rate = cost.cache_read_per_1m or cost.input_per_1m
    numerator = (
        uncached * cost.input_per_1m
        + cached * cache_rate
        + usage.output_tokens * cost.output_per_1m
    )
    if numerator <= 0:
        return 0
    return (numerator + _MICROS_PER_MILLION - 1) // _MICROS_PER_MILLION


# ---------------------------------------------------------------------------
# 事实折叠
# ---------------------------------------------------------------------------


def _nonneg_int(value: JsonValue) -> int:
    """JSON 数字 → 非负 int;bool/字符串/负数一律按 0(fail-open)。"""
    if isinstance(value, bool) or not isinstance(value, int):
        return 0
    return value if value > 0 else 0


def _usage_component(payload: Mapping[str, JsonValue]) -> tuple[int, int, int] | None:
    """payload.usage → (input, output, cached);usage 缺失/畸形返回 None。"""
    raw = payload.get("usage")
    if not isinstance(raw, Mapping):
        return None
    return (
        _nonneg_int(raw.get("input_tokens")),
        _nonneg_int(raw.get("output_tokens")),
        _nonneg_int(raw.get("cached_input_tokens")),
    )


def fold_turn_facts(items: Sequence[Item], *, unit_costs: CostMicros | None) -> UsageFacts:
    """把一个 turn 的 items 折叠成五维已用事实。

    **steps 口径(控制器裁决 R-PF-4)**:``tool_calls`` 非空的 ``model_response``
    条数(声明轮口径)。旧 ``core/turn.py`` 在"一轮请求+全部工具完成后"
    ``state.steps += 1``;resume 重放轮没有新的 ``model_response`` 事实,由
    ``run_turn`` 进程内自增补计——两者相加与旧"每次运行总自增量"逐值一致(T3
    消费面)。

    **web 保守面**:web ``tool_call`` 已落库(= reserve-before-IO:dispatch 前
    写入),但全 turn 找不到同 call_id 的 ``tool_result`` ⇒ dispatch 后中断,
    保守记 1(旧 ``agent_runtime/harness/budget_ops.py:284-322`` reserve 语义的
    parity;禁止 import)。有结果时按结果 status 计费,不重复计。

    tokenization ``source="estimated"`` 照单计入:事实来自 provider 的估算帧,
    来源标签不改记账。``unit_costs`` 为 None 时 cost 维只能是 0(有 cost 限额的
    turn 由 ``admit_dispatch`` fail-closed 挡在 dispatch 之前)。
    """
    # 第一遍:收集所有 tool_result 的 call_id(任意 status),供 web tool_call 配对。
    result_call_ids: set[str] = set()
    for item in items:
        if item.kind != ItemKind.TOOL_RESULT:
            continue
        call_id = item.payload.get("call_id")
        if isinstance(call_id, str):
            result_call_ids.add(call_id)

    tokens = 0
    cost_micros = 0
    steps = 0
    web_requests = 0
    browser_requests = 0
    counted_web_calls: set[str] = set()
    counted_browser_calls: set[str] = set()
    unpaired_web_calls: set[str] = set()

    for item in items:
        if item.kind in (ItemKind.MODEL_RESPONSE, ItemKind.SUMMARY):
            component = _usage_component(item.payload)
            if component is not None:
                input_tokens, output_tokens, cached_input_tokens = component
                # input/output 已各自包含 cache/reasoning 子集,总量不重复相加。
                tokens += input_tokens + output_tokens
                if unit_costs is not None:
                    cost_micros += estimate_cost(
                        Usage(
                            input_tokens=input_tokens,
                            output_tokens=output_tokens,
                            cached_input_tokens=cached_input_tokens,
                        ),
                        unit_costs,
                    )
            if item.kind == ItemKind.MODEL_RESPONSE:
                raw_calls = item.payload.get("tool_calls")
                if isinstance(raw_calls, (list, tuple)) and len(raw_calls) > 0:
                    steps += 1
            continue

        if item.kind == ItemKind.TOOL_RESULT:
            tool_name = item.payload.get("tool_name")
            status = item.payload.get("status")
            if (
                not isinstance(tool_name, str)
                or tool_name not in WEB_TOOL_NAMES
                or not isinstance(status, str)
                or status not in _BILLABLE_WEB_STATUSES
            ):
                continue
            call_id = item.payload.get("call_id")
            if isinstance(call_id, str):
                if call_id in counted_web_calls:
                    continue
                counted_web_calls.add(call_id)
            web_requests += 1
            metadata = item.payload.get("metadata")
            if isinstance(metadata, Mapping) and metadata.get("browser_render") is True:
                if isinstance(call_id, str):
                    if call_id not in counted_browser_calls:
                        counted_browser_calls.add(call_id)
                        browser_requests += 1
                else:
                    browser_requests += 1
            continue

        if item.kind == ItemKind.TOOL_CALL:
            tool_name = item.payload.get("tool_name")
            if not isinstance(tool_name, str) or tool_name not in WEB_TOOL_NAMES:
                continue
            call_id = item.payload.get("call_id")
            if not isinstance(call_id, str):
                # 无 call_id 的 web 调用无法证明有结果:保守计一次。
                web_requests += 1
                continue
            if (
                call_id in result_call_ids
                or call_id in counted_web_calls
                or call_id in unpaired_web_calls
            ):
                continue
            unpaired_web_calls.add(call_id)
            web_requests += 1

    return UsageFacts(
        tokens=tokens,
        cost_micros=cost_micros,
        steps=steps,
        web_requests=web_requests,
        browser_requests=browser_requests,
    )


# ---------------------------------------------------------------------------
# 保护尾 / exposure / remaining / admission
# ---------------------------------------------------------------------------


def protected_tail(limit: int | None) -> int:
    """收尾/恢复专用的 token 保护尾。

    逐字移植旧 ``agent_runtime/budget/accounting.py:187-195``
    ``protected_tokens_for_limit``(禁止 import):
    ``None → 0``;``limit ≤ 1 → 0``;否则
    ``min(8192, max(1, limit // 10), limit - 1)``。普通模型调用不得吃保护尾,
    compaction 续作可以(旧 ``ensure_token_headroom(allow_protected=...)``)。
    """
    if limit is None:
        return 0
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("token limit must be a non-negative integer or None")
    if limit <= 1:
        return 0
    return min(_PROTECTED_TAIL_CAP, max(1, limit // 10), limit - 1)


def exposure(node: BudgetNode) -> UsageFacts:
    """已用 + active 子分配未用份额;唯一预算占用面。"""
    return node.used + node.reserved


def _remaining_dim(limit: int | None, used: int) -> int | None:
    if limit is None:
        return None
    return max(limit - used, 0)


def remaining(node: BudgetNode) -> Limits:
    """各维剩余;无限维保持 None。超额时下钳 0(exposure 仍诚实显示超额)。"""
    used = exposure(node)
    return Limits(
        tokens=_remaining_dim(node.limits.tokens, used.tokens),
        cost_micros=_remaining_dim(node.limits.cost_micros, used.cost_micros),
        steps=_remaining_dim(node.limits.steps, used.steps),
        web_requests=_remaining_dim(node.limits.web_requests, used.web_requests),
        browser_requests=_remaining_dim(node.limits.browser_requests, used.browser_requests),
    )


def normal_token_remaining(node: BudgetNode) -> int | None:
    """普通调用可用的 token 余量(扣除保护尾);无限维返回 None。

    旧 ``accounting.py:205-212 normal_token_remaining`` 语义。
    """
    token_remaining = remaining(node).tokens
    if token_remaining is None:
        return None
    return max(token_remaining - protected_tail(node.limits.tokens), 0)


def admit_dispatch(node: BudgetNode, *, needs_compaction: bool) -> Admission:
    """模型 dispatch 前的唯一准入判据。

    顺序:
    1. cost 限额生效而单价缺失/全零 ⇒ ``pricing_unavailable``(fail-closed,
       绝不按 0 计;旧 ``PricingUnavailableError`` 的 praxis 形);
    2. tokens 维:普通调用有效上限 = ``limit − protected_tail``,compaction 可
       使用完整 limit;exposure ≥ 有效上限即耗尽;
    3. cost/steps 维:exposure ≥ limit 即耗尽;
    4. web/browser 维不在这里判(工具 dispatch 由 ``admit_web_request`` gate)。

    任何耗尽都返回 ``budget_exhausted``;成功 reason 为空串。
    """
    cost_limit = node.limits.cost_micros
    if cost_limit is not None and (node.unit_costs is None or node.unit_costs.is_zero):
        return Admission(False, "pricing_unavailable")

    used = exposure(node)
    token_limit = node.limits.tokens
    if token_limit is not None:
        effective_limit = (
            token_limit
            if needs_compaction
            else max(token_limit - protected_tail(token_limit), 0)
        )
        if used.tokens >= effective_limit:
            return Admission(False, "budget_exhausted")
    if cost_limit is not None and used.cost_micros >= cost_limit:
        return Admission(False, "budget_exhausted")
    if node.limits.steps is not None and used.steps >= node.limits.steps:
        return Admission(False, "budget_exhausted")
    return Admission(True, "")


# ---------------------------------------------------------------------------
# 预算树构建
# ---------------------------------------------------------------------------


def _reserved(children: Mapping[str, ChildAllocation]) -> tuple[UsageFacts, bool]:
    """active 子分配的未用份额(逐维下钳)与 under_accounted 旗标。"""
    reserved: dict[str, int] = {dim: 0 for dim in _DIMENSIONS}
    under_accounted = False
    for child in children.values():
        if child.state != "active":
            continue
        for dim in _DIMENSIONS:
            grant = getattr(child.grant, dim)
            if grant is None:
                continue
            unused = grant - getattr(child.used, dim)
            if unused < 0:
                under_accounted = True
                unused = 0
            reserved[dim] += unused
    return (
        UsageFacts(
            tokens=reserved["tokens"],
            cost_micros=reserved["cost_micros"],
            steps=reserved["steps"],
            web_requests=reserved["web_requests"],
            browser_requests=reserved["browser_requests"],
        ),
        under_accounted,
    )


def _fold_extensions(base: Limits, items: Sequence[Item]) -> Limits:
    """把 ``budget_extension`` 增量累加到 limits。

    规则(只增不减):
    - 多次扩展逐维累加;
    - 无限维(None)没有可增的余量,保持 None(扩展不能把无限变有界);
    - 负增量/畸形 delta 忽略(fail-open:items 是开放事实集合)。
    """
    values: dict[str, int | None] = {dim: getattr(base, dim) for dim in _DIMENSIONS}
    for item in sorted(items, key=lambda entry: entry.seq):
        if item.kind != ItemKind.BUDGET_EXTENSION:
            continue
        raw_delta = item.payload.get("limits_delta")
        if not isinstance(raw_delta, Mapping):
            continue
        try:
            delta = limits_from_payload(raw_delta)
        except (TypeError, ValueError):
            continue
        for dim in _DIMENSIONS:
            change = getattr(delta, dim)
            current = values[dim]
            if change is None or change <= 0 or current is None:
                continue
            values[dim] = current + change
    return Limits(
        tokens=values["tokens"],
        cost_micros=values["cost_micros"],
        steps=values["steps"],
        web_requests=values["web_requests"],
        browser_requests=values["browser_requests"],
    )


def _parse_allocation(item: Item) -> _AllocationSpec | None:
    """child_allocation payload → 读形;畸形 payload 跳过(fail-open 读端)。"""
    payload = item.payload
    allocation_id = payload.get("allocation_id")
    child_thread_id = payload.get("child_thread_id")
    child_root_turn_id = payload.get("child_root_turn_id")
    raw_grant = payload.get("grant")
    if (
        not isinstance(allocation_id, str)
        or not allocation_id
        or not isinstance(child_thread_id, str)
        or not child_thread_id
        or not isinstance(child_root_turn_id, str)
        or not child_root_turn_id
        or not isinstance(raw_grant, Mapping)
    ):
        return None
    try:
        grant = limits_from_payload(raw_grant)
    except (TypeError, ValueError):
        return None
    source = payload.get("source")
    return _AllocationSpec(
        allocation_id=allocation_id,
        child_thread_id=child_thread_id,
        child_root_turn_id=child_root_turn_id,
        grant=grant,
        source=source if isinstance(source, str) else "unknown",
    )


def _child_state(items: Sequence[Item]) -> ChildState:
    """子根 turn 最后一条 turn_status 决定状态;无状态/非终态 ⇒ active。

    控制器裁决:终态 {completed, failed, cancelled} 结算未用份额;
    {paused, interrupted, running} 保守 active(份额继续占用——被打断的子
    可能被 recovery 续跑,不能提前归还预算)。ChildState.interrupted 留给
    T4/T5 的显式级联标签,本折叠不产出。
    """
    status_items = [item for item in items if item.kind == ItemKind.TURN_STATUS]
    if not status_items:
        return "active"
    last = max(status_items, key=lambda entry: entry.seq)
    to = last.payload.get("to")
    if not isinstance(to, str):
        return "active"
    return _CHILD_TERMINAL_STATES.get(to, "active")


def _fold_children(
    items: Sequence[Item],
    *,
    unit_costs: CostMicros | None,
    reader: Callable[[str], Sequence[Item]],
    visited: frozenset[tuple[str, str]],
) -> dict[str, ChildAllocation]:
    """折叠本层 child_allocation;子事实经注入 reader 读取。"""
    children: dict[str, ChildAllocation] = {}
    for item in sorted(items, key=lambda entry: entry.seq):
        if item.kind != ItemKind.CHILD_ALLOCATION:
            continue
        spec = _parse_allocation(item)
        if spec is None:
            continue
        pair = (spec.child_thread_id, spec.child_root_turn_id)
        if pair in visited:
            # 病态环(如分配指回祖先):保守 active + 全额占用,绝不递归失控。
            used = UsageFacts()
            state: ChildState = "active"
        else:
            try:
                raw_child_items = reader(spec.child_thread_id)
            except Exception:  # noqa: BLE001 — 读不到子事实 ⇒ 保守全额占用
                raw_child_items = ()
            child_items = [
                entry for entry in raw_child_items if entry.turn_id == spec.child_root_turn_id
            ]
            used, _grandchildren, _grand_under = _fold_subtree(
                child_items,
                unit_costs=unit_costs,
                reader=reader,
                visited=visited | {pair},
            )
            state = _child_state(child_items)
        children[spec.allocation_id] = ChildAllocation(
            allocation_id=spec.allocation_id,
            child_thread_id=spec.child_thread_id,
            child_root_turn_id=spec.child_root_turn_id,
            grant=spec.grant,
            source=spec.source,
            state=state,
            used=used,
        )
    return children


def _fold_subtree(
    items: Sequence[Item],
    *,
    unit_costs: CostMicros | None,
    reader: Callable[[str], Sequence[Item]],
    visited: frozenset[tuple[str, str]],
) -> tuple[UsageFacts, dict[str, ChildAllocation], bool]:
    """一个 turn 的子树事实:本人 + 全部后代已用;reserved 只看本层 active 子。"""
    own = fold_turn_facts(items, unit_costs=unit_costs)
    children = _fold_children(items, unit_costs=unit_costs, reader=reader, visited=visited)
    total = own
    for child in children.values():
        total = total + child.used
    _reserved_facts, under_accounted = _reserved(children)
    return total, children, under_accounted


def build_tree(
    store: Store,
    *,
    thread_id: str,
    turn_id: str,
    root_limits: Limits,
    unit_costs: CostMicros | None,
    child_reader: Callable[[str], Sequence[Item]] | None = None,
) -> BudgetNode:
    """从 append-only 事实重建一个 turn 的预算节点。

    - 父 items 只取 ``item.turn_id == turn_id``(预算按 turn 计);
    - limits = root_limits + 本 turn 的 ``budget_extension`` 增量;
    - 子分配按 ``child_allocation`` 事实读子线程(单测注入 ``child_reader``;
      生产默认 ``store.read_items``),子事实只取 ``child_root_turn_id`` 的 items;
    - 子终态 ⇒ 未用份额在 exposure 中推导式归还(不写 settle item);
    - 子线程不存在/为空/reader 异常 ⇒ 保守 active 全额占用。
    """
    reader: Callable[[str], Sequence[Item]] = (
        child_reader if child_reader is not None else store.read_items
    )
    parent_items = [item for item in store.read_items(thread_id) if item.turn_id == turn_id]
    limits = _fold_extensions(root_limits, parent_items)
    used, children, under_accounted = _fold_subtree(
        parent_items,
        unit_costs=unit_costs,
        reader=reader,
        visited=frozenset({(thread_id, turn_id)}),
    )
    return BudgetNode(
        turn_id=turn_id,
        limits=limits,
        used=used,
        children=children,
        under_accounted=under_accounted,
        unit_costs=unit_costs,
    )


# ---------------------------------------------------------------------------
# BudgetTracker:run_turn/pipeline 的唯一持有面
# ---------------------------------------------------------------------------


class BudgetTracker:
    """run_turn/pipeline 持有的预算状态;同模块唯一定义。

    不变量(docstring 契约,单执行者锁内合法):
    - ``refresh()`` 从 store 重建折叠,是预算事实的唯一刷新口(step 边界);
    - ``admit_web_request`` **先 refresh 再判**:已落库结果永远可见,内存 in-flight
      只含"已 admit 但结果尚未落库"的请求,二者不相交(不重复计;
      "先 refresh 后 note"纪律);
    - ``note_result`` 必须在对应 ``tool_result`` item 落库**之后**调用,它释放
      in-flight;若在落库前调用,该请求会在下一次 refresh 前被折叠与 in-flight
      双计——这是调用顺序硬要求(T15 管道序:append result → note_result);
    - ``admit_web_request``/``note_result`` 的 ``browser`` 标志必须成对一致
      (同一请求同一面),否则 browser 维 in-flight 会漂移。
    """

    def __init__(
        self,
        *,
        store: Store,
        thread_id: str,
        turn_id: str,
        root_limits: Limits,
        unit_costs: CostMicros | None,
        child_reader: Callable[[str], Sequence[Item]] | None = None,
    ) -> None:
        self._store = store
        self._thread_id = thread_id
        self._turn_id = turn_id
        self._root_limits = root_limits
        self._unit_costs = unit_costs
        self._child_reader = child_reader
        self._node: BudgetNode | None = None
        self._inflight_web = 0
        self._inflight_browser = 0

    def refresh(self) -> BudgetNode:
        """步边界重建折叠;in-flight 不在此清空(它们尚未落库)。"""
        self._node = build_tree(
            self._store,
            thread_id=self._thread_id,
            turn_id=self._turn_id,
            root_limits=self._root_limits,
            unit_costs=self._unit_costs,
            child_reader=self._child_reader,
        )
        return self._node

    @property
    def node(self) -> BudgetNode:
        """最近一次折叠;从未 refresh 时惰性重建(不改变事实,只是读)。"""
        if self._node is None:
            return self.refresh()
        return self._node

    def admit_dispatch(self, *, needs_compaction: bool) -> Admission:
        return admit_dispatch(self.node, needs_compaction=needs_compaction)

    def admit_web_request(self, name: str, *, browser: bool) -> Admission:
        """web 工具 dispatch 前 gate;非 web 工具名不受本维约束。

        browser=True 的请求同时消耗 web_requests 与 browser_requests 两维
        (浏览器抓取仍是 web 请求)。
        """
        if name not in WEB_TOOL_NAMES:
            return Admission(True, "")
        node = self.refresh()
        web_limit = node.limits.web_requests
        used_web = node.used.web_requests + self._inflight_web
        if web_limit is not None and used_web >= web_limit:
            return Admission(False, "budget_exhausted")
        if browser:
            browser_limit = node.limits.browser_requests
            used_browser = node.used.browser_requests + self._inflight_browser
            if browser_limit is not None and used_browser >= browser_limit:
                return Admission(False, "budget_exhausted")
        self._inflight_web += 1
        if browser:
            self._inflight_browser += 1
        return Admission(True, "")

    def note_result(self, name: str, *, browser: bool) -> None:
        """结果落库后释放对应 in-flight(无匹配 admit 时下钳 0,永不负债)。"""
        if name not in WEB_TOOL_NAMES:
            return
        self._inflight_web = max(self._inflight_web - 1, 0)
        if browser:
            self._inflight_browser = max(self._inflight_browser - 1, 0)