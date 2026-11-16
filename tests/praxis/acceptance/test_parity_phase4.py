"""Task 17 Steps 1-2:阶段 4 旧代码行为对照矩阵 + findings 回归指认。

沿用 3A/3B ``acceptance/test_tool_parity{,3b}.py`` 口径:

- 每个数值/错误码/判据都带**旧代码 file:line**出处(相对仓库根的旧
  ``agent_runtime/``,只读参照,绝不 import);阈值以 praxis 的**公开契约**
  钉住——协议常量/纯函数/工具 ``validate_input``/真实 Store/真实 git/真实
  子 Session 行为,不 import 任何模块私有常量。
- **有意差异显式缺席断言而非省略**:每处 divergence 都有一条"不存在面"或
  "新形状"断言 + 一行理由(引用阶段 4 计划「明确简化」表/Q 裁决)。
- findings B2/B4/B14/B23/B29 逐号在 ``test_findings_regression_locations``
  parametrize 可指认(owning 回归测试 AST 存在性)+ 本文件内独立复跑小回归;
  全量 owning 套件由 Step 4 全门禁真跑复确认。

旧代码出处总目(逐节另有行内锚点):
- 份额/settle/reserve:``agent_runtime/harness/budget_ops.py:74-102``
  (B2 病灶 :88 ``granted_tokens = remaining``)、``:170-201``(settle 写
  ``budget_child_settled`` fact)、``:284-322``(reserve_resource_unit);
- 保护尾/普通余量:``agent_runtime/budget/accounting.py:187-195``
  (protected_tokens_for_limit)、``:205-212``(normal_token_remaining);
  单价向上取整:``agent_runtime/budget/pricing.py:159-165``(_ceil_cost);
- 子限额/wait:``agent_runtime/harness/subagents.py:25``(DEFAULT_LIMITS
  4/2/16)、``:26``(TERMINAL)、``:194``(旧 wait 上限 30)、``:199-201``
  (asyncio.wait 过期不取消);
- task 壳/args 上限:``agent_runtime/tools/integrations/subagent.py:52-59``
  (task/context_summary ≤20_000)、``:96``(SubagentPause)、``:153``(name="task");
- memory:``agent_runtime/harness/memory.py:42-45``(常量 32/32_000/2_000/
  24_000)、``:178-183``(_unresolved 查 {"running","unknown"} 漏 dispatched)、
  ``:336``(lease 300)、``:399-415``(job running 行)、``:620/:649``(pending/
  done/needs_reconciliation/budget_exhausted/failed)、``:709``(retry_after
  300)、``:95/:114``(冻结快照/memory_context_revoked);
  ``agent_runtime/harness/memory_identity.py:183-186``(正确集含 dispatched);
  ``agent_runtime/harness/memory_management.py:210``(旧 action 枚举含 forget);
  ``agent_runtime/harness/runtime_store.py:360-361/:375-376``(无 DB 报 false);
- 子工作区:``agent_runtime/harness/child_workspaces.py:184``(diff_limit
  200_000)、``:326-329``(reviewed_through 读满门)、``:202-276``(verification
  receipt 复比);
- web:``agent_runtime/tools/web_limits.py:9-10``(32/64 默认)、
  ``agent_runtime/harness/session.py:1279``(web_request_budget_exceeded);
- 父完成 drain:``agent_runtime/harness/session.py:548/:1512-1513``
  (before_completion=_drain_children)。

有意差异全集(缺席断言,每条一行理由;详见各测试注释):
无 task 同步壳(Q4 不留双实现)/无 SubagentPause 冒泡(Q3 直面)/无 memory
冻结快照+revoke(简化表:每轮重建上下文无漂移面)/无 scope 键(简化表:每
workspace 一库)/非 git 拒 write 子(Q5)/模型调用无 SQLite 预扣(简化表:
执行锁单执行者)/settle 无 item(§5.5 推导)/父完成不 drain(简化表:§5.13
级联列表不含 COMPLETED)/无 reviewed_through 读满门与 verification receipt
(简化表:无旧 UI 面)/web 计数=推导事实+进程内 gate(§5.5 唯一账本)/
delete history action="delete"(T13 注记,旧 forget)/canonical_json 仅
memory 包络口径(args_digest 自有 convention,T10 注记)/空工作区两旗标
报 true(T11 initial_state,旧无 DB 报 false)/wait 上限 60(旧 30,T5/T6
计划钉死)。
"""

from __future__ import annotations

import ast
import asyncio
import dataclasses
import hashlib
import inspect
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest

from praxis.adapters.web.limits import WebLimits
from praxis.adapters.workspaces.git import GitWorktreeProvisioner
from praxis.core import agents as agents_module
from praxis.core import budget as budget_module
from praxis.core import memory as memory_module
from praxis.core.agents import SubagentManager, cascade_for
from praxis.core.budget import (
    WEB_REQUEST_BUDGET_EXCEEDED,
    BudgetTracker,
    admit_dispatch,
    build_tree,
    estimate_cost,
    exposure,
    fold_turn_facts,
    normal_token_remaining,
    protected_tail,
    remaining,
)
from praxis.core.compensation import close_unresolved
from praxis.core.events import EventBus
from praxis.core.memory import (
    JOB_KEY,
    MAX_ENTRY_CHARS,
    MAX_MEMORY_ENTRIES,
    MAX_SOURCE_BYTES,
    MEMORY_LEASE_SECONDS,
    MEMORY_RETRY_SECONDS,
    MEMORY_TOKEN_BUDGET,
    STATE_KEY,
    delete_memory,
    initial_state,
    job_unresolved,
    read_memory_state,
    read_memory_status,
    record_revision,
    remember_memory,
)
from praxis.core.statemachine import TurnStatus, transition
from praxis.interfaces import sdk
from praxis.interfaces.sdk import ScriptedAdapter, open_session
from praxis.protocol import agents as agents_protocol
from praxis.protocol import budget as budget_protocol
from praxis.protocol.agents import AGENT_TOOL_NAMES
from praxis.protocol.budget import (
    MAX_CHILD_FRACTION,
    SUBAGENT_LIMITS_DEFAULT,
    Limits,
    allocation_payload,
    limits_merge_grant,
)
from praxis.protocol.items import ItemKind
from praxis.protocol.json import canonical_json
from praxis.protocol.models import CostMicros, ModelResponse, ModelToolCall, Usage
from praxis.protocol.sandbox import ToolExecContext
from praxis.protocol.tools import (
    CancellationMode,
    InterruptBehavior,
    ResolvedUse,
    ResultStatus,
    Tool,
    ToolContentBlock,
    ToolDefinition,
    ToolEffect,
    ToolResult,
    ToolValidationError,
    args_digest,
)
from praxis.protocol.workspaces import (
    DIFF_PREVIEW_LIMIT_BYTES,
    ApplyResult,
    ChildWorkspaceRequiresGit,
    WorkspaceInspection,
)
from praxis.store.log import ItemEntry, Store
from praxis.store.memory import (
    MEMORY_ENTRY_HISTORY_KEEP,
    MEMORY_JOB_HISTORY_KEEP,
    MEMORY_VERSION_HISTORY_KEEP,
    cas_put_metadata,
    get_metadata,
    prune_job_history,
    prune_versions,
    put_state_enveloped,
    read_state_verified,
)
from praxis.store.turns import create_turn, get_turn
from praxis.tools.builtins.agent_tools import create_agent_tools
from praxis.tools.builtins.child_env import child_env
from praxis.tools.builtins.memory_tools import create_read_memory_tool
from tests.praxis.conftest import make_turn

_TESTS_DIR = Path(__file__).resolve().parents[1]
_ZERO_USAGE = Usage(input_tokens=0, output_tokens=0)


# ---------------------------------------------------------------------------
# 脚手架(与 test_agents_core.py 同口径;真 Session/真 Store/真 git)
# ---------------------------------------------------------------------------


def _resp(
    content: str,
    calls: tuple[tuple[str, str, dict[str, Any]], ...] = (),
    *,
    usage: Usage = _ZERO_USAGE,
) -> ModelResponse:
    return ModelResponse(
        content=content,
        tool_calls=tuple(ModelToolCall(call_id=c, name=n, arguments=a) for c, n, a in calls),
        usage=usage,
        finish_reason="tool_calls" if calls else "stop",
    )


def _toy_tool(
    name: str,
    run: Any,
    *,
    effects: frozenset[ToolEffect] = frozenset(),
) -> Tool:
    return Tool(
        definition=ToolDefinition(name=name, description=f"{name} toy", input_schema={"type": "object"}),
        validate_input=lambda arguments: arguments,
        run=run,
        static_effects=effects,
        resolve_use=lambda arguments: ResolvedUse(effects=effects, targets=()),
        idempotent=True,
        concurrency_safe=True,
        cancellation_mode=CancellationMode.COOPERATIVE,
        interrupt_behavior=InterruptBehavior.CANCEL,
        timeout_seconds=30,
        max_model_output_bytes=64_000,
    )


def _ok(name: str, text: str) -> ToolResult:
    return ToolResult(
        tool_call_id="pending",
        tool_name=name,
        content=(ToolContentBlock(type="text", data={"text": text}),),
        status=ResultStatus.OK,
    )


def _hang_tool(name: str, event: asyncio.Event) -> Tool:
    async def run(arguments: Mapping[str, Any], exec_ctx: ToolExecContext) -> ToolResult:
        await event.wait()
        return _ok(name, f"{name}:released")

    return _toy_tool(name, run)


def _spawn_probe(
    holder: list[SubagentManager],
    results: list[Any],
    *,
    wait_child: bool = True,
) -> Tool:
    """父 turn 内的 spawn 探针(T6 工具面测试替身,同一条 manager 路径)。"""

    async def run(arguments: Mapping[str, Any], exec_ctx: ToolExecContext) -> ToolResult:
        handle = await holder[0].spawn("child task")
        results.append(handle)
        if wait_child:
            results.append(await holder[0].wait(handle.child_thread_id, timeout=5.0))
        return _ok("spawn_probe", handle.child_thread_id)

    return _toy_tool("spawn_probe", run)


def _usage_payload(input_tokens: int, output_tokens: int = 0) -> dict[str, Any]:
    return {"usage": {"input_tokens": input_tokens, "output_tokens": output_tokens}}


def _seed_child(
    store: Store,
    *,
    parent_thread: str = "p",
    parent_turn: str = "pt",
    child_thread: str = "c",
    child_turn: str = "ct",
    grant: Limits = Limits(tokens=500),
    child_usage: int = 0,
    child_status: str | None = None,
) -> None:
    """手工种一棵父→子预算树(公开 Store/turns API;无执行锁=原主已死形态)。"""
    store.ensure_thread(parent_thread)
    store.ensure_thread(child_thread)
    create_turn(store, turn_id=parent_turn, thread_id=parent_thread, binding={})
    create_turn(store, turn_id=child_turn, thread_id=child_thread, binding={})
    store.append_items(
        parent_thread,
        parent_turn,
        [
            ItemEntry(
                ItemKind.CHILD_ALLOCATION,
                allocation_payload(
                    allocation_id="alloc-1",
                    parent_turn_id=parent_turn,
                    child_thread_id=child_thread,
                    child_root_turn_id=child_turn,
                    workspace_mode="read_only",
                    workspace="",
                    grant=grant,
                    requested=None,
                    source="model",
                ),
            )
        ],
    )
    entries: list[ItemEntry] = []
    if child_usage:
        entries.append(ItemEntry(ItemKind.MODEL_RESPONSE, _usage_payload(child_usage)))
    if child_status is not None:
        entries.append(ItemEntry(ItemKind.TURN_STATUS, {"from": "running", "to": child_status, "reason": "test"}))
    if entries:
        store.append_items(child_thread, child_turn, entries)


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(cwd), *args], capture_output=True, check=True, timeout=30
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "praxis-test@example.com")
    _git(root, "config", "user.name", "Praxis Test")
    (root / "a.txt").write_text("base-a\n", encoding="utf-8")
    (root / "b.txt").write_text("base-b\n", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "init")
    return root


@pytest.fixture
def provisioner() -> GitWorktreeProvisioner:
    return GitWorktreeProvisioner(child_env)


# ---------------------------------------------------------------------------
# Step 1 §份额公式(B2)vs budget_ops.py:74-102
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("remaining", [1, 2, 3, 10, 100, 1000, 8192, 100_000])
def test_b2_default_grant_is_half_never_full_remaining(remaining: int) -> None:
    # 旧病灶:budget_ops.py:88 requested_tokens=None 时 granted_tokens=remaining
    # (单子吸干父预算)。praxis 份额公式唯一表达 limits_merge_grant:默认
    # grant = floor(remaining × 0.5),**显式反向断言绝不等于 remaining**。
    for dimension in ("tokens", "cost_micros", "steps"):
        grant = limits_merge_grant(None, Limits(**{dimension: remaining}))
        value = getattr(grant, dimension)
        assert value == remaining // 2
        assert value != remaining, "B2 病灶回归:默认 grant 绝不全额转让父剩余"
        assert value <= remaining * MAX_CHILD_FRACTION


@pytest.mark.parametrize(
    ("requested", "parent_remaining", "expected"),
    [
        # 父无限维 → 透传 requested(None 仍无限;旧 :102 granted=requested parity)。
        (None, None, None),
        (700, None, 700),
        # requested 超份额 → 截断到 floor(remaining×0.5),不报错(旧 :90-99
        # 对超剩余直接 raise BudgetLimitExceededError;praxis 截断+调用方拒)。
        (900, 1000, 500),
        # requested 小于份额 → 如实授予(旧 :100 granted=requested parity)。
        (100, 1000, 100),
        # 默认(None)→ 份额上限。
        (None, 1000, 500),
        # 向下取整:奇数剩余。
        (None, 1001, 500),
        # 份额 floor ≤0 → grant=0(有界且耗尽,不是 None;T5 据此拒 spawn)。
        (None, 1, 0),
        (None, 0, 0),
        (5, 1, 0),
    ],
)
def test_share_formula_per_dimension_table(
    requested: int | None, parent_remaining: int | None, expected: int | None
) -> None:
    # budget_ops.py:74-102 的 praxis 形:每维独立 min(requested or remaining,
    # floor(remaining × MAX_CHILD_FRACTION));None 维不放大。
    grant = limits_merge_grant(
        None if requested is None else Limits(tokens=requested),
        Limits(tokens=parent_remaining),
    )
    assert grant.tokens == expected
    assert grant.cost_micros is None  # 未请求维不凭空有界


def test_share_formula_is_the_only_expression() -> None:
    # CI㉒ 的矩阵侧钉:份额公式唯一表达在 protocol/budget.py;不存在旧
    # "granted = remaining" 全额转让分支的任何残留。
    assert MAX_CHILD_FRACTION == 0.5
    source = inspect.getsource(budget_protocol)
    assert source.count("def limits_merge_grant") == 1
    # AST 标识符面(docstring 提及旧病灶不算):不存在 granted 全额转让变量。
    identifiers = {node.id for node in ast.walk(ast.parse(source)) if isinstance(node, ast.Name)}
    identifiers |= {node.arg for node in ast.walk(ast.parse(source)) if isinstance(node, ast.arg)}
    assert "granted" not in identifiers
    assert "limits_merge_grant(requested, remaining(node))" in inspect.getsource(agents_module)


# ---------------------------------------------------------------------------
# Step 1 §保护尾/普通余量 vs accounting.py:187-195/:205-212
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("limit", "expected"),
    [
        (None, 0),
        (0, 0),
        (1, 0),
        (2, 1),
        (9, 1),
        (10, 1),
        (11, 1),
        (100, 10),
        (1000, 100),
        (81_919, 8191),
        (81_920, 8192),
        (81_921, 8192),
        (100_000, 8192),
    ],
)
def test_protected_tail_verbatim_table(limit: int | None, expected: int) -> None:
    # accounting.py:187-195 逐字:min(8192, max(1, limit//10), limit-1);
    # None/≤1 → 0。双界(8192 帽与 10% 比例)与下限全覆盖。
    assert protected_tail(limit) == expected


def test_protected_tail_rejects_bool_and_negative() -> None:
    # accounting.py:190-191 同形 ValueError(bool/负数/非 int)。
    for bad in (True, -1, 1.5, "100"):
        with pytest.raises(ValueError):
            protected_tail(bad)  # type: ignore[arg-type]


def test_normal_token_remaining_subtracts_tail(store: Store) -> None:
    # accounting.py:205-212:普通余量 = remaining − protected_tail,下钳 0;
    # 无限维 None。compaction 逃生可入保护尾(admit_dispatch needs_compaction)。
    store.ensure_thread("p")
    create_turn(store, turn_id="pt", thread_id="p", binding={})
    store.append_items("p", "pt", [ItemEntry(ItemKind.MODEL_RESPONSE, _usage_payload(100))])
    node = build_tree(store, thread_id="p", turn_id="pt", root_limits=Limits(tokens=1000), unit_costs=None)
    assert normal_token_remaining(node) == 800  # 900 − 100
    assert remaining(node).tokens == 900
    unlimited = build_tree(store, thread_id="p", turn_id="pt", root_limits=Limits(), unit_costs=None)
    assert normal_token_remaining(unlimited) is None
    # 普通调用不得吃保护尾:used=900 → 普通有效上限 900 → 拒;compaction 放行。
    store.append_items("p", "pt", [ItemEntry(ItemKind.MODEL_RESPONSE, _usage_payload(800))])
    node = build_tree(store, thread_id="p", turn_id="pt", root_limits=Limits(tokens=1000), unit_costs=None)
    assert admit_dispatch(node, needs_compaction=False).reason == "budget_exhausted"
    assert admit_dispatch(node, needs_compaction=True).allowed is True


def test_estimate_cost_ceils_once_like_old_pricing() -> None:
    # pricing.py:159-165 _ceil_cost:整个和向上取整一次,不是逐分量取整。
    cost = CostMicros(input_per_1m=1_000_000, output_per_1m=1_000_000)
    assert estimate_cost(Usage(input_tokens=1, output_tokens=1), cost) == 2
    tiny = CostMicros(input_per_1m=1, output_per_1m=1)
    assert estimate_cost(Usage(input_tokens=1, output_tokens=1), tiny) == 1  # ceil(2e-6)=1
    assert estimate_cost(Usage(input_tokens=0, output_tokens=0), tiny) == 0


# ---------------------------------------------------------------------------
# Step 1 §settle 语义 vs budget_ops.py:170-201(推导式差异显式声明)
# ---------------------------------------------------------------------------


def test_settle_is_derivation_not_a_written_item(store: Store) -> None:
    # 旧 settle_child_budget(budget_ops.py:171-201)在事务内写
    # budget_child_settled fact,且要求子 reserved/uncertain/child_reserved
    # 全零否则 raise。**有意差异**(§5.5):praxis settle 是推导不是写——
    # 子根 turn 终态事实 → build_tree 折叠归还残额;无 settle item、无前置
    # raise、守恒由"事实只增"保证。
    _seed_child(store, child_usage=100, child_status="completed")
    before = len(store.read_items("p"))
    node = build_tree(store, thread_id="p", turn_id="pt", root_limits=Limits(tokens=1000), unit_costs=None)
    child = node.children["alloc-1"]
    assert child.state == "done"
    assert child.used.tokens == 100
    assert node.used.tokens == 100  # 子已用并入父 used
    assert node.reserved.tokens == 0  # 未用 400 推导式归还
    assert exposure(node).tokens == 100
    assert remaining(node).tokens == 900
    # 折叠是纯读:父线程零新 item;ItemKind 全集无 settle/resource 第二账本。
    assert len(store.read_items("p")) == before
    kind_values = {kind.value for kind in ItemKind}
    assert "budget_child_settled" not in kind_values
    assert "resource_usage" not in kind_values


@pytest.mark.parametrize(
    ("child_status", "expected_state", "reserved_tokens"),
    [
        ("completed", "done", 0),
        ("failed", "done", 0),
        ("cancelled", "cancelled", 0),
        # 旧 TERMINAL(subagents.py:26)={completed,failed,cancelled};
        # paused/interrupted/running 保守 active(R-T5-1:可续跑的子不能
        # 提前归还预算)——旧 settle 对非终态直接 raise,praxis 继续占用。
        ("running", "active", 500),
        ("paused", "active", 500),
        ("interrupted", "active", 500),
        (None, "active", 500),
    ],
)
def test_child_terminal_states_derive_allocation_state(
    store: Store, child_status: str | None, expected_state: str, reserved_tokens: int
) -> None:
    _seed_child(store, child_status=child_status)
    node = build_tree(store, thread_id="p", turn_id="pt", root_limits=Limits(tokens=1000), unit_costs=None)
    assert node.children["alloc-1"].state == expected_state
    assert node.reserved.tokens == reserved_tokens


# ---------------------------------------------------------------------------
# Step 1 §memory 常量 vs memory.py:42-45(+lease/retry)
# ---------------------------------------------------------------------------


def test_memory_constants_verbatim() -> None:
    # memory.py:42-45 逐值;lease 300(:336 claim_memory_job lease_seconds=300)
    # 与 retry_after 300(:709)同旧值 parity。
    assert MAX_MEMORY_ENTRIES == 32
    assert MAX_SOURCE_BYTES == 32_000
    assert MAX_ENTRY_CHARS == 2_000
    assert MEMORY_TOKEN_BUDGET == 24_000
    assert MEMORY_LEASE_SECONDS == 300.0
    assert MEMORY_RETRY_SECONDS == 300.0


# ---------------------------------------------------------------------------
# Step 1 §job 状态机 vs memory.py:399-415 + B4 判据正反对照
# ---------------------------------------------------------------------------


def test_job_status_literals_cover_old_state_machine() -> None:
    # 旧 job 行状态:running(:404)、pending/done(:620)、
    # needs_reconciliation/budget_exhausted/failed(:649)。praxis 六态全集
    # 在 core/memory.py 字面存在(行为由 T11 套件钉;此处钉形状覆盖)。
    source = inspect.getsource(memory_module)
    for status in ("pending", "running", "done", "failed", "needs_reconciliation", "budget_exhausted"):
        assert f'"{status}"' in source, f"job 状态 {status} 缺失"


def _old_unresolved_replica(operation_statuses: list[str]) -> bool:
    """旧病灶只读复刻(memory.py:178-183,禁止 import):查状态字符串集合
    {"running","unknown"}——dispatched 不在集合内,漏判在途。仅用于正反对照。"""
    return any(status in {"running", "unknown"} for status in operation_statuses)


def test_b4_fact_window_catches_dispatched_old_set_misses(store: Store) -> None:
    # 正对照:dispatched-未 settle 的模型调用 → turn 未终态 → 事实判据在途。
    turn = make_turn(store, "running", "t")
    store.append_items(
        "t", turn.turn_id, [ItemEntry(ItemKind.MODEL_ATTEMPT, {"attempt_id": "a1", "model": "m"})]
    )
    assert job_unresolved(store, [turn.turn_id], {}) is True
    # 反对照:同一 dispatched 事实,旧状态集合判据漏判(B4 病灶字面)。
    assert _old_unresolved_replica(["dispatched"]) is False
    # 工具面:turn 已 FAILED 但声明窗口未闭合 → 仍在途(memory_identity.py:183-186
    # 的正确集含 dispatched;praxis 更进一步——判据只有事实窗口,连终态 turn
    # 的未闭合窗口也不放过)。
    store.append_items(
        "t",
        turn.turn_id,
        [
            ItemEntry(
                ItemKind.MODEL_RESPONSE,
                {"content": "", "tool_calls": [{"call_id": "c1", "name": "toy", "arguments": {}}]},
            ),
            ItemEntry(ItemKind.TOOL_CALL, {"call_id": "c1", "tool_name": "toy"}),
        ],
    )
    transition(store, EventBus(), turn.turn_id, TurnStatus.FAILED, reason="boom")
    assert job_unresolved(store, [turn.turn_id], {}) is True
    # 负对照:终态且窗口全闭合 → 不在途(不误报)。
    closed = make_turn(store, "completed", "t")
    assert job_unresolved(store, [closed.turn_id], {}) is False
    # 读不到 turn 保守在途(与 build_tree fail-open 同向)。
    assert job_unresolved(store, ["no-such-turn"], {}) is True


def test_b4_criterion_is_fact_window_not_status_string_set() -> None:
    # 结构消亡钉:job_unresolved 函数体(除 docstring)不存在任何状态字符串
    # 集合判据;实现只经 TRANSITIONS(终态事实)+ unclosed_windows(窗口事实,
    # compensation 同一判据函数——不得另写窗口匹配)。
    tree = ast.parse(inspect.getsource(memory_module.job_unresolved))
    function = tree.body[0]
    assert isinstance(function, ast.FunctionDef)
    body = function.body[1:] if function.body and isinstance(function.body[0], ast.Expr) else function.body
    for node in body:
        for inner in ast.walk(node):
            if isinstance(inner, ast.Set):
                literals = {elt.value for elt in inner.elts if isinstance(elt, ast.Constant)}
                assert not literals & {"running", "unknown", "dispatched"}, "状态字符串集合判据复辟"
    source = inspect.getsource(memory_module.job_unresolved)
    assert "unclosed_windows" in source
    assert "TRANSITIONS" in source


# ---------------------------------------------------------------------------
# Step 1 §子限额 4/2/16 vs subagents.py:25;wait 不取消 vs :199-201;
#          task/context_summary ≤20_000 vs subagent.py:52-59
# ---------------------------------------------------------------------------


def test_subagent_limits_4_2_16_verbatim() -> None:
    # subagents.py:25 DEFAULT_LIMITS 逐值;形状改为不可变映射(有意差异,
    # 值不变)。行为 guard(并发/总数/深度)由 test_agents_core.py 钉。
    assert dict(SUBAGENT_LIMITS_DEFAULT) == {"max_concurrent": 4, "max_depth": 2, "max_total": 16}


def test_wait_expiry_does_not_cancel_and_clamps_0_60(workspace: Path) -> None:
    # subagents.py:199-201 先例逐条移植:asyncio.wait 过期**不取消** driver。
    # 有意差异:旧 wait 上限 30(:194),praxis 钳位 0..60(T5/T6 计划钉死,
    # 工具 schema maximum=60 同值)。
    release = asyncio.Event()
    holder: list[SubagentManager] = []
    results: list[Any] = []

    async def scenario() -> None:
        tools = [_spawn_probe(holder, results, wait_child=False), _hang_tool("child_hang", release)]
        adapter = ScriptedAdapter(
            [
                _resp("", (("s1", "spawn_probe", {}),)),
                _resp("", (("h1", "child_hang", {}),)),
                _resp("parent done"),
                _resp("child done"),
            ]
        )
        parent = open_session(workspace, adapter, tools=tools, thread_id="parent")
        manager = SubagentManager(parent=parent, store=parent.store)
        holder.append(manager)
        try:
            result = await parent.submit("go")
            assert result.status is TurnStatus.COMPLETED
            handle = results[0]
            status = await manager.wait(handle.child_thread_id, timeout=0.05)
            assert status.state == "running"  # 过期如实返回,子继续跑
            child_turn = get_turn(parent.store, handle.child_root_turn_id)
            assert child_turn is not None and child_turn.status == "running"
            assert (await manager.wait(handle.child_thread_id, timeout=-5)).state == "running"
            release.set()
            assert (await manager.wait(handle.child_thread_id, timeout=999)).state == "done"
        finally:
            release.set()
            await manager.aclose()
            await parent.aclose()

    asyncio.run(scenario())


def test_task_context_summary_20000_and_wait_timeout_60_bounds() -> None:
    # subagent.py:52-59 parity:task/context_summary max_length=20_000。
    # provider 延迟绑定:validate_input 不触达 surface。
    tools = {tool.definition.name: tool for tool in create_agent_tools(lambda: None)}  # type: ignore[arg-type]
    spawn = tools["spawn_subagent"]
    assert len(spawn.validate_input({"task": "x" * 20_000})["task"]) == 20_000
    assert len(spawn.validate_input({"task": "t", "context_summary": "x" * 20_000})["context_summary"]) == 20_000
    with pytest.raises(ToolValidationError):
        spawn.validate_input({"task": "x" * 20_001})
    with pytest.raises(ToolValidationError):
        spawn.validate_input({"task": "t", "context_summary": "x" * 20_001})
    wait = tools["wait_subagent"]
    assert wait.validate_input({"child_thread_id": "c", "timeout": 60})["timeout"] == 60
    with pytest.raises(ToolValidationError):
        wait.validate_input({"child_thread_id": "c", "timeout": 60.01})


# ---------------------------------------------------------------------------
# Step 1 §integrate diff_limit 200_000/preimage vs child_workspaces.py:184,326-329
# ---------------------------------------------------------------------------


def test_diff_preview_cap_200000_and_truncation_flag(repo: Path, provisioner: GitWorktreeProvisioner) -> None:
    # child_workspaces.py:184 diff_limit ≤200_000 parity:inspect 预览有界,
    # 截断旗标与全文尺寸如实披露(旧分页读面 → praxis 有界预览,有意差异)。
    assert DIFF_PREVIEW_LIMIT_BYTES == 200_000
    ws = provisioner.prepare(repo, uuid4().hex, "write")
    try:
        (ws / "a.txt").write_text("x" * 250_000 + "\n", encoding="utf-8")
        inspection = provisioner.inspect(ws)
        assert inspection.diff_bytes > 200_000
        assert inspection.diff_truncated is True
        assert len(inspection.diff_preview.encode("utf-8")) <= 200_000
        assert "a.txt" in inspection.paths
        assert inspection.preimages["a.txt"] is not None  # HEAD 基线 sha
        assert inspection.postimages["a.txt"] is not None
    finally:
        provisioner.discard(ws)


def test_apply_preimage_recheck_conflict_preserves_both_sides(
    repo: Path, provisioner: GitWorktreeProvisioner
) -> None:
    # preimage 复检语义 parity:父仓快照后漂移 → conflicts 保留不写,
    # 父/子双方字节都在(旧事务性 patch_failed 安全面的 praxis 形)。
    ws = provisioner.prepare(repo, uuid4().hex, "write")
    try:
        (ws / "a.txt").write_text("child-version\n", encoding="utf-8")
        (repo / "a.txt").write_text("parent-drift\n", encoding="utf-8")
        inspection = provisioner.inspect(ws)
        result = provisioner.apply(repo, provisioner.diff(ws), preimages=inspection.preimages)
        assert result.conflicts == ["a.txt"]
        assert result.applied == []
        assert (repo / "a.txt").read_text(encoding="utf-8") == "parent-drift\n"
        assert (ws / "a.txt").read_text(encoding="utf-8") == "child-version\n"
    finally:
        provisioner.discard(ws)


def test_no_reviewed_through_gate_or_verification_receipt(
    repo: Path, provisioner: GitWorktreeProvisioner
) -> None:
    # 有意差异(简化表):旧 child_workspaces.py:326-329 要求 reviewed_through
    # 读满全部 diff 页才可 apply,:202-276 verification receipt 复比—— praxis
    # 无该 UI 面,inspect 有界预览 + apply preimage 复检 + DESTRUCTIVE 显式
    # 审批已闭合安全面。缺席断言:无 reviewed_through 字段/无 receipt 复比;
    # 行为:无人"读页"也直接 apply 成功(preimage 一致即写)。
    field_names = {f.name for f in dataclasses.fields(WorkspaceInspection)}
    assert "reviewed_through" not in field_names
    assert {f.name for f in dataclasses.fields(ApplyResult)} == {"applied", "conflicts"}
    assert "reviewed_through" not in inspect.getsource(agents_module)
    ws = provisioner.prepare(repo, uuid4().hex, "write")
    try:
        (ws / "b.txt").write_text("child-b\n", encoding="utf-8")
        inspection = provisioner.inspect(ws)
        result = provisioner.apply(repo, provisioner.diff(ws), preimages=inspection.preimages)
        assert result.applied == ["b.txt"]
        assert result.conflicts == []
        assert (repo / "b.txt").read_text(encoding="utf-8") == "child-b\n"
    finally:
        provisioner.discard(ws)


# ---------------------------------------------------------------------------
# Step 1 §web 计数 parity(T15;3B Q7 归属)vs web_limits.py:9-10/session.py:1279
# ---------------------------------------------------------------------------


def test_web_defaults_32_64_and_error_code_parity() -> None:
    # web_limits.py:9-10 默认 32/64(Q8 默认生效);错误码逐字
    # (harness/session.py:1279 web_request_budget_exceeded)。
    assert WebLimits().turn_requests == 32
    assert WebLimits().browser_requests == 64
    assert WEB_REQUEST_BUDGET_EXCEEDED == "web_request_budget_exceeded"


@pytest.mark.parametrize(
    ("status", "error_code", "billable"),
    [
        ("ok", None, 1),
        ("error", None, 1),
        ("timeout", None, 1),
        ("cancelled", None, 1),
        # R-T2-1 裁决:outcome_unknown = dispatch 后失联,保守计 1
        # (旧 reserve-before-IO parity,budget_ops.py:284-322)。
        ("outcome_unknown", None, 1),
        # dispatch 前拒绝/未执行不占配额。
        ("denied", None, 0),
        ("not_executed", None, 0),
        ("paused", None, 0),
        # T15:预算 gate 自产拒绝零 IO,不占配额(否则 extend 永远续不上)。
        ("error", "web_request_budget_exceeded", 0),
    ],
)
def test_web_fold_status_billing_matrix(store: Store, status: str, error_code: str | None, billable: int) -> None:
    store.ensure_thread("t")
    create_turn(store, turn_id="turn", thread_id="t", binding={})
    payload: dict[str, Any] = {"call_id": "c1", "tool_name": "web_fetch", "status": status}
    if error_code is not None:
        payload["error_code"] = error_code
    store.append_items(
        "t",
        "turn",
        [
            ItemEntry(ItemKind.TOOL_CALL, {"call_id": "c1", "tool_name": "web_fetch"}),
            ItemEntry(ItemKind.TOOL_RESULT, payload),
        ],
    )
    facts = fold_turn_facts(store.read_turn_items("turn"), unit_costs=None)
    assert facts.web_requests == billable


def test_web_unpaired_call_conservative_and_inflight_gate(store: Store) -> None:
    # 旧 reserve_resource_unit(budget_ops.py:284-322)的 praxis 形(有意差异,
    # §5.5 唯一账本):无 resource_usage item/无第二计数器——dispatch 前写
    # tool_call 事实,未配对保守计 1;进程内 in-flight gate 只覆盖"已 admit
    # 未落库"窗口,落库后 note_result 释放,与折叠不双计。
    turn = make_turn(store, "running", "t")
    store.append_items("t", turn.turn_id, [ItemEntry(ItemKind.TOOL_CALL, {"call_id": "c1", "tool_name": "web_fetch"})])
    assert fold_turn_facts(store.read_turn_items(turn.turn_id), unit_costs=None).web_requests == 1
    tracker = BudgetTracker(
        store=store, thread_id="t", turn_id=turn.turn_id, root_limits=Limits(web_requests=3), unit_costs=None
    )
    assert tracker.admit_web_request("web_fetch", browser=False).allowed is True  # 折叠 1 + inflight 0
    assert tracker.admit_web_request("web_fetch", browser=False).allowed is True  # inflight 1
    denied = tracker.admit_web_request("web_fetch", browser=False)
    assert denied.allowed is False and denied.dimension == "web_requests"  # 1+2=3 耗尽
    assert tracker.admit_web_request("bash", browser=False).allowed is True  # 非 web 不受本维约束
    # 结果落库 → note_result 释放:折叠 1(结果计费,声明已配对)+ inflight 1 < 3 → 放行(不双计)。
    store.append_items(
        "t",
        turn.turn_id,
        [ItemEntry(ItemKind.TOOL_RESULT, {"call_id": "c1", "tool_name": "web_fetch", "status": "ok"})],
    )
    tracker.note_result("web_fetch", browser=False)
    assert tracker.admit_web_request("web_fetch", browser=False).allowed is True


# ---------------------------------------------------------------------------
# Step 1 §有意差异全集:显式缺席断言(每条一行理由)
# ---------------------------------------------------------------------------


def test_no_task_sync_shell(workspace: Path) -> None:
    # 理由(Q4/旧设计 §11 不留双实现):旧 task 壳(subagent.py:153 name="task")
    # = spawn+wait 复制,不带走;同步语义用 spawn+wait 两步表达。
    assert AGENT_TOOL_NAMES == frozenset(
        {"spawn_subagent", "wait_subagent", "send_to_subagent", "cancel_subagent", "integrate_child_workspace"}
    )
    assert "task" not in AGENT_TOOL_NAMES
    session = open_session(workspace, ScriptedAdapter([]), thread_id="t", enable_subagents=True)
    try:
        names = set(session.tools)
        assert "task" not in names
        assert AGENT_TOOL_NAMES <= names
        assert len(names) == 20  # 默认 15 + 五件(T6 口径)
    finally:
        asyncio.run(session.aclose())
    default = open_session(workspace, ScriptedAdapter([]), thread_id="t2")
    try:
        assert not AGENT_TOOL_NAMES & set(default.tools)  # 开关默认关=零扰动
        assert len(default.tools) == 15
    finally:
        asyncio.run(default.aclose())


def test_no_subagent_pause_bubble(workspace: Path) -> None:
    # 理由(Q3 直面):旧 SubagentPause(subagent.py:96)冒泡进父 pending_approvals
    # 要给父开"替子审批"第二决策通道,违反唯一审批面——不移植。子 paused
    # 经 wait 如实可见,父 TurnResult.pending_approvals 不含子的待批项。
    assert not hasattr(agents_protocol, "SubagentPause")
    assert "SubagentPause" not in sdk.__all__
    holder: list[SubagentManager] = []
    results: list[Any] = []

    async def scenario() -> None:
        ask_tool = _toy_tool(
            "ask_probe", lambda a, c: _ok("ask_probe", "approved run"), effects=frozenset({ToolEffect.NETWORK})
        )
        tools = [_spawn_probe(holder, results), ask_tool]
        adapter = ScriptedAdapter(
            [
                _resp("", (("s1", "spawn_probe", {}),)),
                _resp("", (("a1", "ask_probe", {}),)),
                _resp("parent done"),
                _resp("child done after approval"),
            ]
        )
        parent = open_session(workspace, adapter, tools=tools, thread_id="parent")
        manager = SubagentManager(parent=parent, store=parent.store)
        holder.append(manager)
        try:
            result = await parent.submit("go")
            assert result.status is TurnStatus.COMPLETED
            assert result.pending_approvals == ()  # 父不冒泡子的待批
            status = results[1]
            assert status.state == "paused"
            assert len(status.pending_approvals) == 1  # 子自己的审批面可见
            assert result.children_summary[0].state == "paused"
        finally:
            await manager.aclose()
            await parent.aclose()

    asyncio.run(scenario())


def test_no_memory_freeze_snapshot_or_revoke() -> None:
    # 理由(简化表):旧冻结快照(memory.py:95 read_frozen_memory_state)与
    # memory_context_revoked(:114)为旧 turn 绑定面服务;praxis 每轮从 items
    # 重建上下文,无中途漂移面——read_memory 读**当前** state,版本自然更新。
    assert not hasattr(memory_module, "read_frozen_memory_state")
    assert not hasattr(memory_module, "memory_context_revoked")
    assert "memory_context_revoked" not in inspect.getsource(memory_module)
    entry_a = {"key": "a", "title": "A", "text": "first", "kind": "fact"}
    entry_b = {"key": "b", "title": "B", "text": "second", "kind": "fact"}
    holder: list[dict[str, Any]] = [{"version": 1, "entries": [entry_a], "use_memories": True}]
    tool = create_read_memory_tool(lambda: holder[0])

    async def read_keys() -> list[str]:
        result = await tool.run({}, ToolExecContext())
        payload = json.loads(result.content[0].data["text"])
        return [entry["key"] for entry in payload["entries"]]

    assert asyncio.run(read_keys()) == ["a"]
    holder[0] = {"version": 2, "entries": [entry_b], "use_memories": True}
    assert asyncio.run(read_keys()) == ["b"]  # 无冻结:同一工具实例见新版本


def test_no_memory_scope_keys_or_relocation(store: Store) -> None:
    # 理由(简化表):旧多工作区共库需要 memory:<scope_id> 命名空间与迁移审计
    # (memory_identity.py);praxis 每 workspace 一库,库即 scope——键名收敛
    # memory:state/memory:job,无 scope 键、无 relocate API。
    assert STATE_KEY == "memory:state"
    assert JOB_KEY == "memory:job"
    assert not any("relocate" in name for name in sdk.__all__)
    remember_memory(store, key="k", text="v")
    delete_memory(store, key="k")

    def keys(conn: Any) -> list[str]:
        return [row[0] for row in conn.execute("SELECT key FROM store_metadata")]

    all_keys = store.transact(keys)
    assert not any("scope" in key for key in all_keys)
    assert STATE_KEY in all_keys
    # memory 键收敛两族(schema_version 是 store 基础设施,非 memory 面)。
    memory_keys = [key for key in all_keys if "memory" in key]
    assert all(key.startswith(("memory:state", "memory:job")) for key in memory_keys)


def test_non_git_workspace_rejects_write_child(workspace: Path, provisioner: GitWorktreeProvisioner) -> None:
    # 理由(Q5):非 git 无 worktree 原语;完整目录复制快照不做(与阶段 6 影子
    # git 重叠)——拒绝 write 子,可诊断文案,只允许 read_only 子。
    with pytest.raises(ChildWorkspaceRequiresGit) as excinfo:
        provisioner.prepare(workspace, uuid4().hex, "write")
    assert "requires a git repository" in str(excinfo.value)


def test_no_model_call_prededuction_settle_never_truncates(store: Store) -> None:
    # 理由(简化表/§5.5):模型调用不做 SQLite 原子预扣(执行锁已单 thread 单
    # 执行者;预扣=第二账本层)——超额诚实暴露,下一次 admission 拒绝;
    # settle 永不截断事实(旧 accounting.py:77-111 好设计移植)。
    assert not hasattr(budget_module, "reserve_resource_unit")
    budget_source = inspect.getsource(budget_module)
    assert "reserve_resource_unit" not in budget_source
    assert "resource_usage" not in budget_source  # 旧第二账本 fact kind 不带走
    store.ensure_thread("p")
    create_turn(store, turn_id="pt", thread_id="p", binding={})
    store.append_items("p", "pt", [ItemEntry(ItemKind.MODEL_RESPONSE, _usage_payload(1500))])
    before = len(store.read_items("p"))
    node = build_tree(store, thread_id="p", turn_id="pt", root_limits=Limits(tokens=1000), unit_costs=None)
    assert node.used.tokens == 1500  # 事实不被截断
    assert exposure(node).tokens == 1500
    assert remaining(node).tokens == 0  # 下钳 0,exposure 仍诚实
    admission = admit_dispatch(node, needs_compaction=False)
    assert admission.allowed is False and admission.reason == "budget_exhausted"
    assert len(store.read_items("p")) == before  # 折叠/准入零写
    # pricing fail-closed:cost 限额生效而单价缺失 → 拒,绝不按 0 计。
    priced = build_tree(
        store, thread_id="p", turn_id="pt", root_limits=Limits(cost_micros=10), unit_costs=None
    )
    assert admit_dispatch(priced, needs_compaction=False).reason == "pricing_unavailable"


def test_parent_completion_does_not_drain_children(workspace: Path) -> None:
    # 理由(简化表):§5.13 级联列表只含 CANCELLED/FAILED/INTERRUPTED,刻意不含
    # COMPLETED——旧 drain/TurnChildrenPendingError(session.py:548/1512-1513)
    # 不移植;异步 spawn 的本意是父不回头也可,子跑完自然 settle。
    assert not hasattr(agents_module, "TurnChildrenPendingError")
    assert "_drain_children" not in inspect.getsource(agents_module)
    release = asyncio.Event()
    holder: list[SubagentManager] = []
    results: list[Any] = []

    async def scenario() -> None:
        tools = [_spawn_probe(holder, results, wait_child=False), _hang_tool("child_hang", release)]
        adapter = ScriptedAdapter(
            [
                _resp("", (("s1", "spawn_probe", {}),)),
                _resp("", (("h1", "child_hang", {}),)),
                _resp("parent done"),
                _resp("child done"),
            ]
        )
        parent = open_session(workspace, adapter, tools=tools, thread_id="parent")
        manager = SubagentManager(parent=parent, store=parent.store)
        holder.append(manager)
        try:
            result = await parent.submit("go")
            # 父 COMPLETED 时子仍在跑——无完成门槛/drain。
            assert result.status is TurnStatus.COMPLETED
            assert result.children_summary[0].state == "running"
            handle = results[0]
            child_turn = get_turn(parent.store, handle.child_root_turn_id)
            assert child_turn is not None and child_turn.status == "running"
            release.set()
            assert (await manager.wait(handle.child_thread_id, timeout=5.0)).state == "done"
        finally:
            release.set()
            await manager.aclose()
            await parent.aclose()

    asyncio.run(scenario())


def test_delete_history_action_is_delete_not_forget(store: Store) -> None:
    # 理由(T13 注记):旧 memory_management.py:210 action 枚举含 "forget";
    # praxis CLI 动词与 history action 统一为 "delete"(语义不变:删除+防复生)。
    remember_memory(store, key="k", text="v")
    delete_memory(store, key="k")
    state = read_memory_state(store)
    assert state is not None
    history = state["entry_history"]["k"]  # type: ignore[index]
    actions = [record["action"] for record in history]  # type: ignore[index]
    assert actions == ["remember", "delete"]
    assert "forget" not in actions
    assert "forget" not in inspect.getsource(memory_module)


def test_canonical_json_envelope_only_args_digest_keeps_own_convention(store: Store) -> None:
    # 理由(T10 注记):canonical_json 唯一口径(protocol/json.py,ensure_ascii=
    # False)只服务 memory 包络;args_digest(protocol/tools.py:426)保持自有
    # convention(json.dumps 默认 ensure_ascii=True)——决策面与持久化面各
    # 自洽,不强行统一(统一会翻案既有审批幂等键)。
    arguments = {"text": "记忆"}
    assert args_digest(arguments) != hashlib.sha256(canonical_json(arguments).encode("utf-8")).hexdigest()
    state: dict[str, Any] = {**initial_state(), "entries": [{"key": "k", "title": "标题", "text": "正文"}]}

    def write(conn: Any) -> int:
        return put_state_enveloped(conn, STATE_KEY, state)

    version = store.transact(write)
    assert version == 1
    raw = get_metadata(store, STATE_KEY)
    assert raw is not None
    envelope = json.loads(raw)
    assert envelope["sha256"] == hashlib.sha256(canonical_json(envelope["state"]).encode("utf-8")).hexdigest()

    def read(conn: Any) -> Any:
        return read_state_verified(conn, STATE_KEY)

    verified = store.transact(read)
    assert verified is not None and verified["version"] == 1


def test_empty_workspace_memory_flags_report_true(store: Store) -> None:
    # 理由(T11 initial_state):旧无 DB 时 read_memory_state 报
    # use_memories/generate_memories=False(runtime_store.py:360-361/375-376);
    # praxis 旗标默认 True,开关闸在 composition enable_memory(默认关)——
    # 空工作区 status 如实报 true,不是"未启用"的假阴性。
    view = read_memory_status(store, {})
    assert view.use_memories is True
    assert view.generate_memories is True
    assert view.version == 0
    assert view.total_entries == 0
    assert initial_state()["use_memories"] is True
    assert initial_state()["generate_memories"] is True


# ---------------------------------------------------------------------------
# Step 2 §findings 回归清单:逐号 parametrize 可指认 + 独立复跑小回归
# ---------------------------------------------------------------------------

#: finding → owning 回归测试(文件相对 tests/praxis,函数名)。全量 owning
#: 套件由 Step 4 全门禁真跑;本表 AST 存在性钉住"回归测试确实挂在编号上"。
_FINDINGS_REGISTRY: dict[str, list[tuple[str, str]]] = {
    "B2": [
        ("test_budget_protocol.py", "test_b2_regression_child_never_gets_full_parent_remaining"),
        ("test_budget_protocol.py", "test_limits_merge_grant_table"),
        ("test_budget_tree.py", "test_exposure_adds_active_unused_grants"),
        ("test_agents_core.py", "test_spawn_default_grant_reserves_half_of_parent_remaining"),
        ("test_agents_core.py", "test_spawn_grant_exhausted_dimension_rejected_zero_write"),
        ("acceptance/test_parity_phase4.py", "test_b2_default_grant_is_half_never_full_remaining"),
    ],
    "B4": [
        ("test_memory_extract.py", "test_job_unresolved_dispatched_tool_window_even_if_turn_terminal"),
        ("test_memory_extract.py", "test_kill9_after_maintenance_dispatch_needs_reconciliation"),
        ("acceptance/test_parity_phase4.py", "test_b4_fact_window_catches_dispatched_old_set_misses"),
    ],
    "B14": [
        ("test_cancel.py", "test_request_cancel_signalled_settles_started_call"),
        ("test_cancel.py", "test_transitions_table_unchanged"),
        ("test_agents_core.py", "test_parent_cancelled_cascades_cancel_request_to_child"),
        ("test_agents_core.py", "test_sweep_interrupted_parent_cascades_and_interrupted_child_keeps_share"),
        ("test_agents_core.py", "test_parent_resume_does_not_revive_cancelled_child"),
        ("acceptance/test_budget_conserve_faults.py", "test_cancel_race_cascade_first_blocks_child_dispatch"),
        ("acceptance/test_budget_conserve_faults.py", "test_cancel_race_dispatch_first_traces_and_stop_settlement"),
        ("acceptance/test_parity_phase4.py", "test_finding_b14_cascade_and_no_permanent_active"),
    ],
    "B23": [
        ("test_store_memory.py", "test_prune_versions_keeps_bounded_window_and_latest"),
        ("test_store_memory.py", "test_prune_job_history_bounded_idempotent_and_live_jobs_survive"),
        ("test_memory_extract.py", "test_record_revision_stamps_and_prunes_entry_history_to_100"),
        ("test_memory_extract.py", "test_version_snapshots_pruned_in_commit_tx"),
        ("acceptance/test_parity_phase4.py", "test_finding_b23_bounded_across_100_cycles"),
    ],
    "B29": [
        ("test_compensation.py", "test_b29_first_missing_result_does_not_abort_group"),
        ("test_compensation.py", "test_b29_brief_case_first_has_result_rest_missing"),
        ("test_cancel.py", "test_b29_cancel_settles_whole_group"),
        ("acceptance/test_parity_phase4.py", "test_finding_b29_whole_group_settlement"),
    ],
}


@pytest.mark.parametrize("finding", sorted(_FINDINGS_REGISTRY))
def test_findings_regression_locations(finding: str) -> None:
    # roadmap §7.4 字面验收的测试面:默认份额上限(B2 行)与父子级联取消
    # (B14 行)的 owning 回归全部在位;五号逐条可指认。
    for relative, function_name in _FINDINGS_REGISTRY[finding]:
        path = _TESTS_DIR / relative
        assert path.is_file(), f"{finding}: 回归文件缺失 {relative}"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
        assert function_name in names, f"{finding}: 回归测试缺失 {relative}::{function_name}"


def test_finding_b14_cascade_and_no_permanent_active(store: Store) -> None:
    # B14 正证:级联唯一实现 cascade_for(父三终态全挂钩)+ 分配终态=推导
    # (子终态事实 → settle)——结构上不存在"永不回落的 active"。
    _seed_child(store)
    assert cascade_for(store, "p", "pt") == 1
    child_items = store.read_turn_items("ct")
    cancels = [item for item in child_items if item.kind == ItemKind.CANCEL_REQUEST]
    assert len(cancels) == 1
    assert cancels[0].payload["source"] == "parent"
    assert cancels[0].payload["parent_turn_id"] == "pt"
    # 幂等:已有未处理 cancel_request → 跳过。
    assert cascade_for(store, "p", "pt") == 0
    # 子收敛 CANCELLED → 分配推导终态,reserved 回落 0(无永久 active)。
    transition(store, EventBus(), "ct", TurnStatus.CANCELLED, reason="cascade")
    node = build_tree(store, thread_id="p", turn_id="pt", root_limits=Limits(tokens=1000), unit_costs=None)
    assert node.children["alloc-1"].state == "cancelled"
    assert node.reserved.tokens == 0
    assert all(child.state != "active" for child in node.children.values())
    assert cascade_for(store, "p", "pt") == 0  # 终态子跳过


def test_finding_b23_bounded_across_100_cycles(store: Store) -> None:
    # B23:版本/历史只增不删的病灶 → 三常量有界 + CAS 同事务 prune。
    # 跨 100 次循环(学习循环的存储缝等价物;真 run_update 循环由
    # test_memory_extract.py::test_version_snapshots_pruned_in_commit_tx 钉)。
    assert (MEMORY_VERSION_HISTORY_KEEP, MEMORY_ENTRY_HISTORY_KEEP, MEMORY_JOB_HISTORY_KEEP) == (20, 100, 10)
    state = initial_state()

    def commit(conn: Any) -> int:
        version = put_state_enveloped(conn, STATE_KEY, state)
        prune_versions(conn, STATE_KEY)  # 生产消费面同事务 prune(core.memory._mutate_state)
        return version

    for _ in range(100):
        version = store.transact(commit)
    assert version == 100
    snapshots = [n for n in range(1, 101) if get_metadata(store, f"{STATE_KEY}:version:{n}") is not None]
    assert snapshots == list(range(81, 101))  # 只留最近 20
    assert get_metadata(store, STATE_KEY) is not None  # latest 指针永不删

    # entry_history:同 key 100 次修订 → 截断到 100(第 101 次才挤掉最旧)。
    revision_state: dict[str, Any] = {**initial_state(), "entry_history": {}}
    entry = {"key": "k", "title": "t", "text": "x", "source_item_ids": ["i1"], "kind": "fact"}
    for i in range(100):
        revision_state["version"] = i
        record_revision(revision_state, entry, action="learn", origin="learned", kind_map={"i1": "user_message"})
    history = revision_state["entry_history"]["k"]
    assert len(history) == MEMORY_ENTRY_HISTORY_KEEP == 100
    assert history[-1]["version"] == 100

    # job history:100 条终态作业行 → 只留最近 10;在跑 JOB_KEY 非数字后缀永存。
    def archive(conn: Any, seq: int) -> None:
        cas_put_metadata(conn, f"{JOB_KEY}:{seq}", None, json.dumps({"status": "done", "history_seq": seq}))
        prune_job_history(conn, "memory")

    for seq in range(1, 101):
        store.transact(lambda conn, bound=seq: archive(conn, bound))
    live = [n for n in range(1, 101) if get_metadata(store, f"{JOB_KEY}:{n}") is not None]
    assert live == list(range(91, 101))


def test_finding_b29_whole_group_settlement(store: Store) -> None:
    # B29 病灶(旧 ownership.py:181-190 首个 missing-result 早退):同窗口 3
    # 调用,第 1 个已有结果、第 2 个 started 无结果、第 3 个未 started →
    # 整组清算 2 条,status 各判正确;幂等:补过结果的窗口第二次清算零产出。
    turn = make_turn(store, "running", "t")
    store.append_items(
        "t",
        turn.turn_id,
        [
            ItemEntry(
                ItemKind.MODEL_RESPONSE,
                {
                    "content": "",
                    "tool_calls": [
                        {"call_id": "c1", "name": "toy", "arguments": {}},
                        {"call_id": "c2", "name": "toy", "arguments": {}},
                        {"call_id": "c3", "name": "toy", "arguments": {}},
                    ],
                },
            ),
            ItemEntry(
                ItemKind.TOOL_RESULT,
                {"call_id": "c1", "tool_name": "toy", "status": "ok", "content": ()},
            ),
            ItemEntry(ItemKind.TOOL_CALL, {"call_id": "c2", "tool_name": "toy"}),
        ],
    )
    tools = {"toy": _toy_tool("toy", lambda a, c: _ok("toy", "done"))}
    items = store.read_turn_items(turn.turn_id)
    entries, unknown = close_unresolved(items, tools, text_for="cancelled")
    assert [(entry.payload["call_id"], entry.payload["status"]) for entry in entries] == [
        ("c2", "outcome_unknown"),
        ("c3", "not_executed"),
    ]
    assert unknown == ["c2"]
    store.append_items("t", turn.turn_id, entries)
    again, again_unknown = close_unresolved(store.read_turn_items(turn.turn_id), tools, text_for="cancelled")
    assert again == [] and again_unknown == []
