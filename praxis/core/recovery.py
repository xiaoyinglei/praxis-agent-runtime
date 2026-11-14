"""Praxis recovery:sweep 扫描与 classify_pending 待决工具调用分类(设计 §5.8)。

本文件是 praxis/ 中 ``transition(..., TurnStatus.INTERRUPTED, ...)`` 唯一合法
调用点(CI 硬规则 5 强制):interrupted 只能由 recovery 写入——人为中断走
pause/cancel,进程死亡走 sweep。

sweep(执行锁粒度 = thread,§5.1):
    遍历本库所有 status="running" 的 turn,先**试取**该 thread 的执行锁——
    - ThreadBusy:另一进程正在执行,跳过,绝不误伤活 turn;
    - 取得到:flock 随进程死亡由内核自动释放,取到即原主已死 →
      transition(INTERRUPTED, reason="executor_process_died"),
      turn_status item 由 write_status 自动携带 last_seq(打断瞬间日志里
      已存在的最后一条 seq),recovery 续跑据此决定从哪读。
    任何打开本库的进程都可触发(如 composition 启动时);自身全程只借锁判定
    生死,标记完立即释放锁。

classify_pending:resume 前的纯函数分类器(Task 9/14 消费),无副作用——
把最后一条 model_response 里尚未拿到 tool_result 的调用,按"是否已写下
tool_call item"(可能已有副作用)与"工具是否幂等"(能否安全重放)标注。
结果配对判据与投影共用 ``core.context.pair_tool_results``:调用身份是
"最后一条 model_response 的声明窗口",不是裸 call_id——同 turn 跨轮复用
call_id 时,第一轮的结果绝不能把第二轮尚未执行的同 id 调用判为已解决
(R-T7-1,否则 resume 静默丢掉一次调用)。

sweep_orphan_command_jobs(Task 15,findings B20/D13):宿主崩溃后,后台 bash
会话的 marker(``command_jobs_dir()/<session_id>.json``)还在,进程组可能还活着。
打开会话时清扫:marker 是**宿主进程**写的,记录 pid/pgid/命令行前缀;清扫
先 ``ps -p <pid> -o pgid=,args=`` 实测,**仅当 pgid 与记录一致且命令行前缀匹配**
才整组 SIGTERM → 1s → SIGKILL(防 PID 复用误杀,Review Focus 5),随后删 marker。
损坏 marker 直接删并记 corrupt;不匹配/进程已死记 stale 并删 marker;探测或
信号自身失败记 untouchable 并保留 marker(留给下次清扫)。**任何情况下不抛**
(它挂在打开会话的关键路径上)。进程组信号走 ``protocol.process.signal_group``
(core 不得 import tools;实现仍只有一份)。
"""

from __future__ import annotations

import json
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from praxis.core.context import pair_tool_results
from praxis.core.locks import ThreadBusy, thread_execution_lock
from praxis.core.statemachine import IllegalTransition, TurnStatus, transition
from praxis.protocol.floor import command_jobs_dir
from praxis.protocol.items import Item, ItemKind
from praxis.protocol.json import JsonValue
from praxis.protocol.models import ModelToolCall
from praxis.protocol.process import signal_group
from praxis.store.turns import list_turns

if TYPE_CHECKING:
    from praxis.core.events import EventBus
    from praxis.protocol.tools import Tool
    from praxis.store.log import Store


def sweep(store: Store, bus: EventBus, workspace: Path) -> list[str]:
    """清扫:running 且执行锁空闲(原主已死)的 turn → interrupted。

    返回被标记的 turn_id 列表(按 list_turns 的 created_at 顺序)。
    单条 turn 的任何失败绝不中断整次清扫:sweep 是 recovery 入口,
    漏扫其余 dead turn 的代价远大于跳过一条。
    """
    marked: list[str] = []
    for turn in list_turns(store, status=TurnStatus.RUNNING.value):
        try:
            with thread_execution_lock(workspace, turn.thread_id):
                transition(
                    store,
                    bus,
                    turn.turn_id,
                    TurnStatus.INTERRUPTED,
                    reason="executor_process_died",
                )
        except ThreadBusy:
            continue
        except IllegalTransition:
            # R6 竞态:快照与取锁之间,该 turn 已被活执行者合法改写(如 completed)
            # ——不再是 dead,跳过并继续清扫其余。执行锁保证这不是"执行中":
            # 活执行者改写状态全程持锁,能取到锁就说明它已经交还了该 thread。
            continue
        marked.append(turn.turn_id)
    return marked


# ---------------------------------------------------------------------------
# 后台命令会话的崩溃遗留清扫(Task 15;B20/D13)
# ---------------------------------------------------------------------------

_ORPHAN_PROBE_TIMEOUT_SECONDS = 5.0
_ORPHAN_TERM_GRACE_SECONDS = 1.0
_MARKER_SUFFIX = ".json"


@dataclass(frozen=True, slots=True)
class OrphanSummary:
    """一个 marker 的清扫结局(killed/stale/corrupt/untouchable)。"""

    session_id: str
    pid: int | None
    state: Literal["killed", "stale", "corrupt", "untouchable"]
    detail: str = ""
    age_seconds: float | None = None


def _probe_process(pid: int) -> tuple[int, str] | None:
    """``ps -p <pid> -o pgid=,args=`` 实测;进程不存在 → None;ps 自身失败抛 OSError。

    返回 ``(pgid, args)``。不吞 ps 的失败:调用方必须把"探不到"和"进程死了"
    区分开——前者不能当 stale 删 marker。
    """
    completed = subprocess.run(
        ["ps", "-p", str(pid), "-o", "pgid=,args="],
        capture_output=True,
        text=True,
        timeout=_ORPHAN_PROBE_TIMEOUT_SECONDS,
        check=False,
    )
    if completed.returncode != 0:
        return None
    line = completed.stdout.strip()
    if not line:
        return None
    parts = line.split(maxsplit=1)
    if len(parts) != 2:
        return None
    try:
        observed_pgid = int(parts[0])
    except ValueError:
        return None
    return observed_pgid, parts[1]


def _kill_group(pgid: int, sig: signal.Signals) -> None:
    """孤儿清扫的唯一信号出口(测试可 monkeypatch;实现是 process.py 原语)。"""
    signal_group(pgid, sig)


def _command_matches(command_prefix: str, observed: str) -> bool:
    """命令行匹配:折叠空白后,实测 args 以空格边界包含记录的命令前缀。

    命令是 ``bash -lc <command>`` 的最后一个 argv 元素(sandbox-exec 也保留完整
    argv),简单命令被 bash tail-exec 后 ps 的 args 直接就是命令文本。用
    "空格边界包含"而不是裸子串,避免 ``true`` 命中 ``/usr/bin/true`` 之类误杀;
    pgid 一致仍是第一道守卫(PID 复用防线,Review Focus 5)。
    """
    expected = " ".join(command_prefix.split())
    if not expected:
        return False
    return f" {expected}" in f" {' '.join(observed.split())}"


def _read_marker(path: Path) -> dict[str, JsonValue] | None:
    """读 marker;任何畸形(坏 JSON/缺字段/类型不符)返回 None(→ corrupt)。"""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(raw, dict):
        return None
    fields = ("session_id", "pid", "pgid", "command_prefix", "created_at")
    if not all(field in raw for field in fields):
        return None
    if not isinstance(raw["session_id"], str) or not raw["session_id"]:
        return None
    if type(raw["pid"]) is not int or type(raw["pgid"]) is not int:
        return None
    if not isinstance(raw["command_prefix"], str):
        return None
    if isinstance(raw["created_at"], bool) or not isinstance(raw["created_at"], (int, float)):
        return None
    return raw


def _unlink_marker(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _sweep_one_marker(marker: Path, now: float) -> OrphanSummary:
    session_id = marker.stem
    record = _read_marker(marker)
    if record is None:
        _unlink_marker(marker)
        return OrphanSummary(session_id, None, "corrupt", "malformed marker removed")
    pid = int(record["pid"])  # type: ignore[arg-type]
    pgid = int(record["pgid"])  # type: ignore[arg-type]
    prefix = str(record["command_prefix"])
    created_at = float(record["created_at"])  # type: ignore[arg-type]
    age = max(0.0, now - created_at)

    try:
        probe = _probe_process(pid)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        # marker 保留:下次清扫可重试,丢失登记会让活进程永久脱管。
        return OrphanSummary(session_id, pid, "untouchable", f"probe failed: {exc}", age)
    if probe is None:
        _unlink_marker(marker)
        return OrphanSummary(session_id, pid, "stale", "process is gone", age)
    observed_pgid, args = probe
    if observed_pgid != pgid or not _command_matches(prefix, args):
        # PID 已复用(或命令不符):绝不向陌生进程组发信号(Review Focus 5)。
        _unlink_marker(marker)
        return OrphanSummary(
            session_id, pid, "stale", "pid reused or command line mismatch", age
        )

    try:
        _kill_group(pgid, signal.SIGTERM)
        time.sleep(_ORPHAN_TERM_GRACE_SECONDS)
        _kill_group(pgid, signal.SIGKILL)
    except (OSError, ValueError) as exc:
        return OrphanSummary(session_id, pid, "untouchable", f"signal failed: {exc}", age)
    _unlink_marker(marker)
    return OrphanSummary(session_id, pid, "killed", "process group terminated", age)


def sweep_orphan_command_jobs(
    jobs_dir: Path | None = None, *, now: float | None = None
) -> tuple[OrphanSummary, ...]:
    """清扫后台命令会话的崩溃遗留 marker(见模块 docstring;**绝不抛**)。

    ``now`` 只影响 summary 的 ``age_seconds``(默认 ``time.time()``),不改变
    清扫判定:marker 存在即登记,年龄不是安全条件(有 pid/pgid/命令行三重守卫)。
    """
    directory = Path(jobs_dir) if jobs_dir is not None else command_jobs_dir()
    try:
        markers = sorted(directory.glob(f"*{_MARKER_SUFFIX}"))
    except OSError:
        return ()
    clock = time.time() if now is None else now
    summaries: list[OrphanSummary] = []
    for marker in markers:
        try:
            summaries.append(_sweep_one_marker(marker, clock))
        except Exception as exc:  # noqa: BLE001 — 关键路径:单条意外绝不中断清扫
            summaries.append(
                OrphanSummary(marker.stem, None, "untouchable", f"unexpected: {exc}")
            )
    return tuple(summaries)


@dataclass(frozen=True, slots=True)
class PendingCall:
    """一条已发出但未拿到 tool_result 的工具调用。

    - started:日志里已有对应 tool_call item —— 副作用可能已经发生;
    - idempotent:tools 表给出的重放安全性(未知工具一律按 False)。
    resume 决策矩阵(§5.9):未 started 的幂等调用可直接重放;
    已 started 或非幂等的必须人工确认后才能重试。
    """

    call: ModelToolCall
    started: bool
    idempotent: bool


def classify_pending(items: Sequence[Item], tools: Mapping[str, Tool]) -> list[PendingCall]:
    """从 items(一个 turn 的完整日志)提取最后一条 model_response 的待决调用。

    规则:
    - 只看**最后**一条 model_response(seq 最大)——更早轮次的调用已被模型
      自己接管;
    - 结果判据走 ``pair_tool_results`` 的声明窗口:只有配给**本条**
      model_response 的 tool_result 才算已解决。同 turn 跨轮复用 call_id 时,
      更早轮次的结果属于更早的声明窗口,不得吞掉本轮的调用(R-T7-1);
    - started = 存在**本条声明之后**(seq 更大)的同 call_id tool_call item
      ——更早轮次同 id 的 tool_call 不是本窗口的 dispatch;
    - idempotent = tools[name].idempotent,未知工具按 False;
    - 读路径零校验(B1):结构不符的条目静默跳过,绝不抛异常。
    """
    last_response: Item | None = None
    for item in items:
        if item.kind == ItemKind.MODEL_RESPONSE and (
            last_response is None or item.seq >= last_response.seq
        ):
            last_response = item

    if last_response is None:
        return []
    raw_calls = last_response.payload.get("tool_calls")
    if not isinstance(raw_calls, (list, tuple)):
        return []

    paired, _orphans = pair_tool_results(items)
    started_call_ids: set[str] = set()
    for item in items:
        if item.kind != ItemKind.TOOL_CALL or item.seq <= last_response.seq:
            continue
        call_id = item.payload.get("call_id")
        if isinstance(call_id, str) and item.turn_id == last_response.turn_id:
            started_call_ids.add(call_id)

    pending: list[PendingCall] = []
    for entry in raw_calls:
        if not isinstance(entry, Mapping):
            continue
        call_id = entry.get("call_id")
        name = entry.get("name")
        if not isinstance(call_id, str) or not isinstance(name, str):
            continue
        if (last_response.turn_id, last_response.seq, call_id) in paired:
            continue  # 本窗口已有结果(更早窗口的结果不算)
        raw_arguments = entry.get("arguments")
        arguments: Mapping[str, JsonValue] = raw_arguments if isinstance(raw_arguments, Mapping) else {}
        tool = tools.get(name)
        pending.append(
            PendingCall(
                call=ModelToolCall(call_id=call_id, name=name, arguments=arguments),
                started=call_id in started_call_ids,
                idempotent=tool.idempotent if tool is not None else False,
            )
        )
    return pending
