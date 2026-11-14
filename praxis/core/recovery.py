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
打开会话时清扫:marker 是**宿主进程**写的,记录 host_pid/pid/pgid/命令行前缀;
清扫先做**宿主存活门**——host_pid 仍活着(含本进程)⇒ ``other_host``,不动不删
(设计 §5.11 支持多进程共存,不能拆活会话的台);宿主已死后才 ``ps -p <pid>
-o pgid=,args=`` 实测,**仅当 pgid 与记录一致且命令行前缀匹配**才整组
SIGTERM → 1s → SIGKILL(防 PID 复用误杀,Review Focus 5),随后删 marker。
损坏 marker 直接删并记 corrupt;不匹配/进程已死记 stale 并删 marker;探测或
信号自身失败记 untouchable 并保留 marker(留给下次清扫)。**任何情况下不抛**
(它挂在打开会话的关键路径上)。信号实现由 composition 注入(``ProcessSignaler``,
R-T15-7),core 不 import tools。

已知局限(如实):
- 命令行守卫是"折叠空白后按空格边界包含",不是全等:对 ``sleep 20``/``ls``/
  ``make`` 这类短通用命令,若 PID 恰好被复用且进程自成组,仍有残余误判可能;
  与 pgid 相等联合使用后风险可接受(且这是能识别 bash tail-exec 的唯一可行匹配)。
- ``_probe_process`` 探的是**组长 pid**:组长先退出而组员仍存活时,``ps``
  查不到进程(或 pgid 不再匹配),清扫只删 marker、不回收存活组员——
  R-T15-2 的字面要求如此,属已知局限。
"""

from __future__ import annotations

import json
import os
import re
import signal
import stat
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
from praxis.protocol.sandbox import ProcessSignaler
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
_MARKER_MAX_BYTES = 4096
_SESSION_ID_PATTERN = re.compile(r"cmd_[0-9a-f]{32}")


@dataclass(frozen=True, slots=True)
class OrphanSummary:
    """一个 marker 的清扫结局(other_host/killed/stale/corrupt/untouchable)。"""

    session_id: str
    pid: int | None
    state: Literal["other_host", "killed", "stale", "corrupt", "untouchable"]
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


def _host_alive(pid: int) -> bool:
    """宿主存活门(R-T15-6):不确定时按"活着"处理——宁可不清,绝不误杀。"""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return True
    return True


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
    """读 marker(拒 symlink、校验属主/权限/普通文件);任何畸形返回 None(→ corrupt)。

    与 ``adapters/config/credentials.py`` 的私密文件纪律对齐(R-T15-9):
    ``O_NOFOLLOW``、``S_ISREG``、uid 必须是当前用户、组/其他无权限、
    ``st_nlink == 1``、有界大小(不把磁盘/攻击者的任意大文件读进内存)。
    """
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return None
    try:
        handle = os.fdopen(descriptor, "rb")
    except OSError:
        os.close(descriptor)
        return None
    with handle:
        try:
            metadata = os.fstat(handle.fileno())
            if (
                not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or metadata.st_mode & 0o077
                or metadata.st_nlink != 1
                or metadata.st_size > _MARKER_MAX_BYTES
            ):
                return None
            raw = json.loads(handle.read())
        except (OSError, ValueError):
            return None
    if not isinstance(raw, dict):
        return None
    fields = ("session_id", "host_pid", "pid", "pgid", "command_prefix", "created_at")
    if not all(field in raw for field in fields):
        return None
    if not isinstance(raw["session_id"], str) or not _SESSION_ID_PATTERN.fullmatch(
        raw["session_id"]
    ):
        return None
    if raw["session_id"] != path.stem:
        return None
    for field in ("host_pid", "pid", "pgid"):
        if type(raw[field]) is not int or raw[field] <= 0:
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


def _sweep_one_marker(marker: Path, now: float, kill_group: ProcessSignaler) -> OrphanSummary:
    session_id = marker.stem
    record = _read_marker(marker)
    if record is None:
        _unlink_marker(marker)
        return OrphanSummary(session_id, None, "corrupt", "malformed marker removed")
    host_pid = int(record["host_pid"])  # type: ignore[arg-type]
    pid = int(record["pid"])  # type: ignore[arg-type]
    pgid = int(record["pgid"])  # type: ignore[arg-type]
    prefix = str(record["command_prefix"])
    created_at = float(record["created_at"])  # type: ignore[arg-type]
    age = max(0.0, now - created_at)

    # R-T15-6:宿主仍活着(含本进程)⇒ 这是另一个活会话的登记,不动不删。
    # 设计 §5.11 明支持 CLI 与 TUI/host 多进程共存,清扫不得拆别人的台。
    if _host_alive(host_pid):
        return OrphanSummary(
            session_id, pid, "other_host", f"host process {host_pid} is alive", age
        )

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
        kill_group(pgid, signal.SIGTERM)
        time.sleep(_ORPHAN_TERM_GRACE_SECONDS)
        kill_group(pgid, signal.SIGKILL)
    except (OSError, ValueError) as exc:
        return OrphanSummary(session_id, pid, "untouchable", f"signal failed: {exc}", age)
    _unlink_marker(marker)
    return OrphanSummary(session_id, pid, "killed", "process group terminated", age)


def sweep_orphan_command_jobs(
    jobs_dir: Path | None = None,
    *,
    now: float | None = None,
    kill_group: ProcessSignaler,
) -> tuple[OrphanSummary, ...]:
    """清扫后台命令会话的崩溃遗留 marker(见模块 docstring;**绝不抛**)。

    ``kill_group`` 由 composition 注入真 ``tools.builtins.process.signal_group``
    (R-T15-7):core 不 import tools,protocol 只提供纯类型别名。
    ``now`` 只影响 summary 的 ``age_seconds``(默认 ``time.time()``),不改变
    清扫判定:marker 存在即登记,年龄不是安全条件(有宿主存活门 + pid/pgid/
    命令行守卫)。
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
            summaries.append(_sweep_one_marker(marker, clock, kill_group))
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
