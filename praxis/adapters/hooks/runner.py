"""command hook 运行器:纯进程机制,无 store、无 Turn 归属(设计 §5.14;3B 笔记 §4)。

分层裁决(计划自审修正):``praxis.adapters`` 不得触达 store/core,所以
``spawn_hook`` 只做 **spawn + 解析 + 清理**,返回 ``HookOutcome``;幂等折叠、
started/settled item 写入、外层取消结算全部在 pipeline(T5)承担。

spawn 姿势逐条对齐旧 ``agent_runtime/hooks.py:163-171``:

- ``cwd=spec.directory``(hooks.json 父目录,绝对路径);
- ``env={"PATH": os.defpath}``——宿主环境不泄漏给 hook;
- stdin=真 HookRequest JSON(UTF-8,写后关闭)、stdout=PIPE、stderr=DEVNULL;
- ``start_new_session=True`` ⇒ 子进程成为进程组组长,清理按**整组**发信号;
- stdout 用 ``readexactly(MAX_HOOK_STDOUT_BYTES)`` 哨兵:能读出 16_385 字节
  即超过 16 KiB 上限(16_384)⇒ ``error``;
- 裁决面:空 stdout ⇒ ``allowed``;JSON object 且键 ⊆ ``{block, feedback}``
  ⇒ ``blocked``/``allowed``(feedback 由 ``HookOutcome`` 截断到 2048 字节);
  未知键、非 object、非 JSON、非 0 退出 ⇒ ``error``;
- 超时 ⇒ ``timeout``;清理一律经**注入的** ``ProcessSignaler`` 对整组
  SIGKILL(与旧实现一致),随后 0.5s 有界排空——孙进程攥着管道也不能拖住
  整个 Turn;外层 ``CancelledError`` 清理后**原样上抛**,绝不吞掉。

``HookOutcome.source`` 留空:调用方(T5)用 ``spec.name`` 组装 ``hook:<name>:<status>``。
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
from collections.abc import Mapping

from praxis.protocol.hooks import (
    MAX_HOOK_STDOUT_BYTES,
    HookOutcome,
    HookSpec,
    HookStatus,
)
from praxis.protocol.json import JsonValue, thaw_json
from praxis.protocol.sandbox import ProcessSignaler

_DECISION_KEYS = frozenset({"block", "feedback"})
_DRAIN_TIMEOUT_SECONDS = 0.5
_DRAIN_CHUNK_BYTES = 16_384
_STDIN_ERRORS = (BrokenPipeError, ConnectionResetError)


async def spawn_hook(
    spec: HookSpec,
    request: Mapping[str, JsonValue],
    *,
    kill_group: ProcessSignaler,
) -> HookOutcome:
    """把 ``request`` 交给 command hook,返回一次调用的结算事实。

    Parameters:
        spec: 已校验的 command hook(``argv`` 绝对、``directory`` 绝对)。
        request: HookRequest 的 JSON 兼容字典;原样序列化进 stdin。
        kill_group: 进程组信号器(composition 绑 ``signal_group``);
            adapters 不得 import tools,故只能注入。

    Returns:
        ``HookOutcome``:``allowed``/``blocked``/``timeout``/``error``。
        除外层取消(``CancelledError``)外不向外抛异常——hook 故障必须
        由调用方映射为 fail-closed 裁决,而不是弄垮 Turn。
    """
    try:
        payload = json.dumps(thaw_json(request), ensure_ascii=False).encode("utf-8") + b"\n"
    except Exception:  # noqa: BLE001 — 序列化失败是 hook 错误,不是 Turn 崩溃
        return HookOutcome("error")

    try:
        proc = await asyncio.create_subprocess_exec(
            *spec.argv,
            cwd=spec.directory,
            env={"PATH": os.defpath},
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
    except OSError:
        return HookOutcome("error")

    stdin = proc.stdin
    assert stdin is not None and proc.stdout is not None
    sender = asyncio.create_task(_send_request(stdin, payload))
    try:
        return await asyncio.wait_for(_interact(proc, sender), spec.timeout_seconds)
    except TimeoutError:
        return HookOutcome("timeout")
    except Exception:  # noqa: BLE001 — 宿主 hook 故障必须转 error,绝不弄垮 Turn
        return HookOutcome("error")
    finally:
        sender.cancel()
        try:
            await asyncio.gather(sender, return_exceptions=True)
        finally:
            _kill_if_running(proc, kill_group)
            await _drain_bounded(proc)


async def _send_request(stdin: asyncio.StreamWriter, payload: bytes) -> None:
    """写完整 HookRequest 后关闭 stdin(hook 用 EOF 判定输入结束)。"""
    try:
        stdin.write(payload)
        await stdin.drain()
    except _STDIN_ERRORS:
        pass
    finally:
        stdin.close()


async def _interact(
    proc: asyncio.subprocess.Process, sender: asyncio.Task[None]
) -> HookOutcome:
    """读 stdout、等退出、解析裁决;超出整次调用时限由外层 wait_for 取消。"""
    stdout = proc.stdout
    assert stdout is not None
    try:
        # readexactly 强制总量上限:read(n) 会提前返回,不能当上限用。
        await stdout.readexactly(MAX_HOOK_STDOUT_BYTES)
    except asyncio.IncompleteReadError as end:
        output = end.partial
    else:
        return HookOutcome("error")  # 16_385 字节可读 ⇒ 超过 16 KiB 上限
    await sender
    if await proc.wait() != 0:
        return HookOutcome("error")
    return _parse_decision(output)


def _parse_decision(output: bytes) -> HookOutcome:
    """空 stdout ⇒ allowed;严格 ``{block, feedback}`` 形状;其余一律 error。"""
    if not output.strip():
        return HookOutcome("allowed")
    try:
        result = json.loads(output)
    except (TypeError, ValueError):
        return HookOutcome("error")
    if not isinstance(result, dict) or set(result) - _DECISION_KEYS:
        return HookOutcome("error")
    block = result.get("block", False)
    feedback = result.get("feedback", "")
    if type(block) is not bool or not isinstance(feedback, str):
        return HookOutcome("error")
    status: HookStatus = "blocked" if block else "allowed"
    return HookOutcome(status, feedback)


def _kill_if_running(
    proc: asyncio.subprocess.Process, kill_group: ProcessSignaler
) -> None:
    """超时/取消/异常路径:主进程未回收或 stdout 未 EOF ⇒ 整组 SIGKILL。"""
    stdout = proc.stdout
    if proc.returncode is None or (stdout is not None and not stdout.at_eof()):
        kill_group(proc.pid, signal.SIGKILL)


async def _drain_bounded(proc: asyncio.subprocess.Process) -> None:
    """0.5s 有界排空:组外孙进程攥着管道时绝不让 Turn 等 EOF。"""
    stdout = proc.stdout
    try:
        async with asyncio.timeout(_DRAIN_TIMEOUT_SECONDS):
            if stdout is not None:
                while await stdout.read(_DRAIN_CHUNK_BYTES):
                    pass
            await proc.wait()
    except (TimeoutError, OSError):
        pass


__all__ = ["spawn_hook"]