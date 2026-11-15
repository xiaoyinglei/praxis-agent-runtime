"""M-3:透传保真判据的唯一实现(provider 无关,测"客户端不额外缓冲")。

要测的对象不是"provider 多久吐一帧"(供应商行为,会随端点/负载变化),而是
"praxis 的 adapter 在底层帧可读之后,是否立即转发 delta"。因此时钟锚点放在
**底层帧可读**处,delta 时刻在 adapter 之外记录;两者之差就是"一帧解析 + 事件
转发"的客户端开销。

``ObservingHttpClient`` 在 ``response.aiter_bytes``(httpx 解码之后)打点,而不是
transport 的原始字节流:后者在 provider 启用 gzip 时会把压缩字节当帧切分,
产生伪帧。openai SDK 的 SSE decoder 与 ollama 的 ``aiter_lines`` 都从
``aiter_bytes`` 消费,所以这是两者共同的"帧可读"边界。
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

# 一帧 SSE 解析 + 转发为 ModelTextDelta 的实测开销在毫秒级;250ms 给足 CI
# 调度抖动,同时仍能抓住"adapter 之外多 300ms"的 RED 证伪 wrapper。
PASSTHROUGH_TOLERANCE_SECONDS = 0.25


@dataclass(frozen=True)
class ObservedFrame:
    """底层响应体里的一行(SSE event / NDJSON 帧)及其可读时刻。"""

    at: float
    line: bytes


class FrameRecorder:
    """记录底层响应体每一行可读的时刻;行即最小可观测帧单位。"""

    def __init__(self) -> None:
        self.frames: list[ObservedFrame] = []
        self._pending = b""

    def feed(self, chunk: bytes) -> None:
        self._pending += chunk
        while b"\n" in self._pending:
            line, self._pending = self._pending.split(b"\n", 1)
            line = line.rstrip(b"\r")
            # SSE 的事件间空行与 NDJSON 的空行不承载 payload,不算帧。
            if line:
                self.frames.append(ObservedFrame(at=time.monotonic(), line=line))


class ObservingHttpClient(httpx.AsyncClient):
    """``httpx.AsyncClient`` 子类:在 bytes 被上层取走时打点(仅流式响应)。"""

    def __init__(self, recorder: FrameRecorder, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.recorder = recorder

    async def send(
        self,
        request: httpx.Request,
        *,
        stream: bool = False,
        **kwargs: Any,
    ) -> httpx.Response:
        response = await super().send(request, stream=stream, **kwargs)
        if not stream:
            # 非流式响应已被读完,事后迭代不会产生新的可观测到达时刻。
            return response
        original = response.aiter_bytes

        async def tapped(chunk_size: int | None = None) -> AsyncIterator[bytes]:
            async for chunk in original(chunk_size):
                self.recorder.feed(chunk)
                yield chunk

        response.aiter_bytes = tapped  # type: ignore[method-assign]
        return response


def _payload(line: bytes) -> dict[str, Any] | None:
    """SSE ``data:`` 行或 NDJSON 行 → JSON 对象;非 payload 行返回 None。"""
    if line.startswith(b"data:"):
        raw = line[5:].strip()
        if raw == b"[DONE]":
            return None
    elif line.startswith(b"{"):
        raw = line
    else:
        return None
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _content_text(obj: dict[str, Any]) -> str | None:
    """OpenAI delta 或 Ollama message 里的非空 content。"""
    choices = obj.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            delta = first.get("delta")
            if isinstance(delta, dict):
                content = delta.get("content")
                if isinstance(content, str) and content:
                    return content
    message = obj.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str) and content:
            return content
    return None


def content_frame_times(frames: Sequence[ObservedFrame]) -> list[float]:
    """承载非空 content 的底层帧的可读时刻序列。"""
    times: list[float] = []
    for frame in frames:
        obj = _payload(frame.line)
        if obj is not None and _content_text(obj) is not None:
            times.append(frame.at)
    return times


def passthrough_fidelity_problem(
    recorder: FrameRecorder,
    delta_times: Sequence[float],
) -> str | None:
    """三条 provider 无关判据;返回可读失败原因,``None`` 表示透传保真。

    1. 透传时延有界:首个 delta 不晚于首个 **content 帧** 250ms。锚定 content
       帧而不是"第一帧",因为 role/keepalive 帧不承载 delta,拿它当时钟会把
       provider 的探路帧误判成客户端缓冲。
    2. 无合并:delta 数 == content 帧数(±1,允许首帧空 delta / 角色帧)。
    3. 相对顺序:第 i 个 delta 不早于第 i 个 content 帧。
    """
    frame_times = [frame.at for frame in recorder.frames]
    content_times = content_frame_times(recorder.frames)
    if not frame_times:
        return "传输层未观测到任何帧:无法判定透传保真(响应体形状不符或未流式返回)"
    if len(content_times) < 2:
        # 显式分支,不 skip:provider 合并成单帧时,透传保真没有可判定的对象。
        return (
            f"provider 单帧返回(content 帧={len(content_times)}, 总帧={len(frame_times)}),"
            "无法判定透传保真;这不能当作客户端缓冲的证据"
        )
    if len(delta_times) < 2:
        return f"adapter 只产出 {len(delta_times)} 个 text delta(要求 >= 2)"
    if abs(len(delta_times) - len(content_times)) > 1:
        return (
            f"delta 数与 content 帧数不符:{len(delta_times)} vs {len(content_times)}"
            "(> ±1;可能被合并或丢失)"
        )
    latency = delta_times[0] - content_times[0]
    if latency > PASSTHROUGH_TOLERANCE_SECONDS:
        return (
            f"首个 delta 比首个 content 帧晚 {latency * 1000:.0f}ms"
            f"(> {PASSTHROUGH_TOLERANCE_SECONDS * 1000:.0f}ms):客户端在帧到达后额外缓冲"
        )
    for index, (delta_at, frame_at) in enumerate(zip(delta_times, content_times, strict=False)):
        if delta_at < frame_at - 1e-6:
            return (
                f"第 {index} 个 delta 早于对应 content 帧 "
                f"{(frame_at - delta_at) * 1000:.1f}ms:时序不可能"
            )
    return None