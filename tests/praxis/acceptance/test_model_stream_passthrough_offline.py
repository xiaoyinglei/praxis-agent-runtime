"""M-3:透传保真判据的离线双向证据(``httpx.MockTransport``,不触网)。

live 用例的判据已从"首个 delta 落在总时长前 50%"(测供应商吐帧节奏)换成
"adapter 不额外缓冲"(测 praxis 的性质)。本文件用确定性替身证明它**既不是
放宽**、也有鉴别力:

1. GREEN:复刻 controller 在真实端点实测的形状(长 TTFB ~0.4s,随后一帧内
   成批下发全部 SSE 帧)。同一形状下旧比例判据必然 RED(首帧落在总时长后
   50%),新判据必须 GREEN——因为没有客户端缓冲。
2. RED:在 adapter 之外包一层"每个事件前多等 300ms"的 wrapper,同一判据必须
   RED,且失败原因只能来自"客户端额外缓冲"。
3. 单帧 provider:如实报"无法判定透传保真",不 skip;相对顺序分支也有覆盖。
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Sequence
from typing import Any

import httpx

from praxis.adapters.models.openai_compat import OpenAICompatAdapter
from praxis.protocol.models import (
    ModelEvent,
    ModelMessage,
    ModelSettings,
    ModelSpec,
    ModelStreamFinished,
    ModelTextDelta,
    assemble_model_events,
)
from tests.praxis.acceptance.incrementality import incrementality_problem
from tests.praxis.acceptance.passthrough import (
    FrameRecorder,
    ObservedFrame,
    ObservingHttpClient,
    passthrough_fidelity_problem,
)

_CHUNK_ID = "chatcmpl-passthrough"
# controller 2026-11-15 在真实端点实测:首个 content 帧在 ~1.8s,12 帧在 0.27s
# 内成批到达。离线替身保留"长 TTFB + 成批"的形状(缩短到 0.4s 控制测试时长)。
_TTFB_SECONDS = 0.4
_CONTENT_FRAMES = 10
_PROMPT = "Output the numbers 1 to 20, one per line, with no explanation."


def _sse_chunk(delta: dict[str, Any], *, finish: str | None = None,
               usage: dict[str, int] | None = None) -> bytes:
    entry: dict[str, Any] = {
        "id": _CHUNK_ID,
        "object": "chat.completion.chunk",
        "created": 0,
        "model": "m",
        "choices": [],
    }
    if usage is not None:
        entry["usage"] = usage
    else:
        entry["choices"] = [{"index": 0, "delta": delta, "finish_reason": finish}]
    return f"data: {json.dumps(entry)}\n\n".encode()


def _frames() -> list[bytes]:
    frames = [
        _sse_chunk({"content": f"line {index}: batched frame\n"})
        for index in range(_CONTENT_FRAMES)
    ]
    frames.append(_sse_chunk({}, finish="stop"))
    frames.append(
        _sse_chunk({}, usage={"prompt_tokens": 12, "completion_tokens": 60, "total_tokens": 72})
    )
    frames.append(b"data: [DONE]\n\n")
    return frames


class _BatchedSseStream(httpx.AsyncByteStream):
    """长 TTFB 后一次性吐出全部帧(真实端点的成批形状)。"""

    def __init__(self, frames: Sequence[bytes]) -> None:
        self._frames = frames

    async def __aiter__(self) -> AsyncIterator[bytes]:
        await asyncio.sleep(_TTFB_SECONDS)
        yield b"".join(self._frames)


def _transport(frames: Sequence[bytes]) -> httpx.MockTransport:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_BatchedSseStream(frames),
        )

    return httpx.MockTransport(handler)


def _adapter(frames: Sequence[bytes], recorder: FrameRecorder) -> OpenAICompatAdapter:
    client = ObservingHttpClient(recorder, transport=_transport(frames))
    return OpenAICompatAdapter(
        ModelSpec(model_id="m", base_url="https://unit.test/v1"),
        api_key="k",
        http_client=client,
    )


class _ExtraDelayAdapter:
    """adapter 之外多一层 300ms 延迟:模拟客户端额外缓冲。"""

    def __init__(self, inner: OpenAICompatAdapter, delay: float = 0.3) -> None:
        self._inner = inner
        self._delay = delay

    def build_request(
        self,
        messages: Sequence[ModelMessage],
        tools: Sequence[Any],
        settings: ModelSettings,
    ) -> Any:
        return self._inner.build_request(messages, tools, settings)

    async def stream(self, req: Any) -> AsyncIterator[ModelEvent]:
        first = True
        async for event in self._inner.stream(req):
            if first:
                await asyncio.sleep(self._delay)
                first = False
            yield event


async def _timed(
    adapter: Any, request: Any
) -> tuple[float, list[tuple[float, ModelEvent]]]:
    started = time.monotonic()
    timed: list[tuple[float, ModelEvent]] = []
    async for event in adapter.stream(request):
        timed.append((time.monotonic(), event))
    return started, timed


def _delta_and_finish(
    timed: Sequence[tuple[float, ModelEvent]],
) -> tuple[list[float], float]:
    delta_times = [at for at, event in timed if isinstance(event, ModelTextDelta)]
    finished_at = next(at for at, event in timed if isinstance(event, ModelStreamFinished))
    return delta_times, finished_at


def test_offline_batched_provider_frames_are_not_client_buffered() -> None:
    """成批下发 + 长 TTFB ⇒ 新判据 GREEN;同形状下旧比例判据仍 RED。"""
    recorder = FrameRecorder()
    adapter = _adapter(_frames(), recorder)
    request = adapter.build_request(
        [ModelMessage(role="user", content=_PROMPT)], [], ModelSettings()
    )
    started, timed = asyncio.run(_timed(adapter, request))
    delta_times, finished_at = _delta_and_finish(timed)

    problem = passthrough_fidelity_problem(recorder, delta_times)
    assert problem is None, problem

    # 换测对象不是放宽:同一形状下旧的"首个 delta 在前 50%"判据仍然失败,
    # 因为首帧本来就在 ~TTFB 处才可读。
    ratio_problem = incrementality_problem(
        content_length=len(assemble_model_events([event for _, event in timed]).content),
        delta_times=delta_times,
        started=started,
        finished_at=finished_at,
    )
    assert ratio_problem is not None, "batched shape must still fail the old ratio criterion"
    print(f"\nGREEN: passthrough ok; old ratio criterion says: {ratio_problem}")


def test_offline_extra_client_delay_is_detected() -> None:
    """adapter 之外多 300ms ⇒ 同一判据必须 RED(且只因额外缓冲)。"""
    recorder = FrameRecorder()
    adapter = _ExtraDelayAdapter(_adapter(_frames(), recorder))
    request = adapter.build_request(
        [ModelMessage(role="user", content=_PROMPT)], [], ModelSettings()
    )
    _, timed = asyncio.run(_timed(adapter, request))
    delta_times, _ = _delta_and_finish(timed)

    problem = passthrough_fidelity_problem(recorder, delta_times)
    assert problem is not None, "a 300ms client-side delay must be detected"
    assert "额外缓冲" in problem, problem
    print(f"\nRED: {problem}")


def test_single_frame_provider_is_reported_not_skipped() -> None:
    """provider 合并成单帧:如实给出"无法判定",而不是悄悄跳过。"""
    recorder = FrameRecorder()
    recorder.frames.append(
        ObservedFrame(at=1.0, line=_sse_chunk({"content": "only frame"}).strip())
    )
    problem = passthrough_fidelity_problem(recorder, [1.001])
    assert problem is not None
    assert "provider 单帧返回" in problem, problem
    assert "无法判定透传保真" in problem, problem


def test_relative_order_violation_is_detected() -> None:
    """delta 早于对应 content 帧属于不可能时序,必须报出。"""
    recorder = FrameRecorder()
    recorder.frames.append(
        ObservedFrame(at=1.0, line=_sse_chunk({"content": "a"}).strip())
    )
    recorder.frames.append(
        ObservedFrame(at=1.1, line=_sse_chunk({"content": "b"}).strip())
    )
    problem = passthrough_fidelity_problem(recorder, [0.9, 1.2])
    assert problem is not None
    assert "早于对应 content 帧" in problem, problem