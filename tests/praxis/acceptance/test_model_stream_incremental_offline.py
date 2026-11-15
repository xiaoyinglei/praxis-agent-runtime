"""M-2:增量性判据的离线可失败证明(``httpx.MockTransport``,不触网)。

live 用例 ``test_model_stream_live.py::test_live_stream_is_incremental_and_usage_is_provider``
的失败曾把"单字提示词只会来一帧"误判成流式缺陷。本文件用确定性替身证明新判据
(SUT 与 live 共用 ``incrementality.incrementality_problem``)真的有鉴别力:

1. 逐帧产出形状(真实 ``OpenAICompatAdapter`` + 慢速 SSE 流)⇒ 判据 GREEN;
2. 假装全缓冲形状(同一底层流,但把全部事件攒到最后一帧之后才一次性产出)⇒
   判据必须 RED,且失败原因只能来自"首个 delta 落在后 50%"(内容与帧数都满足);
3. 两种失败文案可区分:内容太短 vs 单帧到达。
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
    ModelMessage,
    ModelSettings,
    ModelSpec,
    ModelStreamFinished,
    ModelTextDelta,
    assemble_model_events,
)
from tests.praxis.acceptance.incrementality import (
    MIN_CONTENT_CHARS,
    incrementality_problem,
)

_CHUNK_ID = "chatcmpl-incremental"
# 帧间隔要显著大于客户端建连/SDK 首帧开销(实测 ~100ms),否则定时抖动会把
# 逐帧形状误判为"后 50%"。10 帧 × 50ms 让首帧比例稳定在 ~0.2。
_FRAME_DELAY_SECONDS = 0.05
_TEXT_FRAMES = 10
_FINISH_USAGE = {"prompt_tokens": 12, "completion_tokens": 60, "total_tokens": 72}
_PROMPT = "Output the numbers 1 to 20, one per line, with no explanation."


def _sse_chunk(
    delta: dict[str, Any],
    *,
    finish: str | None = None,
    usage: dict[str, int] | None = None,
) -> bytes:
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


def _sse_frames() -> list[bytes]:
    frames = [
        _sse_chunk({"content": f"line {index}: incremental frame\n"})
        for index in range(_TEXT_FRAMES)
    ]
    frames.append(_sse_chunk({}, finish="stop"))
    frames.append(_sse_chunk({}, usage=_FINISH_USAGE))
    frames.append(b"data: [DONE]\n\n")
    return frames


class _SlowSseStream(httpx.AsyncByteStream):
    """按真实 provider 的节奏逐帧吐出字节:帧与帧之间有可观测的间隔。"""

    def __init__(self, frames: Sequence[bytes]) -> None:
        self._frames = frames

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for frame in self._frames:
            await asyncio.sleep(_FRAME_DELAY_SECONDS)
            yield frame


def _transport(frames: Sequence[bytes]) -> httpx.MockTransport:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=_SlowSseStream(frames),
        )

    return httpx.MockTransport(handler)


def _real_adapter(frames: Sequence[bytes]) -> OpenAICompatAdapter:
    client = httpx.AsyncClient(transport=_transport(frames))
    return OpenAICompatAdapter(
        ModelSpec(model_id="m", base_url="https://unit.test/v1"),
        api_key="k",
        http_client=client,
    )


class _FullBufferAdapter:
    """假装全缓冲的 adapter 形状:先收完底层全部事件,再一次性产出。

    这是 live 比例判据要抓的实现形状——首帧与 finish 几乎同时到达,
    ``delta[0] < finished[-1]``(旧断言)依然成立,只有比例判据会失败。
    """

    def __init__(self, inner: OpenAICompatAdapter) -> None:
        self._inner = inner

    def build_request(
        self,
        messages: Sequence[ModelMessage],
        tools: Sequence[Any],
        settings: ModelSettings,
    ) -> Any:
        return self._inner.build_request(messages, tools, settings)

    async def stream(self, req: Any) -> AsyncIterator[Any]:
        buffered = [event async for event in self._inner.stream(req)]
        for event in buffered:
            yield event


async def _timed_events(adapter: Any, req: Any) -> tuple[float, list[tuple[float, Any]]]:
    started = time.monotonic()
    timed: list[tuple[float, Any]] = []
    async for event in adapter.stream(req):
        timed.append((time.monotonic(), event))
    return started, timed


def _evaluate(
    started: float, timed: Sequence[tuple[float, Any]]
) -> tuple[str, int, str | None]:
    response = assemble_model_events([event for _, event in timed])
    delta_times = [at for at, event in timed if isinstance(event, ModelTextDelta)]
    finished_at = next(
        at for at, event in timed if isinstance(event, ModelStreamFinished)
    )
    problem = incrementality_problem(
        content_length=len(response.content),
        delta_times=delta_times,
        started=started,
        finished_at=finished_at,
    )
    return response.content, len(delta_times), problem


def test_offline_incremental_stream_passes_criterion() -> None:
    """逐帧产出 ⇒ GREEN:首帧在总时长的前 50% 内到达。"""
    adapter = _real_adapter(_sse_frames())
    request = adapter.build_request([ModelMessage(role="user", content=_PROMPT)], [], ModelSettings())
    started, timed = asyncio.run(_timed_events(adapter, request))
    content, delta_count, problem = _evaluate(started, timed)
    assert len(content) >= MIN_CONTENT_CHARS, content
    assert delta_count >= 2, delta_count
    assert problem is None, problem


def test_offline_full_buffer_adapter_fails_ratio_criterion() -> None:
    """全缓冲产出 ⇒ 判据必须 RED,且只因比例判据失败(内容/帧数都够)。"""
    adapter = _FullBufferAdapter(_real_adapter(_sse_frames()))
    request = adapter.build_request([ModelMessage(role="user", content=_PROMPT)], [], ModelSettings())
    started, timed = asyncio.run(_timed_events(adapter, request))
    content, delta_count, problem = _evaluate(started, timed)
    assert len(content) >= MIN_CONTENT_CHARS, content
    assert delta_count >= 2, delta_count
    assert problem is not None, "full-buffer shape must fail the incrementality criterion"
    assert "first delta arrived after" in problem, problem


def test_incrementality_problem_distinguishes_failure_causes() -> None:
    """内容太短 vs 单帧到达:文案必须可区分,避免下次再误判。"""
    short = incrementality_problem(
        content_length=5, delta_times=[0.1, 0.2], started=0.0, finished_at=1.0
    )
    assert short is not None and "内容太短" in short and "更长的提示词" in short, short
    single = incrementality_problem(
        content_length=100, delta_times=[0.1], started=0.0, finished_at=1.0
    )
    assert single is not None and "单帧到达" in single, single
    late = incrementality_problem(
        content_length=100, delta_times=[0.9, 0.95], started=0.0, finished_at=1.0
    )
    assert late is not None and "first delta arrived after" in late, late