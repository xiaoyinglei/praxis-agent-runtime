"""Task 18 Step 2:真实模型流式冒烟（**默认跳过,需用户显式授权**）。

授权点（两件,缺一不跑）：
1. ``PRAXIS_LIVE_SMOKE=1`` —— 明确同意向真实 provider 发请求（消费 token）；
2. ``PRAXIS_LIVE_MODEL_ID=<已注册模型 id>`` —— 指定注册库中的模型（凭据按
   ``api_key_env`` 经 ``resolve_model_credential`` 解析:环境变量优先,其次
   Keychain/0600 store;Ollama 可无凭据）。

届时命令（主工作区已有注册与凭据时）：
    PRAXIS_LIVE_SMOKE=1 PRAXIS_LIVE_MODEL_ID=<id> \
      uv run pytest tests/praxis/acceptance/test_model_stream_live.py -v -s

断言口径（brief Step 2）：
- 真增量：首个 ``Model*Delta`` 的到达时刻严格早于流结束时刻（证明不是伪流式
  缓冲后一次性吐出）；
- ``usage.source == "provider"``（provider 未回 usage 时实现会标 estimated）；
- ``measure_request``（praxis 唯一估算入口）与 provider ``input_tokens`` 偏差
  < 25%:仅对 ``openai-compatible`` 作硬门；``ollama`` 属"本地模型",偏差只打印
  记录进执行文档 §10,不作硬门；
- ``model_attempt``:无故障时每个模型请求恰产生 **1 条** ``kind="ok"`` 记录
  （重试机制只会追加更多记录;多工具 turn 的第二次请求另有一条,逐请求唯一）。

本文件在默认门禁里整体 skip,不产生网络请求。
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from praxis.adapters.config.credentials import resolve_model_credential
from praxis.adapters.config.store import ModelConfigStore
from praxis.adapters.models.tokenization import measure_request
from praxis.composition import model_adapter_for
from praxis.interfaces.cli.main import app
from praxis.interfaces.sdk import open_session
from praxis.protocol.items import ItemKind
from praxis.protocol.models import (
    ModelMessage,
    ModelReasoningDelta,
    ModelSettings,
    ModelStreamFinished,
    ModelTextDelta,
    ModelToolCallDelta,
    assemble_model_events,
)
from praxis.protocol.policy import NetworkPolicy, PermissionPolicy
from praxis.protocol.tools import ToolDefinition
from praxis.tools.builtins.filesystem import create_read_tool

pytestmark = pytest.mark.skipif(
    os.environ.get("PRAXIS_LIVE_SMOKE") != "1",
    reason="live provider requires explicit user authorization",
)

_DELTA_TYPES = (ModelTextDelta, ModelReasoningDelta, ModelToolCallDelta)
_SYSTEM = "You are a terse workspace agent. Use tools when asked."
_TURN_PROMPT = (
    "Read a.txt with the read tool, then read b.txt with the read tool, "
    "then answer with one short sentence summarizing both files. You must call tools."
)


def _model_id() -> str:
    model_id = os.environ.get("PRAXIS_LIVE_MODEL_ID")
    if not model_id:
        pytest.skip("set PRAXIS_LIVE_MODEL_ID to an id registered in the model store")
    return model_id


def _spec_and_adapter(model_id: str) -> tuple[Any, Any]:
    store = ModelConfigStore()
    try:
        spec = store.resolve_spec(model_id)
    finally:
        store.close()
    api_key = (
        resolve_model_credential(spec.api_key_env, spec.base_url or "")
        if spec.api_key_env is not None
        else None
    )
    return spec, model_adapter_for(spec, api_key)


def test_models_check_live() -> None:
    # 入口探针:CLI 的 models check 走真实 provider(文本步 + 支持工具时的工具步)。
    model_id = _model_id()
    result = CliRunner().invoke(app, ["models", "check", model_id])
    assert result.exit_code == 0, f"models check failed:\n{result.output}\n{result.exception!r}"
    assert f"ok: {model_id}" in result.output
    # usage 行给出 provider/estimated 归因;真实远端应报告 provider。
    assert "(provider)" in result.output, result.output


def test_live_stream_is_incremental_and_usage_is_provider(tmp_path: Path) -> None:
    model_id = _model_id()
    spec, adapter = _spec_and_adapter(model_id)
    read_tool = create_read_tool(tmp_path)
    definitions: list[ToolDefinition] = [read_tool.definition]
    request = adapter.build_request(
        [
            ModelMessage(role="system", content=_SYSTEM),
            ModelMessage(role="user", content="Reply with the single word: ready"),
        ],
        definitions,
        ModelSettings(),
    )
    timings: list[tuple[float, Any]] = []

    async def collect() -> list[Any]:
        events: list[Any] = []
        async for event in adapter.stream(request):
            timings.append((time.monotonic(), event))
            events.append(event)
        return events

    events = asyncio.run(collect())
    response = assemble_model_events(events)
    delta_times = [at for at, event in timings if isinstance(event, _DELTA_TYPES)]
    finished_times = [at for at, event in timings if isinstance(event, ModelStreamFinished)]
    assert delta_times, "no incremental delta event observed"
    assert finished_times, "no ModelStreamFinished event observed"
    assert delta_times[0] < finished_times[-1], "stream was buffered, not incremental"
    assert response.usage.source == "provider", (
        f"provider did not report usage (source={response.usage.source!r})"
    )

    predicted = measure_request(request).count
    observed = response.usage.input_tokens
    if observed > 0:
        deviation = abs(predicted - observed) / observed
        if spec.provider == "openai-compatible":
            # brief 的硬门:估算与 provider prompt_tokens 偏差 < 25%。
            assert deviation < 0.25, (
                f"measure_request deviates {deviation:.1%} from provider input_tokens"
            )
        else:
            # 本地模型(ollama 等)偏差只记录,不作硬门(执行文档 §10)。
            print(f"local-model token deviation: predicted={predicted} provider={observed} "
                  f"({deviation:.1%})")


def test_live_multitool_turn_records_one_ok_attempt_per_request(tmp_path: Path) -> None:
    model_id = _model_id()
    _spec, adapter = _spec_and_adapter(model_id)
    (tmp_path / "a.txt").write_text("alpha\n", encoding="utf-8")
    (tmp_path / "b.txt").write_text("beta\n", encoding="utf-8")

    async def run_turn() -> Any:
        session = open_session(
            tmp_path,
            adapter,
            thread_id="live-smoke",
            policy=PermissionPolicy(network=NetworkPolicy(decision="allow")),
        )
        try:
            return await session.submit(_TURN_PROMPT), session.items()
        finally:
            session.close()

    result, items = asyncio.run(run_turn())
    assert result.status.value == "completed", (result.status, result.reason)
    calls = [i for i in items if i.kind == ItemKind.TOOL_CALL]
    assert len(calls) >= 2, f"expected a real multi-tool turn, got calls={calls!r}"
    assert {c.payload.get("tool_name") for c in calls} >= {"read"}

    responses = [i for i in items if i.kind == ItemKind.MODEL_RESPONSE]
    attempts = [i for i in items if i.kind == ItemKind.MODEL_ATTEMPT]
    assert responses, "turn produced no model_response items"
    # 无故障:每个模型请求恰一条 ok 尝试;多工具 turn 的后续请求各算一条,
    # 不存在同请求的重复/重试记录。
    assert len(attempts) == len(responses), (
        f"expected one attempt per request; attempts={[a.payload for a in attempts]!r}"
    )
    for attempt in attempts:
        assert attempt.payload.get("kind") == "ok", attempt.payload
        assert attempt.payload.get("retry_at") is None, attempt.payload