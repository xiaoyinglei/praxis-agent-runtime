"""Task 18 Step 2:真实模型流式冒烟（**默认跳过,需用户显式授权**）。

授权点（两件,缺一不跑）：
1. ``PRAXIS_LIVE_SMOKE=1`` —— 明确同意向真实 provider 发请求（消费 token）；
2. ``PRAXIS_LIVE_MODEL_ID=<已注册模型 id>`` —— 指定注册库中的模型（凭据按
   ``api_key_env`` 经 ``resolve_model_credential`` 解析:环境变量优先,其次
   Keychain/0600 store;Ollama 可无凭据）。

届时命令（主工作区已有注册与凭据时）：
    PRAXIS_LIVE_SMOKE=1 PRAXIS_LIVE_MODEL_ID=<id> \
      uv run pytest tests/praxis/acceptance/test_model_stream_live.py -v -s

断言口径（M-3 起）：
- 透传保真（provider 无关）:用 ``ObservingHttpClient`` 记录底层 SSE/NDJSON 帧
  可读的时刻,断言 adapter 产出的 delta 相对这些帧 (a) 时延 <= 250ms、
  (b) 数量不被合并、(c) 不早于对应帧。判据实现在 ``acceptance/passthrough.py``,
  由 ``test_model_stream_passthrough_offline.py`` 用 MockTransport 同时给出
  GREEN（成批下发不成问题）与 RED（adapter 外多 300ms 必被抓）证据。
  **已删除**旧的"首个 delta 落在总时长前 50%"判据:它测的是供应商的吐帧节奏
  (端点先等 ~1.8s 再成批下发 12 帧,该比例恒等于 ~1),无法保护"praxis 不额外
  缓冲"这条我们真正拥有的性质。客户端不缓冲的确定性结构证明仍在
  ``test_model_stream.py`` 的 MockTransport 用例与
  ``test_model_stream_incremental_offline.py``(M-2)中;
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

import httpx
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
    ModelSettings,
    ModelTextDelta,
    assemble_model_events,
)
from praxis.protocol.policy import NetworkPolicy, PermissionPolicy
from praxis.protocol.tools import ToolDefinition
from praxis.tools.builtins.filesystem import create_read_tool
from tests.praxis.acceptance.passthrough import (
    FrameRecorder,
    ObservingHttpClient,
    passthrough_fidelity_problem,
)

pytestmark = pytest.mark.skipif(
    os.environ.get("PRAXIS_LIVE_SMOKE") != "1",
    reason="live provider requires explicit user authorization",
)

_SYSTEM = "You are a terse workspace agent. Use tools when asked."
# 增量性判据要求至少两帧;单字回答("ready")本来就只会来一帧,因此提示词
# 必须要求必然多 token 的输出(M-2:此前用单字提示词,把测试自己写坏了)。
_INCREMENTAL_PROMPT = "Output the numbers 1 to 20, one per line, with no explanation."
_TURN_PROMPT = (
    "Read a.txt with the read tool, then read b.txt with the read tool, "
    "then answer with one short sentence summarizing both files. You must call tools."
)


def _model_id() -> str:
    model_id = os.environ.get("PRAXIS_LIVE_MODEL_ID")
    if not model_id:
        pytest.skip("set PRAXIS_LIVE_MODEL_ID to an id registered in the model store")
    # N-6:默认套件把 PRAXIS_CONFIG_DIR 隔离到 per-test 空目录;授权 live 冒烟
    # 必须由运维显式给出注册库(fixture 会原样尊重),否则这里只会看到空库。
    from tests.praxis.conftest import _ISOLATION_DECISIONS  # noqa: PLC0415

    if not _ISOLATION_DECISIONS.get("reused_operator_env", False):
        pytest.skip(
            "live smoke requires an operator-provided PRAXIS_CONFIG_DIR "
            "(test isolation would otherwise hide the model registry)"
        )
    return model_id


def _resolve_credentialed_spec(model_id: str) -> tuple[Any, str | None]:
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
    return spec, api_key


def _spec_and_adapter(
    model_id: str,
    *,
    http_client: httpx.AsyncClient | None = None,
) -> tuple[Any, Any]:
    spec, api_key = _resolve_credentialed_spec(model_id)
    return spec, model_adapter_for(spec, api_key, http_client=http_client)


def test_models_check_live() -> None:
    # 入口探针:CLI 的 models check 走真实 provider(文本步 + 支持工具时的工具步)。
    model_id = _model_id()
    result = CliRunner().invoke(app, ["models", "check", model_id])
    assert result.exit_code == 0, f"models check failed:\n{result.output}\n{result.exception!r}"
    assert f"ok: {model_id}" in result.output
    # usage 行给出 provider/estimated 归因;真实远端应报告 provider。
    assert "(provider)" in result.output, result.output


def test_live_stream_is_incremental_and_usage_is_provider(tmp_path: Path) -> None:
    # 底层帧可读时刻由 ObservingHttpClient 记录;delta 时刻在本函数里记录。
    # 判据只比较这两条时间线,provider 等 1.8s 再成批下发 12 帧也不影响结论。
    model_id = _model_id()
    spec, api_key = _resolve_credentialed_spec(model_id)
    recorder = FrameRecorder()
    # 不显式给 transport:保留 AsyncClient 默认的 transport(含 trust_env 代理配置),
    # 只额外在 aiter_bytes 处打点;timeout 沿用注册配置,避免改变 live 冒烟行为。
    observing_client = ObservingHttpClient(recorder, timeout=spec.timeout_seconds)
    adapter = model_adapter_for(spec, api_key, http_client=observing_client)
    read_tool = create_read_tool(tmp_path)
    definitions: list[ToolDefinition] = [read_tool.definition]
    request = adapter.build_request(
        [
            ModelMessage(role="system", content=_SYSTEM),
            ModelMessage(role="user", content=_INCREMENTAL_PROMPT),
        ],
        definitions,
        ModelSettings(),
    )

    async def collect() -> tuple[list[Any], list[float]]:
        events: list[Any] = []
        delta_times: list[float] = []
        async for event in adapter.stream(request):
            if isinstance(event, ModelTextDelta):
                delta_times.append(time.monotonic())
            events.append(event)
        return events, delta_times

    events, delta_times = asyncio.run(collect())
    response = assemble_model_events(events)
    problem = passthrough_fidelity_problem(recorder, delta_times)
    assert problem is None, problem
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
