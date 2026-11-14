"""统一构造器探针:两步都走 ``adapter.build_request`` + 唯一失败分类器(B6 消亡)。

旧旁路(B6 的确切死因)是探针自己拼 SDK 参数:``max_output_tokens`` 这种 SDK
不认识的键被原样塞进 ``chat.completions.create``,于是"探针失败"与"真实 turn
失败"是两套事实。本模块只做两件事:

1. **文本步**:``build_request([user("reply with the word ok")], ())`` → ``complete``;
2. **工具步**(``spec.supports_tools`` 为真时):带一个真实 ``ToolDefinition``
   走**同一** ``build_request`` → 能装配出 tool_calls 或模型自然拒答都算 ok。

所有异常都经 ``adapters/models/retry.classify_failure`` 归一(唯一分类器),
detail = 步骤名 + 固定文案 + kind(+有界 diagnostic);凭据类失败精确到
``api_key_env``,绝不出现"check endpoint, model, and credential environment"
这类误导归因。探针不进 turn 循环,也不写任何 item。
"""

from __future__ import annotations

from dataclasses import dataclass

from praxis.adapters.models.retry import classify_failure
from praxis.protocol.models import (
    ModelAdapter,
    ModelMessage,
    ModelResponse,
    ModelSettings,
    ModelSpec,
    Usage,
)
from praxis.protocol.retry import ModelFailure
from praxis.protocol.tools import ToolDefinition

_TEXT_STEP = "text"
_TOOL_STEP = "tools"
_TEXT_PROMPT = "reply with the word ok"
_TOOL_PROMPT = (
    "Call the {name} tool with no arguments. "
    "If you cannot call tools, explain why in one sentence."
)


@dataclass(frozen=True, slots=True)
class ProbeOutcome:
    """一次探针结果;``steps`` 是已执行的步骤名(含失败的那一步,按执行顺序)。

    ``usage`` 是文本步的记账(探针的主要可负担性信号);文本步未跑通时恒为 None。
    """

    ok: bool
    steps: tuple[str, ...]
    detail: str | None
    usage: Usage | None


def _spec_of(adapter: ModelAdapter) -> ModelSpec | None:
    """适配器经公开 ``spec`` 属性暴露静态配置;没有该属性则只跑文本步。"""
    spec = getattr(adapter, "spec", None)
    return spec if isinstance(spec, ModelSpec) else None


def _detail(step: str, failure: ModelFailure, spec: ModelSpec | None) -> str:
    """把分类器事实变成可行动的一句话;凭据失败点名 ``api_key_env``。"""
    if failure.kind == "authentication":
        env = spec.api_key_env if spec is not None else None
        root = (
            f"credential environment {env} is not set"
            if env
            else "credential rejected by provider"
        )
    else:
        root = failure.message
    detail = f"{step} step failed: {root} ({failure.kind})"
    if failure.diagnostic:
        detail = f"{detail}: {failure.diagnostic}"
    return detail


def _probe_tool(tool_name: str) -> ToolDefinition:
    return ToolDefinition(
        name=tool_name,
        description="Praxis model probe no-op tool; it takes no arguments.",
        input_schema={"type": "object", "properties": {}},
    )


async def probe_model(
    adapter: ModelAdapter, *, tool_name: str = "__probe_noop"
) -> ProbeOutcome:
    """探测一个已装配的模型适配器;失败永不抛穿,而是进 ``detail``。"""
    spec = _spec_of(adapter)
    steps: list[str] = []

    steps.append(_TEXT_STEP)
    try:
        text_request = adapter.build_request(
            [ModelMessage(role="user", content=_TEXT_PROMPT)], (), ModelSettings()
        )
        text_response: ModelResponse = await adapter.complete(text_request)
    except Exception as exc:  # noqa: BLE001 — 所有 provider 失败都归一进 detail
        return ProbeOutcome(
            ok=False,
            steps=tuple(steps),
            detail=_detail(_TEXT_STEP, classify_failure(exc), spec),
            usage=None,
        )
    usage = text_response.usage

    if spec is not None and spec.supports_tools:
        steps.append(_TOOL_STEP)
        try:
            tool_request = adapter.build_request(
                [ModelMessage(role="user", content=_TOOL_PROMPT.format(name=tool_name))],
                (_probe_tool(tool_name),),
                ModelSettings(),
            )
            tool_response = await adapter.complete(tool_request)
        except Exception as exc:  # noqa: BLE001 — 同上
            return ProbeOutcome(
                ok=False,
                steps=tuple(steps),
                detail=_detail(_TOOL_STEP, classify_failure(exc), spec),
                usage=usage,
            )
        if not tool_response.tool_calls and not tool_response.content.strip():
            return ProbeOutcome(
                ok=False,
                steps=tuple(steps),
                detail=(
                    f"{_TOOL_STEP} step failed: "
                    "provider returned neither text nor tool calls"
                ),
                usage=usage,
            )

    return ProbeOutcome(ok=True, steps=tuple(steps), detail=None, usage=usage)


__all__ = ["ProbeOutcome", "probe_model"]