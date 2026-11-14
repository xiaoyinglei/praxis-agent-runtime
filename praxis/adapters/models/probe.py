"""统一构造器探针:两步都走 ``adapter.build_request`` + 唯一失败分类器(B6 消亡)。

旧旁路(B6 的确切死因)是探针自己拼 SDK 参数:``max_output_tokens`` 这种 SDK
不认识的键被原样塞进 ``chat.completions.create``,于是"探针失败"与"真实 turn
失败"是两套事实。本模块只做两件事:

1. **文本步**:``build_request([user("reply with the word ok")], ())`` → ``complete``;
2. **工具步**(``spec.supports_tools`` 为真时):带一个真实 ``ToolDefinition``
   走**同一** ``build_request`` → 能装配出 tool_calls 或模型自然拒答都算 ok。

失败分两阶段:

- **本地配置期**(``build_request`` 抛 ``ValueError``,如 wire-owned
  ``provider_options``、无工具能力却带工具、坏 ``base_url``):不喂给 provider
  分类器,detail 明确写 "local model configuration error"——运维校验配置时不会
  被指向"provider 协议违规"。
- **provider 期**(``complete`` 抛出的任何异常):经
  ``adapters/models/retry.classify_failure``(唯一分类器)归一,detail =
  步骤名 + 固定文案 + kind(+有界 diagnostic)。凭据类失败按调用方传入的
  ``credential_source``(env/store/missing)归因;缺省 None 时才回落旧的
  env 存在性判断(只读存在/非空,绝不读值)。绝不出现旧实现的
  "check endpoint, model, and credential environment" 这种误导归因。

探针不进 turn 循环,也不写任何 item。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from praxis.adapters.config.credentials import CredentialSource
from praxis.adapters.models.retry import classify_failure
from praxis.protocol.models import (
    DROPPED_TOOL_ARGUMENTS_MARKER,
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


class _LocalConfigError(Exception):
    """``build_request`` 期的本地配置错误;在进 provider 分类器之前分流。"""


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


def _credential_root(env: str) -> str:
    """兼容旧调用方(未传 source)的 env-only 归因;值本身绝不读出/输出。"""
    if not os.environ.get(env):
        return f"credential environment {env} is not set"
    return f"credential from {env} was rejected by the provider"


def _named(env: str | None) -> str:
    return f" for {env}" if env else ""


def _detail(
    step: str,
    failure: ModelFailure,
    spec: ModelSpec | None,
    source: CredentialSource | None = None,
) -> str:
    """把分类器事实变成可行动的一句话;凭据失败按来源归因(R-T6-8)。

    ``source`` 由调用方给出(env/store/missing);``None`` 保留旧的 env-only
    推导,仅用于没有注册表上下文的直接探针调用。
    """
    if failure.kind in ("authentication", "permission"):
        env = spec.api_key_env if spec is not None else None
        if source == "env":
            root = f"credential from {env} was rejected by the provider" if env else (
                "credential was rejected by the provider"
            )
        elif source == "store":
            root = f"saved credential{_named(env)} was rejected by the provider"
        elif source == "missing":
            root = f"no credential available{_named(env)}"
        elif env:
            root = _credential_root(env)
        else:
            root = "credential was rejected by the provider"
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


async def _build_and_complete(
    adapter: ModelAdapter,
    *,
    prompt: str,
    tools: tuple[ToolDefinition, ...],
) -> ModelResponse:
    """两步共用的唯一路径:本地构造期与 provider 期在此边界明确分开。"""
    try:
        request = adapter.build_request(
            [ModelMessage(role="user", content=prompt)], tools, ModelSettings()
        )
    except ValueError as exc:  # 本地配置错误(含 ToolsNotSupportedError)
        raise _LocalConfigError(str(exc)) from exc
    return await adapter.complete(request)


async def probe_model(
    adapter: ModelAdapter,
    *,
    tool_name: str = "__probe_noop",
    credential_source: CredentialSource | None = None,
) -> ProbeOutcome:
    """探测一个已装配的模型适配器;失败永不抛穿,而是进 ``detail``。

    ``credential_source`` 是调用方对凭据来源的归因(env/store/missing);
    缺省 None 时保留旧的 env-only 文案。
    """
    spec = _spec_of(adapter)
    steps: list[str] = []

    steps.append(_TEXT_STEP)
    try:
        text_response = await _build_and_complete(adapter, prompt=_TEXT_PROMPT, tools=())
    except _LocalConfigError as exc:
        return ProbeOutcome(
            ok=False,
            steps=tuple(steps),
            detail=f"{_TEXT_STEP} step failed: local model configuration error: {exc}",
            usage=None,
        )
    except Exception as exc:  # noqa: BLE001 — 所有 provider 失败都归一进 detail
        return ProbeOutcome(
            ok=False,
            steps=tuple(steps),
            detail=_detail(_TEXT_STEP, classify_failure(exc), spec, credential_source),
            usage=None,
        )
    usage = text_response.usage

    if spec is not None and spec.supports_tools:
        steps.append(_TOOL_STEP)
        try:
            tool_response = await _build_and_complete(
                adapter,
                prompt=_TOOL_PROMPT.format(name=tool_name),
                tools=(_probe_tool(tool_name),),
            )
        except _LocalConfigError as exc:
            return ProbeOutcome(
                ok=False,
                steps=tuple(steps),
                detail=f"{_TOOL_STEP} step failed: local model configuration error: {exc}",
                usage=usage,
            )
        except Exception as exc:  # noqa: BLE001 — 同上
            return ProbeOutcome(
                ok=False,
                steps=tuple(steps),
                detail=_detail(_TOOL_STEP, classify_failure(exc), spec, credential_source),
                usage=usage,
            )
        if DROPPED_TOOL_ARGUMENTS_MARKER in tool_response.content:
            # 装配器丢弃了畸形 tool call;这不是"自然拒答",而是工具协议失败。
            return ProbeOutcome(
                ok=False,
                steps=tuple(steps),
                detail=(
                    f"{_TOOL_STEP} step failed: "
                    "provider returned malformed tool call arguments"
                ),
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
