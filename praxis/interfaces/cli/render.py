"""CLI 读端渲染:Item 日志 → 一行人读文本,fail-open,永不抛(B1 读端最后一环)。

kind 是开放集合(protocol/items.py):未知 kind 不报错、不丢弃,降级为
``[kind] payload keys: [...]`` 摘要;已知 kind 遇畸形 payload 同样降级。
log/status 等展示路径因此永远不会因一条脏记录报废整个会话。
"""

from __future__ import annotations

from collections.abc import Mapping

from praxis.interfaces.sdk import Item, ItemKind

_PREVIEW_LIMIT = 100


def render_item(item: Item) -> str:
    """单条 item 的安全渲染:任何输入都返回字符串,绝不向上抛。"""
    try:
        return f"{item.seq:>6}  {_dispatch(item)}"
    except Exception:  # noqa: BLE001 — B1 读端契约:渲染失败降级,不炸调用方
        return f"{item.seq:>6}  {_fallback(item)}"


def _fallback(item: Item) -> str:
    """未知 kind / 渲染异常的统一降级:kind + payload 键名摘要。"""
    try:
        kind = str(item.kind)
    except Exception:  # noqa: BLE001
        kind = "?"
    try:
        keys = sorted(str(key) for key in item.payload)
    except Exception:  # noqa: BLE001
        keys = []
    body = f"payload keys: [{', '.join(keys)}]" if keys else "(no payload keys)"
    return f"[{kind}] {body}"


def _dispatch(item: Item) -> str:
    payload = item.payload
    kind = item.kind
    if kind == ItemKind.USER_MESSAGE:
        return f"user: {_preview(_str_field(payload, 'text'))}"
    if kind == ItemKind.STEERING:
        return f"steer: {_preview(_str_field(payload, 'text'))}"
    if kind == ItemKind.MODEL_RESPONSE:
        return _render_model_response(payload)
    if kind == ItemKind.TOOL_CALL:
        return (
            f"tool_call {_str_field(payload, 'tool_name')} call={_str_field(payload, 'call_id')} "
            f"digest={_short_digest(payload)}"
        )
    if kind == ItemKind.TOOL_RESULT:
        return (
            f"tool_result {_str_field(payload, 'tool_name')} call={_str_field(payload, 'call_id')} "
            f"[{_str_field(payload, 'status', '?')}] {_preview(_result_text(payload))}"
        )
    if kind == ItemKind.TURN_STATUS:
        reason = _str_field(payload, "reason")
        tail = f" ({reason})" if reason else ""
        return f"turn {_str_field(payload, 'from', '?')}→{_str_field(payload, 'to', '?')}{tail}"
    if kind == ItemKind.APPROVAL_REQUEST:
        effects = _str_field(payload, "effects", "") or _joined_effects(payload)
        tail = f" effects={effects}" if effects else ""
        return (
            f"approval_request {_str_field(payload, 'tool_name')} "
            f"call={_str_field(payload, 'call_id')} digest={_short_digest(payload)}{tail}"
        )
    if kind == ItemKind.APPROVAL_DECISION:
        return (
            f"approval_decision call={_str_field(payload, 'call_id')} "
            f"verdict={_str_field(payload, 'verdict', '?')}"
        )
    if kind == ItemKind.SUMMARY:
        return f"summary: {_preview(_str_field(payload, 'text'))}"
    if kind == ItemKind.COMPACTION_BOUNDARY:
        low = payload.get("covered_from")
        high = payload.get("covered_to")
        if isinstance(low, int) and isinstance(high, int) and not isinstance(low, bool):
            return f"compaction_boundary covers seq {low}..{high}"
        return _fallback(item)
    return _fallback(item)


def _render_model_response(payload: Mapping[str, object]) -> str:
    content = _preview(_str_field(payload, "content")) or "(no text)"
    calls = _tool_call_names(payload)
    if calls:
        return f"model: {content} → calls: {', '.join(calls)}"
    return f"model: {content}"


def _tool_call_names(payload: Mapping[str, object]) -> list[str]:
    """model_response.tool_calls → ["name(call_id)"];畸形条目跳过(B1:读端不校验)。"""
    raw_calls = payload.get("tool_calls")
    names: list[str] = []
    if isinstance(raw_calls, (list, tuple)):
        for entry in raw_calls:
            if not isinstance(entry, Mapping):
                continue
            call_id = entry.get("call_id")
            name = entry.get("name")
            if isinstance(name, str):
                names.append(f"{name}({call_id})" if isinstance(call_id, str) else name)
    return names


def _joined_effects(payload: Mapping[str, object]) -> str:
    raw_effects = payload.get("effects")
    if isinstance(raw_effects, (list, tuple)):
        return ",".join(str(effect) for effect in raw_effects)
    return ""


def _result_text(payload: Mapping[str, object]) -> str:
    """tool_result 文本预览:content 文本块优先,退回 error_message,皆无则空。"""
    texts: list[str] = []
    blocks = payload.get("content")
    if isinstance(blocks, (list, tuple)):
        for block in blocks:
            if not isinstance(block, Mapping) or block.get("type") != "text":
                continue
            data = block.get("data")
            if isinstance(data, Mapping) and isinstance(data.get("text"), str):
                texts.append(data["text"])
    if not texts:
        error_message = payload.get("error_message")
        if isinstance(error_message, str) and error_message:
            texts.append(error_message)
    return " | ".join(texts)


def _str_field(payload: Mapping[str, object], key: str, default: str = "") -> str:
    value = payload.get(key)
    return value if isinstance(value, str) else default


def _short_digest(payload: Mapping[str, object]) -> str:
    digest = _str_field(payload, "args_digest")
    return digest[:12] if digest else "?"


def _preview(text: str, limit: int = _PREVIEW_LIMIT) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"
