"""MCP 异常组摊平的**唯一实现**(T16 交接 / T19)。

MCP SDK(anyio task group、stdio/streamable-http 传输)在握手失败时抛出的常是
``ExceptionGroup``(实测形状 ``ExceptionGroup(ExceptionGroup(TimeoutError))``)。
把这种结构原样交给用户,诊断只会显示 "unhandled errors in a TaskGroup"。
本模块提供全包唯一的有界摊平:叶子数与嵌套深度双上限(敌意/深嵌套 group
不会造成无界工作),文案固定上限 512 字符。

T17 的 ``descriptors._call_failure_message`` 与 T19 的 connect 诊断共用本
模块——不允许出现第二个 flattener(T18 修复轮/T16 交接 ruling)。
"""

from __future__ import annotations

from collections.abc import Callable

_MAX_ERROR_MESSAGE_CHARS = 512
_MAX_LEAF_MESSAGES = 8
_MAX_LEAF_SCAN = 64
_MAX_LEAF_DEPTH = 64


def exception_leaves(error: BaseException) -> tuple[list[BaseException], int]:
    """非递归摊平 group,返回 ``(叶子列表, 未展开节点数)``。

    上限:最多 ``_MAX_LEAF_MESSAGES`` 个叶子、``_MAX_LEAF_DEPTH`` 层嵌套、
    ``_MAX_LEAF_SCAN`` 个被扫描的 group 节点。超限节点计入 excess 而不是
    递归下钻/追加到消息。
    """
    leaves: list[BaseException] = []
    excess = 0
    stack: list[tuple[BaseException, int]] = [(error, 0)]
    groups = 0
    while stack:
        current, depth = stack.pop()
        if isinstance(current, BaseExceptionGroup):
            if depth >= _MAX_LEAF_DEPTH or groups >= _MAX_LEAF_SCAN:
                excess += 1
                continue
            groups += 1
            stack.extend((sub, depth + 1) for sub in reversed(current.exceptions))
        elif len(leaves) < _MAX_LEAF_MESSAGES:
            leaves.append(current)
        else:
            excess += 1
    return leaves, excess


def bounded_message(message: str, *, fallback: str) -> str:
    """压平空白并截断到 512 字符;空文本回落到固定文案。"""
    collapsed = " ".join(message.split())
    return collapsed[:_MAX_ERROR_MESSAGE_CHARS] or fallback


def bounded_exception_message(error: BaseException, *, fallback: str) -> str:
    """叶子摘要(``类型: 文本``,分号连接);超限只标注 ``+N more``。"""
    leaves, excess = exception_leaves(error)
    detail = "; ".join(f"{type(leaf).__name__}: {leaf}" for leaf in leaves)
    if excess:
        detail = f"{detail}; (+{excess} more)" if detail else f"{fallback}; (+{excess} more)"
    return bounded_message(detail or fallback, fallback=fallback)


def diagnostic_message(
    error: BaseException,
    *,
    fallback: str,
    redact: Callable[[str], str] | None = None,
) -> str:
    """诊断口径:外层 kind + 第一片叶子的摘要(T19 controller ruling)。

    只取第一片叶子,避免把整组异常铺进一行状态输出;完整叶子列表仍由
    ``bounded_exception_message`` 提供给工具调用失败面。

    修复轮 #2(redaction ordering):``redact`` 在**压平空白与截断之前**对
    原始文本执行,并在 collapse 之后**再次**执行——跨越 512 边界的秘密不会
    因截断留下前缀,含空白的秘密也不会因 collapse 而匹配不到。调用方负责
    "raw secret + collapsed secret 双形态"替换。
    """
    leaves, _excess = exception_leaves(error)
    kind = type(error).__name__
    if leaves:
        first = leaves[0]
        raw = f"{kind}: {type(first).__name__}: {first}"
    else:
        raw = kind
    if redact is not None:
        raw = redact(raw)
    collapsed = " ".join(raw.split())
    if redact is not None:
        collapsed = redact(collapsed)
    return collapsed[:_MAX_ERROR_MESSAGE_CHARS] or fallback


__all__ = [
    "bounded_exception_message",
    "bounded_message",
    "diagnostic_message",
    "exception_leaves",
]
