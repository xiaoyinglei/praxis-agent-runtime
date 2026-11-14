"""自动压缩(设计 §5.7;D11/D15):超阈值 → 模型摘要 → ``summary`` + ``compaction_boundary``。

B22 纪律:压缩**只追加两条 item**,summary 放摘要文本、boundary 只放被覆盖的
seq 区间;原 item 永远留在 store,读端 ``build_context`` 按 boundary 折叠。
正文绝不进压缩日志——摘要文本来自模型输出,不复制被覆盖的投影。

R-T9-2:覆盖上界不得切断任何声明窗口(``model_response`` 声明了 tool_calls
而结果尚未落库/落在 tail)——否则切掉声明后,幸存的结果会成为孤儿 tool 消息,
provider 会拒收。判据复用 ``context.pair_tool_results``(投影与恢复的唯一配对
来源),本模块不写第二套配对逻辑。

失败安全:摘要模型异常、空摘要、store 写入失败 → ``logger.warning`` 后返回
``None``;**压缩失败绝不 fail turn**(turn 继续用原上下文)。
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final

from praxis.core.context import (
    build_context,
    elide_old_tool_outputs,
    pair_tool_results,
    to_messages,
)
from praxis.protocol.items import Item, ItemKind
from praxis.protocol.json import JsonValue, thaw_json
from praxis.protocol.models import (
    ModelAdapter,
    ModelMessage,
    ModelSettings,
    ModelSpec,
    Usage,
    WireRequest,
)
from praxis.store.log import ItemEntry, Store

logger = logging.getLogger(__name__)

COMPACT_VERSION: Final = "praxis-compaction-v1"
_ALGORITHM_VERSION: Final = "v1"
_DEFAULT_KEEP_TAIL_ITEMS: Final = 8
_DEFAULT_RESERVE_OUTPUT: Final = 8192
# 摘要输出预算的**安全上界**(D15 数据)。真实值由 composition 按
# ``min(SUMMARY_MAX_OUTPUT_TOKENS, spec.max_output_tokens)`` 注入——小输出窗口
# 的模型不会被 2048 硬顶撞出 provider 上限。
SUMMARY_MAX_OUTPUT_TOKENS: Final = 2048
# 摘要请求正文的硬字符上限:elide 只能约束工具输出,用户/助手正文本身无界,
# 没有这一层,压缩会在"长历史"上把自身请求撑爆(context overflow → 静默失败)。
SUMMARY_PROMPT_MAX_CHARS: Final = 16_000
_TRUNCATION_NOTE: Final = "[... 更近的内容已截断 ...]\n"
_ARGS_PREVIEW_CHARS: Final = 200

# 这些 kind 不参与覆盖区间选择(turn_status/approval_* 不投影成模型消息;
# 排除它们能让"最后 keep_tail_items 条"数的是真正的对话内容)。
_NON_COVERABLE_KINDS: Final = frozenset(
    {ItemKind.TURN_STATUS, ItemKind.APPROVAL_REQUEST, ItemKind.APPROVAL_DECISION}
)

_SUMMARY_INSTRUCTION: Final = (
    "请把下面的对话压缩成简洁的中文摘要:保留事实、决定、未完成事项和关键结果;"
    "只输出摘要正文,不要编造,不要复述无关细节。"
)


@dataclass(frozen=True, slots=True)
class CompactionOutcome:
    """一次成功压缩的事实:被覆盖的 seq 区间、摘要长度与本次摘要调用的 usage。"""

    covered_from: int
    covered_to: int
    summary_chars: int
    usage: Usage | None


def default_compact_limit(spec: ModelSpec, reserve_output: int = _DEFAULT_RESERVE_OUTPUT) -> int:
    """ModelSpec → 自动压缩阈值(D15:数据,不是 core 里的魔法数字)。

    优先用显式 ``auto_compact_token_limit``(0 是合法值,表示预算耗尽;
    **不得**用 ``or`` 把它误当成"没配");否则从上下文窗口扣掉输出预算与保留区。
    返回值 0 表示不自动压缩。
    """
    if spec.auto_compact_token_limit is not None:
        return spec.auto_compact_token_limit
    return max(0, spec.context_window_tokens - spec.max_output_tokens - reserve_output)


def default_summary_max_output_tokens(spec: ModelSpec) -> int:
    """摘要请求的输出预算 = ``min(安全上界, spec.max_output_tokens)``。

    ``OpenAICompatAdapter`` 会把 ``settings.max_output_tokens`` 原样透传给
    provider;硬编码 2048 会让输出窗口更小的模型直接拒收摘要请求。
    """
    return min(SUMMARY_MAX_OUTPUT_TOKENS, spec.max_output_tokens)


async def maybe_compact(
    *,
    store: Store,
    adapter: ModelAdapter,
    thread_id: str,
    turn_id: str,
    items: Sequence[Item],
    limit_tokens: int,
    keep_tail_items: int = _DEFAULT_KEEP_TAIL_ITEMS,
    summary_max_output_tokens: int = SUMMARY_MAX_OUTPUT_TOKENS,
    measure_tokens: Callable[[WireRequest], int] | None = None,
) -> CompactionOutcome | None:
    """超阈值时压缩 ``items`` 的较旧部分;未超阈值/区间为空/失败 → ``None``。

    1. 按 D11 读侧链 ``build_context`` → ``elide_old_tool_outputs`` → ``to_messages``
       构建与 turn 真实请求一致的投影,经 ``adapter.build_request`` 定型的字节测量;
       ``<= limit_tokens`` 直接返回 ``None``(不写任何东西);
    2. 覆盖区间 = 可见对话中除最后 ``keep_tail_items`` 条可压缩 item 之外的部分;
       上界按 R-T9-2 回退到不切断声明窗口的位置;为空 → ``None``;
    3. 用**同一个** ``adapter.complete()`` 请求摘要(单次调用,不再递归压缩);
       摘要投影同样经过 elide,并有 ``SUMMARY_PROMPT_MAX_CHARS`` 硬上限;
    4. 一次批量追加 ``SUMMARY`` + ``COMPACTION_BOUNDARY`` 两条 item;
    5. 任何失败 → ``logger.warning`` + ``None``。

    ``measure_tokens`` 是 D2 的唯一测量入口(``measure_request``)的注入形式:
    ``praxis_core_purity`` 禁止 core import adapters,由 composition 注入。
    ``summary_max_output_tokens`` 同样由 composition 按 ModelSpec 注入。
    """
    if measure_tokens is None:
        logger.warning("compaction skipped: no token measurement injected")
        return None
    try:
        visible = build_context(items)
        measured = adapter.build_request(
            # 与 turn 的真实请求同一条读侧链(D11):折叠 → 省略 → 投影。
            to_messages(elide_old_tool_outputs(visible)),
            (),
            ModelSettings(max_output_tokens=summary_max_output_tokens),
        )
        if measure_tokens(measured) <= limit_tokens:
            return None

        coverage = _coverage_range(visible, keep_tail_items)
        if coverage is None:
            return None
        covered_from, covered_to, covered_items = coverage

        summary_req = adapter.build_request(
            [ModelMessage(role="user", content=_summary_prompt(covered_items))],
            (),
            ModelSettings(max_output_tokens=summary_max_output_tokens),
        )
        response = await adapter.complete(summary_req)
        summary_text = response.content.strip()
        if not summary_text:
            logger.warning("compaction produced an empty summary; leaving context unchanged")
            return None

        store.append_items(
            thread_id,
            turn_id,
            [
                ItemEntry(
                    kind=ItemKind.SUMMARY,
                    payload={
                        "text": summary_text,
                        "covered_from": covered_from,
                        "covered_to": covered_to,
                        "compact_version": COMPACT_VERSION,
                        "model_id": summary_req.model,
                    },
                ),
                ItemEntry(
                    kind=ItemKind.COMPACTION_BOUNDARY,
                    payload={
                        "algorithm_version": _ALGORITHM_VERSION,
                        "covered_from": covered_from,
                        "covered_to": covered_to,
                    },
                ),
            ],
        )
        return CompactionOutcome(
            covered_from=covered_from,
            covered_to=covered_to,
            summary_chars=len(summary_text),
            usage=response.usage,
        )
    except Exception:  # noqa: BLE001 — 压缩失败绝不 fail turn
        logger.warning("compaction failed; leaving context unchanged", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# 覆盖区间选择
# ---------------------------------------------------------------------------


def _coverage_range(
    visible: Sequence[Item], keep_tail_items: int
) -> tuple[int, int, list[Item]] | None:
    """选覆盖区间:排除最后 ``keep_tail_items`` 条可压缩 item,再按声明窗口回退上界。"""
    ordered = sorted(enumerate(visible), key=lambda entry: (entry[1].seq, entry[0]))
    compressible = [item for _position, item in ordered if item.kind not in _NON_COVERABLE_KINDS]
    if keep_tail_items <= 0:
        candidates = compressible
    else:
        candidates = compressible[: max(0, len(compressible) - keep_tail_items)]
    if not candidates:
        return None
    covered_from = min(item.seq for item in candidates)
    covered_to = _safe_covered_to(visible, max(item.seq for item in candidates))
    if covered_to < covered_from:
        return None
    covered_items = [item for item in visible if covered_from <= item.seq <= covered_to]
    return covered_from, covered_to, covered_items


def _safe_covered_to(visible: Sequence[Item], upper: int) -> int:
    """把覆盖上界回退到不切断任何声明窗口的位置(R-T9-2)。

    对每个 seq <= upper 的 ``model_response`` 声明:若它的某个 tool_call 没有
    配对结果(未闭合),或配对结果 seq > upper(结果会幸存而声明被折叠),
    上界必须退到该声明之前。回退可能暴露更早的跨界配对,故迭代到不动点
    ——每次都严格减小上界,必然终止。

    配对判据完全来自 ``pair_tool_results``;本函数只枚举声明的 call_id。
    """
    paired, _orphans = pair_tool_results(visible)
    current = upper
    while True:
        barrier: int | None = None
        for item in visible:
            if item.kind != ItemKind.MODEL_RESPONSE or item.seq > current:
                continue
            call_ids = _declaration_call_ids(item)
            unsafe = call_ids is None  # 畸形声明:保守视为不可切
            if call_ids is not None:
                for call_id in call_ids:
                    ref = paired.get((item.turn_id, item.seq, call_id))
                    if ref is None or ref[0] > current:
                        unsafe = True
                        break
            if unsafe:
                barrier = item.seq if barrier is None else min(barrier, item.seq)
        if barrier is None:
            return current
        current = barrier - 1


def _declaration_call_ids(item: Item) -> list[str] | None:
    """枚举一条 ``model_response`` 声明的 call_id;``None`` 表示畸形/不可判定。

    只看 call_id 本身(配对判据仍在 ``pair_tool_results``);缺 ``name`` 等更深
    结构由 pair_tool_results 的严格重建决定,保守起见这里也把畸形当屏障。
    """
    raw_calls = item.payload.get("tool_calls")
    if raw_calls is None:
        return []
    if not isinstance(raw_calls, (list, tuple)):
        return None
    call_ids: list[str] = []
    for entry in raw_calls:
        if not isinstance(entry, Mapping):
            return None
        call_id = entry.get("call_id")
        if not isinstance(call_id, str):
            return None
        call_ids.append(call_id)
    return call_ids


def _summary_prompt(covered_items: Sequence[Item]) -> str:
    """固定中文指令 + 有界的覆盖区间投影文本(只进请求,不落库)。

    压缩恰恰在"旧工具输出很多"时触发,所以这里必须再过一层
    ``elide_old_tool_outputs(keep_last=0)``(R-T9-1 保证 artifact 指针行原样
    保留);用户/助手正文本身无界,再由 ``SUMMARY_PROMPT_MAX_CHARS`` 硬截断。
    """
    projection = "\n".join(
        _render_message(message)
        for message in to_messages(elide_old_tool_outputs(covered_items, keep_last=0))
    )
    header = f"{_SUMMARY_INSTRUCTION}\n\n"
    budget = SUMMARY_PROMPT_MAX_CHARS - len(header)
    if budget <= 0:
        return header[:SUMMARY_PROMPT_MAX_CHARS]
    if len(projection) <= budget:
        return header + projection
    keep = max(0, budget - len(_TRUNCATION_NOTE))
    # 保留最早的一段(被压缩历史的起点),末尾注明更近的内容已截断。
    return header + projection[:keep] + _TRUNCATION_NOTE


def _render_message(message: ModelMessage) -> str:
    """一条投影消息的单行渲染;tool_call-only 的助手轮也带上有界的调用名/参数。"""
    parts: list[str] = []
    if message.content:
        parts.append(message.content)
    if message.tool_calls:
        rendered_calls = ", ".join(
            f"{call.name}({_bounded_arguments(call.arguments)})" for call in message.tool_calls
        )
        parts.append(f"[tool_calls: {rendered_calls}]")
    return f"{message.role}: {' '.join(parts)}"


def _bounded_arguments(arguments: Mapping[str, JsonValue]) -> str:
    """tool_call 参数摘要:JSON 渲染并截到 ``_ARGS_PREVIEW_CHARS``。"""
    try:
        rendered = json.dumps(thaw_json(arguments), ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        rendered = repr(dict(arguments))
    if len(rendered) > _ARGS_PREVIEW_CHARS:
        return rendered[:_ARGS_PREVIEW_CHARS] + "..."
    return rendered
