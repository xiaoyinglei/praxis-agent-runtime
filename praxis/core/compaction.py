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
from praxis.core.events import EventBus
from praxis.protocol.items import Item, ItemKind
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
# 摘要输出预算上界(与 ModelSpec 的 max_output_tokens 取小由调用方适配;
# 本函数拿不到 spec,2048 是安全上界)。
_SUMMARY_MAX_OUTPUT_TOKENS: Final = 2048
_DEFAULT_RESERVE_OUTPUT: Final = 8192

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

    优先用显式 ``auto_compact_token_limit``;否则从上下文窗口扣掉输出预算与
    保留区。返回值 0 表示预算耗尽——调用方把 0/None 都当作"不自动压缩"。
    """
    return spec.auto_compact_token_limit or max(
        0, spec.context_window_tokens - spec.max_output_tokens - reserve_output
    )


async def maybe_compact(
    *,
    store: Store,
    bus: EventBus,
    adapter: ModelAdapter,
    thread_id: str,
    turn_id: str,
    items: Sequence[Item],
    limit_tokens: int,
    keep_tail_items: int = _DEFAULT_KEEP_TAIL_ITEMS,
    measure_tokens: Callable[[WireRequest], int] | None = None,
) -> CompactionOutcome | None:
    """超阈值时压缩 ``items`` 的较旧部分;未超阈值/区间为空/失败 → ``None``。

    1. 按 D11 读侧链 ``build_context`` → ``elide_old_tool_outputs`` → ``to_messages``
       构建与 turn 真实请求一致的投影,经 ``adapter.build_request`` 定型的字节测量;
       ``<= limit_tokens`` 直接返回 ``None``(不写任何东西);
    2. 覆盖区间 = 可见对话中除最后 ``keep_tail_items`` 条可压缩 item 之外的部分;
       上界按 R-T9-2 回退到不切断声明窗口的位置;为空 → ``None``;
    3. 用**同一个** ``adapter.complete()`` 请求摘要(单次调用,不再递归压缩);
    4. 一次批量追加 ``SUMMARY`` + ``COMPACTION_BOUNDARY`` 两条 item;
    5. 任何失败 → ``logger.warning`` + ``None``。

    ``measure_tokens`` 是 D2 的唯一测量入口(``measure_request``)的注入形式:
    ``praxis_core_purity`` 禁止 core import adapters,由 composition 注入。
    ``bus`` 预留给未来压缩事件,当前不 emit。
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
            ModelSettings(max_output_tokens=_SUMMARY_MAX_OUTPUT_TOKENS),
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
            ModelSettings(max_output_tokens=_SUMMARY_MAX_OUTPUT_TOKENS),
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
    """固定中文指令 + 覆盖区间的投影文本(只进请求,不落库)。"""
    projection = "\n".join(
        f"{message.role}: {message.content}" for message in to_messages(covered_items)
    )
    return f"{_SUMMARY_INSTRUCTION}\n\n{projection}"