"""skill 激活态投影:从 item 序列提取本回合已激活的 skill id(设计 §5.14;T9)。

纯函数:扫描 ``tool_result`` 项中 **已提交成功** 的 ``invoke_skill`` 结果——
payload 的 ``tool_name == "invoke_skill"``、``status == "ok"``、``is_error`` 不是
``True``——读取 ``metadata["skill_id"]``(仅字符串、非空)进集合。畸形/未知形状
一律跳过(fail-open,永不抛)。

**从已提交事实重算**是唯一事实源:``invoke_skill`` 的 success 结果就是"该 skill
在本回合被激活"的权威记录;没有第二份内存态可漂移。item 属于单个 turn
(``Store.read_turn_items`` 按 turn 过滤),因此新回合天然从空集开始——与旧
``tool_orchestrator._context_for_turn`` 的 turn 局部语义一致。

同 ``core.planning.latest_plan`` 先例:本模块只 import ``praxis.protocol``,
core 内部消费者(pipeline)直接调用。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from praxis.protocol.items import Item

_INVOKE_SKILL_TOOL_NAME = "invoke_skill"


def latest_active_skill_ids(items: Sequence[Item]) -> frozenset[str]:
    """本回合已成功激活的 skill id 集合(失败/畸形结果一律跳过)。

    - 只看 ``kind == "tool_result"``;调用项、模型消息等一律忽略;
    - 只看 ``tool_name == "invoke_skill"``、``status == "ok"``、``is_error`` 非
      ``True`` 的结果——错误结果不携带可靠 ``metadata``,不得授予激活态;
    - ``metadata`` 必须是 mapping,``skill_id`` 必须是非空字符串;
    - 返回不可变 ``frozenset``,调用方不能反向修改事实。
    """
    active: set[str] = set()
    for item in items:
        if item.kind != "tool_result":
            continue
        payload = item.payload
        if payload.get("tool_name") != _INVOKE_SKILL_TOOL_NAME:
            continue
        if payload.get("status") != "ok" or payload.get("is_error") is True:
            continue
        metadata = payload.get("metadata")
        if not isinstance(metadata, Mapping):
            continue
        skill_id = metadata.get("skill_id")
        if isinstance(skill_id, str) and skill_id:
            active.add(skill_id)
    return frozenset(active)
