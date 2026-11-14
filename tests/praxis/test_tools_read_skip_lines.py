"""K-5:read 行模式的跳行循环必须可推进(超大 start_line 不占死事件循环)。

从 ``test_tools_files_search.py`` 拆出,使安全修复(S 组)与正确性修复(K 组)
的提交边界清晰;断言本身与原计划一致,不弱化。
"""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any

import pytest

from praxis.protocol.sandbox import ToolExecContext
from praxis.protocol.tools import ResultStatus, ToolValidationError
from praxis.tools.builtins.filesystem import create_read_tool


def test_read_huge_start_line_yields_to_event_loop(tmp_path: Path) -> None:
    """K-5:超大 start_line 的跳行循环必须周期性让出事件循环。

    没有让出点时,ticker 在 read 完成前一次都跑不到(旧行为);每 256 行
    ``await asyncio.sleep(0)`` 后,ticker 能持续推进——这正是超时/取消/其它
    任务得以运行的前提。
    """
    (tmp_path / "many.txt").write_text(
        "".join(f"line {i}\n" for i in range(20_000)), encoding="utf-8"
    )
    ticker_ticks = 0

    async def scenario() -> Any:
        nonlocal ticker_ticks

        async def ticker() -> None:
            nonlocal ticker_ticks
            while True:
                ticker_ticks += 1
                await asyncio.sleep(0)

        tick_task = asyncio.create_task(ticker())
        try:
            return await create_read_tool(tmp_path).run(
                {"path": "many.txt", "start_line": 19_000, "max_lines": 5},
                ToolExecContext(),
            )
        finally:
            tick_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await tick_task

    result = asyncio.run(scenario())
    assert result.status is ResultStatus.OK
    assert "line 18999" in result.text  # 跳过 18999 行后从第 19000 行开始
    assert ticker_ticks >= 10, f"event loop made no progress during skip: {ticker_ticks}"


def test_read_start_line_has_upper_bound(tmp_path: Path) -> None:
    tool = create_read_tool(tmp_path)
    with pytest.raises(ToolValidationError):
        # K-5:start_line 有上限,避免多 GB 文件上的无界跳行。
        tool.validate_input({"path": "f.txt", "start_line": 10_000_001})
    assert (
        tool.validate_input({"path": "f.txt", "start_line": 10_000_000})["start_line"]
        == 10_000_000
    )