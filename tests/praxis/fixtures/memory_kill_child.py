"""T11 测试③(B4 回归钉)的子进程:真 kill -9 的受害者。

流程:预置已完成源 turn → enable_memory 会话 submit → 终态钩子起 memory
driver → maintenance stage1 子 turn dispatch(慢剧本 3s 窗口)→ 写 ready
文件后长眠,等父进程 SIGKILL。无任何用户态清理——lease/job/子 turn 全部
留给重开进程的事实判据(未终态 turn + 未闭合窗口)收敛。
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from uuid import uuid4

from praxis.adapters.models.scripted import ScriptedAdapter
from praxis.composition import build_session
from praxis.core.events import EventBus
from praxis.core.statemachine import TurnStatus, transition
from praxis.protocol.items import ItemKind
from praxis.protocol.models import ModelResponse, Usage
from praxis.store.log import ItemEntry, Store
from praxis.store.memory import get_metadata
from praxis.store.turns import create_turn, get_turn

THREAD_ID = "kill-src"


def _resp(content: str) -> ModelResponse:
    return ModelResponse(
        content=content, tool_calls=(), usage=Usage(input_tokens=0, output_tokens=0), finish_reason="stop"
    )


async def main(workspace: Path) -> None:
    db = workspace / ".praxis" / "praxis.sqlite"
    db.parent.mkdir(parents=True, exist_ok=True)
    store = Store(db)
    try:
        turn_id = uuid4().hex
        create_turn(store, turn_id=turn_id, thread_id=THREAD_ID, binding={})
        appended = store.append_items(
            THREAD_ID,
            turn_id,
            [ItemEntry(kind=ItemKind.USER_MESSAGE, payload={"text": "I prefer neovim for editing."})],
        )
        transition(store, EventBus(), turn_id, TurnStatus.COMPLETED, reason="seeded")
        ref = appended[0].item_id
    finally:
        store.close()

    answer = json.dumps(
        {
            "entries": [
                {
                    "key": "k1",
                    "title": "K1",
                    "text": "User prefers neovim.",
                    "source_item_ids": [ref],
                    "kind": "preference",
                }
            ]
        }
    )
    session = build_session(
        workspace,
        ScriptedAdapter([_resp("parent done"), _resp(answer), _resp(answer)], delay_seconds=3.0),
        thread_id=THREAD_ID,
        tools=[],
        enable_memory=True,
    )
    try:
        await session.submit("hello")
        # 等 maintenance stage1 真 dispatch:job running + turn_ids 已登记 +
        # 子根 turn running(慢剧本流式在途)。
        deadline = time.monotonic() + 20.0
        while True:
            raw = get_metadata(session.store, "memory:job")
            job = {} if raw is None else json.loads(raw)
            if job.get("status") == "running" and job.get("turn_ids"):
                turn = get_turn(session.store, job["turn_ids"][0])
                if turn is not None and turn.status == "running":
                    break
            if time.monotonic() > deadline:
                raise RuntimeError("maintenance never dispatched")
            await asyncio.sleep(0.05)
        (workspace / "ready").write_text("dispatched", encoding="utf-8")
        while True:
            await asyncio.sleep(0.1)
    finally:
        await session.aclose()


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1])))