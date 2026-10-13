#!/usr/bin/env python
"""Try the real terminal UI without connecting to a model or writing chat state."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Literal, cast

from prompt_toolkit.completion import WordCompleter

from agent_runtime.terminal_app import TerminalChatApp

SAMPLE = """已经查看了项目入口，可以从下面三处开始阅读：

1. `cli.py` 接收终端命令。
2. `terminal_app.py` 管理输入、显示和键盘交互。
3. `terminal_ui.py` 保存对话和工具记录。

```python
def greet(name: str) -> str:
    return f"你好，{name}"
```

这是一段固定的示例回答，用来比较界面，没有调用模型。
"""


async def preview(ui: TerminalChatApp) -> None:
    ui.view.message("界面试玩 · 不调用模型，不保存聊天。\n"
                    "随意输入体验发送；/long 长回答 · /copy 复制 · /exit 退出。\n", "notice")
    answer = SAMPLE
    while True:
        message = (await ui.prompt()).strip()
        if message == "/exit":
            await ui.flush_scrollback()
            return
        if not message:
            continue
        if message == "/copy":
            await ui.copy_answer(answer)
            continue
        if message.startswith("/") and message != "/long":
            await ui.show_text("试玩命令：/long 长回答 · /copy 复制 · /exit 退出")
            continue
        ui.display.begin_turn()
        group = ui.view.add_group()
        tool = ui.view.add_tool(group, "示例：读取项目入口")
        tool.append("这里只展示工具记录的排版，没有执行文件读取。")
        group.expanded = False
        answer = SAMPLE if message != "/long" else SAMPLE + "\n" + "\n".join(
            f"{index}. 示例长回答：滚动查看上下文、中文和 `代码` 的显示。" for index in range(1, 61)
        )

        async def stream(text: str = answer) -> None:
            for offset in range(0, len(text), 50):
                ui.view.message(text[offset:offset + 50])
                await asyncio.sleep(0.025)

        try:
            await ui.operation(stream())
        except asyncio.CancelledError:
            current = asyncio.current_task()
            if current is not None and current.cancelling():
                raise
            ui.display.interrupted()
        ui.display.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ui", choices=("inline", "fullscreen"), default="fullscreen")
    args = parser.parse_args()
    ui = TerminalChatApp(
        model="界面试玩（无模型）", workspace=str(Path.cwd()),
        ui_mode=cast(Literal["inline", "fullscreen"], args.ui),
    )
    ui.editor.completer = WordCompleter(
        ["/long", "/copy", "/exit"], sentence=True,
        meta_dict={"/long": "长回答", "/copy": "复制", "/exit": "退出"},
    )
    asyncio.run(ui.run(lambda: preview(ui)))


if __name__ == "__main__":
    main()
