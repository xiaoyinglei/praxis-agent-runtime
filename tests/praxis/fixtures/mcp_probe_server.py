"""T16 探针 MCP server:**真实** stdio 子进程(已装 ``mcp`` SDK,不是 fake transport)。

测试用它证明三件事,每件都必须由一个真的被 execve 出来的进程回答:
1. S7:配置里的密钥只经 **envp** 到达 server(``env_probe`` 回显
   ``os.environ``),argv 里一个 ``KEY=VALUE`` 都没有(``argv`` 回显
   ``sys.argv``,测试侧再用 ``ps`` 独立复查);
2. 宿主环境不透传:``PRAXIS_HOST_ONLY_SENTINEL`` 在子进程里必须缺失;
3. owner-task 生命周期的失败面:``--startup-delay`` 让握手挂起(连 stdin 都
   不读),``--exit-delay`` 让关闭挂起(stdin EOF 后仍不退),
   ``--exit-immediately`` 让 server 在握手期就死掉。

``fs_probe`` 是**沙箱真实性**的行为证据:真 seatbelt 下读工作区外的宿主
文件必须被内核拒绝,passthrough 前缀下则读得到——两层差异由此可断言,
不靠"spec 对象长什么样"自证。

纪律:stdout 只走 JSON-RPC(``stdio_server()``),任何诊断都不得打印到
stdout,否则传输被污染。``--token`` 是每次启动唯一的 argv 标记,测试用它
在 ``ps`` 里精确定位/复查本实例(不依赖 pid 猜测,也不会误伤别的测试)。

本文件是**测试夹具**,不是产品代码:``env_probe`` 故意把完整环境与密钥值
回显进 tool result,好让断言有可观测证据;产品侧绝不做这种事。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from typing import Any

from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

_SECRET_VAR = "SECRET_MCP_KEY"
_SENTINEL_VAR = "PRAXIS_HOST_ONLY_SENTINEL"


def _echo_tool() -> Tool:
    return Tool(
        name="echo",
        description="Echo the supplied message.",
        inputSchema={
            "type": "object",
            "properties": {"message": {"type": "string"}},
            "required": ["message"],
            "additionalProperties": False,
        },
    )


def _env_probe_tool() -> Tool:
    return Tool(
        name="env_probe",
        description="Report this process's argv, pid and full environment.",
        inputSchema={"type": "object", "properties": {}, "additionalProperties": False},
    )


def _fs_probe_tool() -> Tool:
    return Tool(
        name="fs_probe",
        description="Try to read one byte of a path and report whether the OS allowed it.",
        inputSchema={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    )


def build_server(token: str) -> Server:
    """三个工具:echo(可用性)、env_probe(S7 证据)、fs_probe(沙箱真实性证据)。"""
    server = Server("praxis-mcp-probe")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [_echo_tool(), _env_probe_tool(), _fs_probe_tool()]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        if name == "echo":
            return [TextContent(type="text", text=f"echo: {arguments.get('message', '')}")]
        if name == "fs_probe":
            path = str(arguments.get("path", ""))
            try:
                with open(path, encoding="utf-8") as stream:
                    stream.read(1)
                outcome: dict[str, Any] = {"read_ok": True, "error": None}
            except OSError as error:
                outcome = {"read_ok": False, "error": type(error).__name__}
            return [TextContent(type="text", text=json.dumps({"path": path, **outcome}, sort_keys=True))]
        if name == "env_probe":
            payload = {
                "token": token,
                "pid": os.getpid(),
                "ppid": os.getppid(),
                "argv": list(sys.argv),
                "cwd": os.getcwd(),
                "env": dict(os.environ),
                "secret": os.environ.get(_SECRET_VAR),
                "sentinel": os.environ.get(_SENTINEL_VAR),
            }
            return [TextContent(type="text", text=json.dumps(payload, sort_keys=True))]
        raise ValueError(f"unknown tool: {name}")

    return server


async def _serve(token: str, exit_delay: float) -> None:
    server = build_server(token)
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())
    if exit_delay > 0:
        # 关闭挂起面:stdin 已 EOF,server 仍不退 ⇒ 上层 close_timeout 必须兜住。
        await asyncio.sleep(exit_delay)


def main() -> None:
    parser = argparse.ArgumentParser(description="Praxis MCP stdio probe server (test fixture)")
    parser.add_argument("--token", required=True, help="unique argv marker for ps probes")
    parser.add_argument("--startup-delay", type=float, default=0.0, help="block before serving (hung handshake)")
    parser.add_argument("--exit-delay", type=float, default=0.0, help="linger after stdin EOF (hung close)")
    parser.add_argument("--exit-immediately", action="store_true", help="die before the handshake completes")
    args = parser.parse_args()
    if args.exit_immediately:
        sys.exit(3)
    if args.startup_delay > 0:
        # 阻塞式 sleep:连 stdin 都不读,initialize 必然挂起到 ready_timeout。
        time.sleep(args.startup_delay)
    asyncio.run(_serve(args.token, args.exit_delay))


if __name__ == "__main__":
    main()
