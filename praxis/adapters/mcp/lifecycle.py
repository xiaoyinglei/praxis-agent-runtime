"""MCP stdio 生命周期:**owner task 唯一实现** + S7 envp 修复(3B §5.10;T16)。

机制逐条移植旧 ``_owned_mcp_connection``(``agent_runtime/runtime/mcp.py:
552-623``,审查点名的好设计),只换掉 S7 病灶(``:625-667``):

**为什么必须专起一个 owner task**:MCP SDK 的 ``stdio_client``/``ClientSession``
内部是 AnyIO cancel scope + task group,而 AnyIO 的 cancel scope 是**任务局部**
的——进入与退出必须发生在同一个 task 里。若在外层 task 进入、在另一个 task
(或取消后的外层)退出,关闭会泄漏或直接报错。因此:专起 ``mcp-owner:<name>``
task,task 内用 ``AsyncExitStack`` 持有全部资源,``asyncio.timeout(ready)``
内完成 spawn + ``initialize`` + ``list_tools``,就绪经 ``ready`` future 交给
外层,然后停在 ``stop.wait()``;失败装进同一个 future;外层 ``finally`` 一律
``stop.set()``,await 前被取消(``ready.cancelled()``)⇒ ``task.cancel()``;
关闭走 ``asyncio.shield(task)`` + ``wait_for(close_timeout)``,超时再
``cancel`` + ``gather``。**无自动重试**;单个 owner 只关自己的栈,异常原样
上抛给调用方(T19 逐 server 捕获转 ``mcp_server_unavailable``)。

**S7 正解**:配置 env 的每一个值只经 execve 的 **envp** 进入子进程,argv 里
零 ``KEY=VALUE``(详见 ``_stdio_parameters`` 的注释:旧 ``/usr/bin/env -i``
argv 重置在本姿势下没有必要,且它本身就是密钥进 ps 的病灶)。

**fail-closed(R-PF-5)**:``sandbox_spec is None`` ⇒ ``SandboxRefusal``,
不存在裸宿主派生路径(与 ``search_text`` 的 ``search_backend_unavailable``
同一姿势)。沙箱编译不在本模块:T19 的 manager 经 ``compile_sandbox`` 编译后
把成品 spec 传进来。

**注入纪律(R-T15-7)**:adapters 不得 import tools,所以最小环境由调用方注入
``EnvBuilder``(composition 绑 ``tools.builtins.child_env`` 的唯一实现);
``ScratchLocator`` 同族别名供 T19/T20 使用。本模块只消费签名。

**相对 brief 签名的有意收敛**(controller ruling 为准):
- ``ready_timeout`` 不另开参数,取 ``config.startup_timeout_seconds``
  (T15 已校验正有限);``close_timeout_seconds`` 默认 5.0(旧同值)。
- 不注入 ``kill_group``/``ProcessSignaler``:mcp 1.27 的 ``stdio_client``
  自己 ``start_new_session=True``,关闭时做 SIGTERM→SIGKILL 的**进程组**
  升级(``terminate_posix_process_tree``),孤儿回收已在 SDK 内闭合;
  测试用 ``ps`` 独立复查(含超时/取消两条路径)。
- ``cwd`` 逐字取 ``config.cwd``(T15 已保证落在 workspace 之内);为 None 时
  由 SDK 用进程 cwd,装配侧(T19)负责给 workspace 默认值。
- ``McpConnection`` 就定义在本模块:一个三字段数据类不值得再开 types.py。
  T18(HTTP)若需要 ``InitializeResult`` 再按只增不改扩展。
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.types import Tool as McpTool

from praxis.adapters.mcp.catalog import list_all_tools
from praxis.adapters.mcp.config import StdioServerConfig
from praxis.adapters.sandbox.base import BACKEND_CONFIGURATION_HINT
from praxis.protocol.sandbox import EnvBuilder, SandboxRefusal, SandboxSpec

logger = logging.getLogger(__name__)

# 后端名字取自 adapters/sandbox 的唯一词汇来源(S1 反模式:后端字面量不得
# 抄进本包字符串);文案整体仍是 controller 钉死的 fail-closed 拒绝原文。
_NO_SANDBOX_REASON = f"mcp stdio server requires a sandbox backend; {BACKEND_CONFIGURATION_HINT}"


@dataclass(frozen=True, slots=True)
class McpConnection:
    """一条**已就绪**的 stdio 连接:initialize + list_tools 都已完成。

    ``session`` 的生命周期由 owner task 独占(调用方只调用它,不关闭它);
    ``tools`` 是**完整**目录:T17 的 ``catalog.list_all_tools`` 在就绪前完成
    cursor 分页并施加 4096 工具 / 256 页 / 4 MiB 上限,绝不会只交第一页。
    """

    server_name: str
    session: ClientSession
    tools: tuple[McpTool, ...]


@asynccontextmanager
async def owned_stdio_connection(
    config: StdioServerConfig,
    *,
    workspace: Path,
    sandbox_spec: SandboxSpec | None,
    env_builder: EnvBuilder,
    close_timeout_seconds: float = 5.0,
) -> AsyncIterator[McpConnection]:
    """在沙箱内派生一个 stdio MCP server,就绪后 yield,退出时有界关闭。

    Parameters:
        config: T15 已校验/已展开 ``${VAR}`` 的 stdio 配置(含 startup 超时)。
        workspace: 真实工作区,``env_builder`` 的第二个入参(PATH 白名单用)。
        sandbox_spec: 已编译的沙箱执行形态;``None`` ⇒ ``SandboxRefusal``
            (fail-closed,无裸宿主派生路径)。
        env_builder: 子进程最小环境的唯一实现(composition 注入 ``child_env``)。
        close_timeout_seconds: 关闭宽限;超时即 cancel owner 并 gather。

    Raises:
        SandboxRefusal: 没有沙箱后端。
        Exception: 传输/握手失败原样上抛(调用方逐 server 隔离,不重试)。
    """
    if sandbox_spec is None:
        raise SandboxRefusal(_NO_SANDBOX_REASON)
    params = _stdio_parameters(config, workspace=workspace, spec=sandbox_spec, env_builder=env_builder)
    # MCP SDK contexts use task-local AnyIO cancel scopes. The owning task must
    # enter and leave them, including when discovery fails or shutdown is bounded.
    ready: asyncio.Future[McpConnection] = asyncio.get_running_loop().create_future()
    stop = asyncio.Event()

    async def serve() -> None:
        try:
            async with AsyncExitStack() as resources:
                async with asyncio.timeout(config.startup_timeout_seconds):
                    streams = await resources.enter_async_context(stdio_client(params))
                    session = await resources.enter_async_context(ClientSession(streams[0], streams[1]))
                    await session.initialize()
                    catalog = await list_all_tools(session)
                if not ready.done():
                    ready.set_result(
                        McpConnection(server_name=config.name, session=session, tools=catalog)
                    )
                await stop.wait()
        except BaseException as error:
            if not ready.done():
                ready.set_exception(error)
            elif not isinstance(error, asyncio.CancelledError):
                logger.warning("MCP connection %s closed: %s", config.name, type(error).__name__)

    task = asyncio.create_task(serve(), name=f"mcp-owner:{config.name}")
    try:
        yield await ready
    finally:
        stop.set()
        if ready.cancelled():
            task.cancel()
        try:
            await asyncio.wait_for(asyncio.shield(task), close_timeout_seconds)
        except TimeoutError:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


def _stdio_parameters(
    config: StdioServerConfig,
    *,
    workspace: Path,
    spec: SandboxSpec,
    env_builder: EnvBuilder,
) -> StdioServerParameters:
    """S7 主案:沙箱前缀 + 真实命令拼进 argv,**全部 env 值只走 envp**。

    旧病灶(``runtime/mcp.py:625-667``)把 server env 拼成
    ``/usr/bin/env -i KEY=VALUE ... command args`` 塞进 argv,动机是"显式 env
    不能影响 sandbox-exec 自身(例如动态加载器设置),故必须在沙箱内部重置"。
    **这个动机在 Python 传 envp 的姿势下不存在**:``stdio_client`` 把
    ``params.env`` 经 execve 的 envp 交给外层 ``sandbox-exec``,而
    ``sandbox-exec`` 是**原地 exec** 内层命令(实测 server pid == 直接子进程
    pid),envp 原样带进内层 exec——server 看到且只看到这份环境,不需要任何
    argv 重置。argv 重置反而把密钥明文写进 ``ps`` 可见的命令行(S7 本体),
    因此 ``KEY=VALUE`` argv 与 ``/usr/bin/env -i`` 形态全仓禁止。config.env
    本身是运维经 T15 信任门(属主/权限/sha256)授权的来源,不是模型可控输入。

    环境拼装顺序 ``{**env_builder(scratch, workspace), **config.env}``:
    最小环境是底,配置显式值覆盖同名键。scratch 取编译产物 ``spec.scratch_dir``
    (``compile_sandbox`` 已 mkdtemp;等价于 ``scratch_dir_for(spec)`` 的非 None
    分支),本函数不做任何 mkdir。

    SDK 行为留档(mcp 1.27):``stdio_client`` 会把 ``get_default_environment()``
    的六个宿主键(HOME/LOGNAME/PATH/SHELL/TERM/USER)merge 在 ``params.env``
    **之下**——同名键被我们的字典覆盖(HOME/PATH 由 child_env 决定),其余四键
    不含凭据;宿主凭据/哨兵类变量不在该白名单内,不可能透传(测试逐条钉死)。
    """
    environment = {**env_builder(Path(spec.scratch_dir), workspace), **config.env}
    return StdioServerParameters(
        command=spec.argv_prefix[0],
        args=[*spec.argv_prefix[1:], config.command, *config.args],
        env=environment,
        cwd=config.cwd,
    )


__all__ = ["McpConnection", "owned_stdio_connection"]
