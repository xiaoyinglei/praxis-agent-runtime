# 后台执行与重新接入

后台宿主是本机独立 Python 进程，调用同一套 Harness。提交命令先写入 rollout 数据库，再返回 Thread/Turn 身份。终端退出、SDK 句柄丢弃和等待超时都不取消任务。电脑关机、进程被杀或执行环境离线时，实际工作会停止；恢复按持久事实决定下一步。

## CLI

在目标工作区运行：

```sh
uv run agent run "检查并修复测试失败" --background --input-id fix-tests-001 \
  --checkpoint-db /absolute/path/runtime.sqlite --max-steps 20
```

返回 JSON 中的 `thread_id`、`turn_id`、`checkpoint_db`、`workspace` 用于重新接入。后续命令均使用相同的身份和数据库：

```sh
uv run agent background status THREAD TURN --checkpoint-db /absolute/path/runtime.sqlite --workspace /absolute/workspace
uv run agent background events THREAD TURN --checkpoint-db /absolute/path/runtime.sqlite --workspace /absolute/workspace
uv run agent background wait THREAD TURN --checkpoint-db /absolute/path/runtime.sqlite --workspace /absolute/workspace --timeout 30
uv run agent background result THREAD TURN --checkpoint-db /absolute/path/runtime.sqlite --workspace /absolute/workspace
uv run agent background send THREAD TURN "先检查边界情况" --input-id steer-001 --checkpoint-db /absolute/path/runtime.sqlite --workspace /absolute/workspace
```

`status` 显示真实 Turn 状态、恢复动作、子任务和命令收据。`queue_status` 只描述初始命令的交付状态；`applied` 不能解释为编码验收通过。`events` 的游标可传给下一次 `--after`。状态、事件和结果读取不启动模型或宿主。

暂停需要审批或补充输入时，使用 `status.recovery.request_id` 和其列出的动作：

```sh
uv run agent background respond THREAD TURN REQUEST allow_once --input-id approval-001 --checkpoint-db /absolute/path/runtime.sqlite --workspace /absolute/workspace
uv run agent background respond THREAD TURN REQUEST continue --text "用户的回答" --input-id answer-001 --checkpoint-db /absolute/path/runtime.sqlite --workspace /absolute/workspace
uv run agent background cancel THREAD TURN --input-id cancel-001 --checkpoint-db /absolute/path/runtime.sqlite --workspace /absolute/workspace
```

相同 `input_id` 和相同输入返回原收据；同一 ID 对应不同输入会报冲突。审批绑定具体请求；人工重试授权绑定当时的 Turn 版本，不能在下一次崩溃后自动授权另一轮模型请求。模型结果未知时必须显式选择恢复动作；未知工具副作用仍须走现有核对流程，取消不会抹掉这类事实或退还未知预算。

默认沿用现有 `ask` 模式；原有 `--approval-mode`、工具开关、资源权限和预算参数继续生效。续聊继承已有会话设置，显式 `ask` 会撤回自动执行授权。`--previous-turn-id` 仅接在终态 Thread head 后；排队期间前驱改变会阻止执行。

需要前台运行宿主时：

```sh
uv run agent background serve --checkpoint-db /absolute/path/runtime.sqlite --concurrency 4
```

普通后台提交会自动启动独立进程。同一数据库由 OS 文件锁保证只有一个宿主；单 Thread 只领取一个驱动命令。同一宿主串行调度共享工作区的顶层驱动；附着子任务使用独立来源快照和执行目录，具体合同见 [工作树与子任务集成](worktrees.md#子任务快照与主代理集成)。

## Python SDK

```python
import asyncio
from pathlib import Path
from agent_runtime import Agent

async def main():
    agent = Agent(
        workspace_path=Path("/absolute/workspace"),
        checkpoint_db=Path("/absolute/path/runtime.sqlite"),
    )
    handle = await agent.start_background("检查并修复测试失败", input_id="fix-tests-001", max_steps=20)
    print(handle.thread_id, handle.turn_id)
    attached = agent.attach_background(thread_id=handle.thread_id, turn_id=handle.turn_id)
    print(await attached.status())
    print(await attached.wait(timeout=30))
    # wait 遇到暂停会立即返回；respond 后可根据 status.commands 跟踪回复收据。
    # 在任务完成后读取：
    print((await attached.result()).answer)

asyncio.run(main())
```

`autostart=False` 可只排队，再由嵌入式 `BackgroundHost` 或前台 `serve` 执行。附件沿用现有原文件引用语义，在实际执行时读取；不会复制附件或承诺原文件内容永久不变。队列保存模型定义与请求偏好；定义改变时阻止执行，要求重新提交。

## 恢复与当前边界

- 宿主重启后释放旧宿主的领取，代次检查拦截旧驱动继续派发模型或工具。
- 已提交的响应和终态从 Harness 恢复；未知模型请求不会自动重发。
- 子任务分配、暂停、结算和父任务唤醒均读取 canonical allocation/Turn；暂停子任务可用其 Thread/Turn 身份接入。
- 工作区暂时不存在时保留待执行命令，环境恢复后重试领取。
- 后台宿主不是系统开机服务，尚未实现跨关机自动启动、完整渠道、CI/PR 后续和发布验收。


## 独立子任务与重新接入

`task` 和 `spawn_agent` 复用相同的附着子任务生命周期。新的附着子任务先冻结来源、持久分配执行目录和预算，再开始模型或工具。后台 `status.children[].environment` 显示来源、base、实际目录、产物和集成/验证状态。`done` 仅表示子 Turn 已结束，不表示主工作区已经集成或通过验收。

在正常项目入口，可用子任务的 Thread/Turn 身份执行 `background status/events/send/respond/cancel`。SDK 中由父 `Agent.attach_background(thread_id=child_thread, turn_id=child_turn)` 接入也成立：接入会沿 canonical lineage 校验来源，随后使用实际子目录。其它工作区不能凭子任务 ID 接入。权限撤销与审批降级读取祖先最新的持久策略，宿主重启不会恢复旧写权限。

主代理需要读取真实 patch，显式集成或放弃产物，并在主工作区重新运行验证。未集成、验证过期、子任务未知执行或未结清预算都会阻止正常完成。工具的验证记录只证明指定命令成功执行过；用户目标是否满足仍需审查，完整版本化交付证据属于后续交付工作。
