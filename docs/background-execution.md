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

普通后台提交会自动启动独立进程。同一数据库由 OS 文件锁保证只有一个宿主；单 Thread 只领取一个驱动命令。目前同一宿主还串行调度共享工作区的顶层驱动，子写任务的独立工作树与集成属于后续工作。

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
- 后台宿主不是系统开机服务，尚未实现跨关机自动启动、独立写工作树集成、完整渠道、CI/PR 和发布验收。
