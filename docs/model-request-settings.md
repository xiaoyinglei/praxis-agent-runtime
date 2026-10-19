# 模型能力与请求设置

模型能力、用户偏好和实际请求分别保存。模型能力来自模型目录中的显式声明；不根据模型名称推断能力，也不在每次生成前请求服务商元数据。

## 使用

```sh
uv run agent chat --model deepseek-flash --effort high --max-output-tokens 65536
uv run agent run "检查项目" --model deepseek-flash --effort low --max-output-tokens 65536
```

聊天命令：

- `/effort`：查看思考强度、模型支持列表和输出额度。
- `/effort low`、`/effort high`：设置模型声明支持的等级。
- `/effort default`：恢复模型默认强度。
- `/effort off`：关闭思考，仅支持声明了关闭控制的模型。
- `/output 65536`：设置单次输出额度，包含思考和最终正文 token。
- `/output default`：恢复模型默认输出额度。

工作过程中也可输入这些命令。设置先保存到当前聊天的 rollout，从下一次新模型请求生效。已发送的请求、结果未知的请求和它们的恢复重试使用原快照。聊天重新打开会恢复偏好；切换模型使用目标模型自己的偏好。不支持或能力未知的强度会报错，不更改状态，也不调用模型。

Python SDK 使用同一验证和冻结链路：

```python
from agent_runtime import Agent

agent = Agent(model="deepseek-flash", reasoning_effort="high", max_output_tokens=65536)
async with agent.session() as session:
    session.configure_request(reasoning_effort="low")
    print(session.request_settings())
    result = await session.submit("检查项目")
```

持有 Session 时通过 `session.configure_request()` 修改该聊天。`Agent.configure_request()` 选择后续打开 Session 的初始偏好。host 模型会话文件使用原有锁、版本及 fingerprint 检查；聊天内设置使用 rollout 事务和请求设置版本检查，避免另一个旧 Session 静默覆盖。

## 模型声明

```yaml
models:
  example-model:
    capability: chat
    provider: example-provider
    context_window_tokens: 1048576
    max_output_tokens: 131072  # 服务商允许的单次输出硬上限
    reasoning:
      supported_efforts: [low, high]
      default_effort: high
      thinking_toggle: thinking
    defaults:
      max_output_tokens: 65536  # 正常请求的默认额度
```

`reasoning` 缺失或 `supported_efforts: null` 表示未知，空列表表示明确不支持。`off` 独立于 effort 等级；`thinking_toggle` 可声明 `thinking` 或 `chat_template_kwargs`，运行时才发送对应协议字段。固定思考 token 预算没有被伪装成通用 effort 控制。用户模型可以用 `agent models add/update --from <模型声明文件>` 配置相同字段。

主 Agent 输出优先级：用户额度 → 模型默认额度 → 原阶段预算作为保守默认值。新请求中，阶段默认 32K 不再充当覆盖显式额度的硬上限；服务商硬上限、上下文窗口及显式累计预算仍有效。内部摘要继续使用独立阶段预算，只有模型声明支持时才关闭思考。未声明可关闭的模型保留其默认行为。

每次请求在 `request_ref.effective_settings` 保存设置版本、思考强度、申请额度、生效额度和约束原因；终端显示生效值，值发生变化才再次显示。较大的输出预留会减少可用输入空间，既有压缩与预算准入继续生效。旧冻结定义不增加默认字段；旧待恢复请求保留原阶段上限和原 wire 数据。

仓库的 `deepseek-flash` 声明使用 2026-10-18 核验的[官方模型元数据](https://api-docs.deepseek.com/api/list-models/)：low/high/max，默认 high；单次硬上限 393216。默认请求额度选择 65536，用户可独立调整。

## 计划接口

模型看到的当前计划使用 `plan[].step`、`status`、`step_id`，与 `update_plan` 工具输入一致；压缩交接中的结构化计划使用相同字段。SDK 和历史存储中的 `AgentPlan.steps[].title` 保持兼容；工具仍拒绝错误字段，不把错误别名偷偷转换。
