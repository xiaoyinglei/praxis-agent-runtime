# Agent 产品契约

日期：2026-07-22
状态：当前实施依据

`docs/superpowers/specs/2026-07-18-agent-public-api-lifecycle-cleanup-design.md`
只是历史记录，不再决定当前实现。

## 产品主语

Agent 只有一种执行：用户给出当前消息，运行一个 Turn，得到一个
`AgentResult`。

- `run()` / `arun()` 执行当前消息。
- 不传 `previous_turn_id` 时，从空历史开始。
- 传 `previous_turn_id` 时，自然延续已有上下文。
- `chat` 只是自动传递上一个 `turn_id` 的 CLI 交互循环，不是第二套生命周期。
- `resume()` / `aresume()` 只恢复原来被审批、澄清或中断阻塞的同一个 Turn，不用于续聊。

用户不需要事先声明“这是多轮对话”，也不需要先建 Session。

## 稳定公开面

Python SDK 的产品入口是 `agent_runtime.Agent`。CLI 只调用这个入口，不调用
`AgentService` 或 `Agent` 私有方法。

```text
CLI / Python SDK
  -> Agent.run | arun | astream | resume | aresume
  -> AgentService
  -> AgentLoop
  -> ToolExecutor
  -> checkpoint / TurnStore / StreamEvent / AgentResult
```

公开身份只使用 `turn_id`。不再将同一概念同时暴露为 `run_id`、
`thread_id` 或 `session_id`。当前用户消息始终是当前用户消息；首条消息只存在于
canonical history，不会回填覆盖当前请求。

## 能力保留线

收敛 API 不等于删减产品能力。下列能力必须继续经过同一条主路径可用：

- 普通回答和自然多轮上下文；
- 本地文件分析、代码搜索、编辑和受限命令执行；
- 流式文本、工具、计划、恢复事件；
- 工具审批、持久 checkpoint 和跨进程恢复；
- 模型选择、本地模型健康检查和云模型密钥诊断；
- Skill、MCP、subagent 和 working memory；
- 显式配置的 RAG knowledge provider、evidence 和 citation；
- 稳定 `AgentResult`、usage、diagnostic、pause 和 tool-call 投影。

只能删除已被主路径完整替代的重复实现、无法达到的死路径和私有旁路。
不以行数为目标，不删成 demo，不新建第二套 Registry、Executor、事件系统或 Runtime。

## Workspace 和配置

- `workspace_path` 是 Agent 可读写的用户项目根目录。
- Agent 自身的 scratch、log 和外部附件位于 `.rag/agent_runtime/`，不在项目根目录伪造业务目录。
- workspace 内的附件直接引用，不复制；workspace 外的附件按 Turn 归档。
- 清单只包含当前请求明确传入的文件，不将旧 Turn 的附件偷渡到新 Turn。
- 环境变量优先级是：已导出的进程环境、`AGENT_ENV_FILE`、当前 workspace `.env`、linked worktree 共享 `.env`。
- 密钥缺失、模型不可用和本地服务冲突是有界产品错误，CLI 不打印内部 traceback 和对象状态。

## 验收方式

每次改动先验证真实公开路径，再用测试防回归。最小产品验收包括：

1. 云模型真实 CLI 调用；
2. 本地模型普通回答和文件工具调用；
3. SDK 两个 Turn 的自然延续；
4. `astream()` 的真实文本和生命周期事件；
5. 真实写工具的 pause -> approve -> resume -> 文件落盘；
6. CLI 错误是简短、可操作的产品诊断；
7. Git 状态不出现非用户意图的 runtime 文件。

内部 fake model、私有 helper 和单元测试不能单独支撑“产品可用”结论。
