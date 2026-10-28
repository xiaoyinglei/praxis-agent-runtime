# 模型输出与 Runtime 事件

模型输出、执行事实和公共通知是三种契约，复用现有持久化与事件派发设施。

| 边界 | 负责的事实 | 不授予的权限 |
| --- | --- | --- |
| `modeling.events.ModelEvent` | 一次模型请求的输出对象、内容、响应终态与用量 | 执行工具、批准操作、完成 Turn |
| `RuntimeState` / `RolloutRecord` | 请求与尝试、执行、审批、恢复、预算、完成判定 | 从模型文字推断执行已经成功 |
| `streaming.events.StreamEvent` | CLI / SDK 可观察的内容、状态和交互请求 | 直接作为业务状态的写入来源 |

Provider 原生 chunk 在 Gateway 内转换。Model 事件不携带 Runtime 的 thread、turn、attempt 或公开 Item ID。
Runtime 将 request 内的 `output_id` 映射到包含 attempt 身份的公开 Item；不同 attempt 可以复用相同模型输出 ID。
工具的 `tool_call_id`、输出对象 ID 和 provider block index 分别保留，不能相互替代。

## Model 契约

`HarnessModel.dispatch(prepared, *, event_sink=None)` 返回 `HarnessModelResponse`。
回调中的类型为：

- `ModelResponseStarted`：响应开始，可携带 provider 响应身份。
- `ModelOutputStarted`：独立输出对象开始。
- `ModelOutputDelta`：指定对象的内容或工具参数增量。
- `ModelOutputCompleted`：该对象的权威完整快照。
- `ModelResponseCompleted`：请求终态、停止原因和用量。

`ModelOutput` 支持 text、reasoning、reasoning_summary、plan、tool_call、refusal。
reasoning signature 等不透明数据放在对象的 `provider_data` 中，保存到私有事实，不投影到公共通知。

`HarnessModelResponse.outputs` 保留有序对象。text、reasoning_content、plan_content、tool_calls 是便捷投影。
简单的非流式模型仍可从 text / tool_calls 构造结果，构造时立即规范化为对象快照；不存在旧 delta 协议旁路。
流式回调与返回结果的对象、响应身份、终态及 provider 用量必须一致；适配器可以追加价格和耗时字段。
对象完成前的预览可以由完整快照修正，对象完成后不能继续追加或更改。

传输失败仍通过明确的 dispatch 异常分类，避免回调与异常分别宣告两个结果。
确认取消、明确拒绝和结果未知有不同的 Runtime 后果。关闭本地订阅者不证明 provider 已取消请求。

## Runtime 与公共通知

Runtime 把状态转换提交为事实，再从已提交批次派发公共通知。公共流包括：

- Turn / Item 的开始、增量、完成、暂停、恢复和取消。
- `model_attempt_updated`：准备、派发、开始接收响应、完成、重试、拒绝、取消或结果未知。
- `tool_operation_updated`：工具接纳、拒绝、授权变化及恢复核对等控制状态。
- `interaction_requested/resolved/invalidated`：可操作请求、请求身份和解决结果；用户回复正文不作为广播载荷。
- `context_compaction_started/completed`：已提交的上下文变换及版本范围。
- `completion_decided`：Runtime 的结束决定和独立 validation 状态。
- `verification_completed`：已记录的验证事实与来源。

生成中的工具调用以 `TOOL_CALL` Item 和 `tool_arguments` 增量呈现，`execution_started=false`。
执行过程使用独立的 TOOL / COMMAND Item。只有完整响应被 Runtime 接纳并经过授权后才能进入执行。
模型计划文本与 Runtime 已提交的结构化计划也分别记录。

attempt 完成仅表示收到明确结果、可以结算用量，不表示响应完整或任务完成。
`response_status=incomplete`、`incomplete_reason`、`stop_reason` 随 attempt 通知公开。
已经生成完成的内容对象保持 success；整个请求被截断时，工具不会从该不完整步骤执行。

## 持久化、回放与恢复

- 输出对象开始与完整快照持久化；token 增量不逐个写入事实日志。
- `model_response_started` 保存已观察到的 provider 身份，即使第一段内容到达前就断流也可恢复诊断。
- 断流或 lease 到期只关闭尚未完成的对象；已完成对象保留。未知的工具效果不会被文本生成事件替代。
- 最终聚合 model_response 保存有序 outputs，供恢复和上下文构建使用；公共投影抑制重复的聚合正文。
- 历史请求快照与 `HarnessMessage` / `ModelMessage` 保留输出对象和不透明数据。
- 重连依赖 `ReplayEvent.cursor`，不能依赖 `StreamEvent.sequence`；后者不是持久化恢复游标。
- 当前公共投影 epoch 为 `harness-canonical-stream-v4`，旧游标需要重新同步。

当前 OpenAI Chat Completions wire 显式投影 text、reasoning_content 和 tool_calls。
保存 provider_data 不代表已经实现 Claude 原生内容块或签名回传，也不代表 provider 支持恢复远端响应流。
压缩通知描述已提交的上下文变换；昂贵摘要请求的等待、失败和用量由其模型 attempt 通知描述。

## 参考与验证

对照源码使用本地固定版本：Codex `ddf04ad26789d040f9ef6a96736f76602e35a6cc` 的
`core/src/session/turn.rs`、`stream_events_utils.rs`；Claude 参考副本
`276b8e6a0160939c74f04074a5e10ec1ba76982b` 的 `services/api/claude.ts`、`QueryEngine.ts`。
后者仓库自述为历史反编译快照，仅用于比较，不作为当前官方实现或复制授权的依据。

核心回归：

```sh
uv run pytest -q tests/agent/test_model_events.py \
  tests/agent/harness/test_model_output_lifecycle.py \
  tests/agent/harness/test_runtime_notifications.py \
  tests/provider/test_llm_gateway.py
```

覆盖多对象顺序与身份、交错工具参数、截断与断流、取消、lease 恢复、私有数据过滤、
实时与重开回放一致性、下一轮上下文恢复，以及不随历史长度增长的 delta 身份校验。
