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
请求准备只创建 attempt 身份；公开输出身份在真实输出块开始时生成，不按固定 channel 预分配。

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

`ModelMessage` 是 Runtime 与 Provider 共用的消息类型。它只保存 role、body、tool_call_id 和 context_category；
assistant 的 body 是有序 `ModelOutput` 元组，其他角色的 body 是文本。content、reasoning_content、tool_calls
均为只读计算属性，不与 body 重复保存。`ModelToolCall.arguments` 保存不可变的解析参数。

`HarnessModelResponse` 只保存 outputs 和响应元数据；text、reasoning_content、plan_content、tool_calls 是只读投影。
构造器只接收 outputs。简单模型可显式使用 `from_parts(...)` 生成输出块；这个工厂不会保存平铺字段，
也不能与 outputs 同时传入。Gateway 的 `AgentModelResponse.turn` 同样是计算结果。
完整快照校验输出 ID 和工具调用 ID 唯一。工具是否可执行还取决于完整响应和停止原因，不能从方便读取的投影反推；
例如 END_TURN 同时包含工具块会被拒绝，不完整响应保留输出事实但不提供可执行调用。
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
- 最终聚合 model_response 只保存有序 outputs 与响应元数据，供恢复和上下文构建使用；公共投影抑制重复的聚合正文。
- 历史请求快照使用同一个 ModelMessage body，保留输出对象和不透明数据。历史、答案、压缩、memory 和恢复
  均从该内容计算文本和工具调用；不再分别保存或接受另一份 text / tool_calls / reasoning_content。
- 重连依赖 `ReplayEvent.cursor`，不能依赖 `StreamEvent.sequence`；后者不是持久化恢复游标。
- 当前公共投影 epoch 为 `harness-canonical-stream-v4`，旧游标需要重新同步。

当前 OpenAI Chat Completions wire 显式投影 text、reasoning_content 和 tool_calls。
本地 flat JSON 提示词只发送一次消息 body，不包含 provider_data；私有回放元数据不变成提示词正文。
保存 provider_data 不代表已经实现 Claude 原生内容块或签名回传，也不代表 provider 支持恢复远端响应流。
压缩通知描述已提交的上下文变换；昂贵摘要请求的等待、失败和用量由其模型 attempt 通知描述。

## 参考与验证

对照源码使用本地固定版本：Codex `ddf04ad26789d040f9ef6a96736f76602e35a6cc` 的
`core/src/session/turn.rs`、`stream_events_utils.rs`；Claude 参考副本
`276b8e6a0160939c74f04074a5e10ec1ba76982b` 的 `services/api/claude.ts`、`QueryEngine.ts`。
后者仓库自述为历史反编译快照，仅用于比较，不作为当前官方实现或复制授权的依据。

本次沿 Codex 的 ResponseItem → OutputItemDone → record_conversation_items → rollout reconstruction，
以及 Claude 的 content_block → AssistantMessage → QueryEngine.recordTranscript → sessionStorage 恢复链核对边界。
两者都保留带类型的内容对象，并在协议边界生成所需表示。这里采用有序输出事实和边界投影，删除
Harness / Provider 两份消息类、平铺内容与 outputs 的并存写入、纯字段搬运和按 channel 收尾的路径。
保留本项目的 attempt / output / tool operation 分离、同事务提交以及提交后通知；
不采用 Claude 历史副本依赖延迟序列化及共享对象可变引用更新最终 stop_reason / usage 的方式。

核心回归：

```sh
uv run pytest -q tests/agent/test_model_events.py \
  tests/agent/harness/test_model_output_lifecycle.py \
  tests/agent/harness/test_runtime_notifications.py \
  tests/provider/test_llm_gateway.py
```

覆盖多对象顺序与身份、交错工具参数、截断与断流、取消、lease 恢复、私有数据过滤、
实时与重开回放一致性、下一轮上下文恢复，以及不随历史长度增长的 delta 身份校验。
