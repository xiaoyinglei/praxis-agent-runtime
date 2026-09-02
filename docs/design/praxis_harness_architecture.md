# Praxis Harness 目标架构与验收契约

日期：2026-08-25  
状态：目标契约，实施尚未完成  
基线：`origin/main@be4e6a04af8cf01d975b78bac524dd7a333c1d0b`

## 1. 契约地位

本文是 Praxis Agent 新编排内核的唯一目标架构与验收依据。它取代
`docs/design/agent_product_contract.md` 中以下运行时结论：

- 只使用 `turn_id`，不建立真实 Thread；
- `AgentService -> AgentLoop -> ToolExecutor` 永久作为目标主链；
- LangGraph checkpoint 与 TurnStore 并存；
- `LoopState` 作为 Context、Tool、Memory、Planning 和恢复的共享状态。

`docs/superpowers/specs/2026-08-04-praxis-hard-cutover-design.md` 仍然是品牌、包迁移、
RAG 解耦和仓库交付的历史记录，但它关于“保留现有运行时主链”的范围限制不适用于本次内核替换。

本文不允许通过改名、拆文件、增加 facade、增加测试数量或保留旧内核 fallback 来宣称完成。
只有第 16 节的证据矩阵全部成立，旧编排被删除，目标才算达成。

## 2. 设计依据与适用范围

设计依据：

1. OpenAI Codex open harness 当前源码，特别是 Thread/Turn/Item 协议、Session、
   TurnContext、StepContext、ContextManager、ToolRouter、ToolRegistry、
   ToolOrchestrator 和 rollout 边界。
2. OpenAI 官方《Codex as a platform: build on the open agent harness》对 harness
   职责的定义：状态、上下文、工具、沙箱、审批、事件、失败与跨 Turn 连续性。
3. 《深入理解 AI Agent：设计原理与工程实践》v1.4 对 Harness 五类功能的归纳：
   Context、Tools、Constrain、Verify、Correct。
4. Praxis 已有真实实现、测试、安全反例和 Code Agent benchmark。

目标产品边界：

- 单用户、trusted-local、macOS 优先；
- 面向本地 Python/Git 工作区的 Code/Data Agent；
- CLI 优先，Python SDK 使用同一条运行时路径；
- RAG、MCP、Skill 和 subagent 是可选能力提供者；
- 不扩张为多租户远程执行平台，不引入 RBAC、计费或分布式调度。

## 3. 当前架构判决

当前代码包含大量应保留资产，但编排所有权错误：

- `agent_runtime/service.py` 同时负责装配、Turn 创建、lease、checkpoint、恢复、流式输出和收尾；
- `agent_runtime/loop/runtime.py` 同时负责模型、规划、压缩、工具激活、持久化、事件和完成判断；
- `LoopState` 有 42 个顶层字段，多份消息历史、工具账本和审批状态同时存在；
- `TurnStore` 与 LangGraph checkpoint 都保存权威运行事实，恢复依靠同步与 reconciliation；
- core、memory、provider、skills 等模块反向依赖 `LoopState`；
- `StreamEvent` 使用进程内全局 sequence，不能作为跨进程回放协议。

因此，继续“抽薄 AgentService”或“拆 AgentLoop 文件”不能解决问题；全面重写模型、工具、
RAG、安全和 CLI/SDK 又会丢弃已验证资产。选择的方案是：

> 用新的最小 Harness 内核替换旧编排；迁移成熟资产；在真实证据通过后切换公共路径；
> 最后删除旧 AgentService、AgentLoop、LoopState 和 LangGraph checkpoint 运行时。

## 4. 架构总图

```text
CLI / Python SDK
        |
   public Agent facade
        |
 RuntimeComposition (assembly only)
        |
   ThreadManager ---------------- RolloutStore
        |                      durable truth + projections
 Session(thread_id)
        |
 run_turn(TurnContext)
        |
 capture StepContext -> ModelClient
        |                 |
        +-- exact context/tool snapshot
        |
 execute tool calls through ToolOrchestrator
        |
 permission -> approval -> sandbox -> runtime
```

依赖只能沿箭头向下。CLI、SDK 和 UI 是客户端；它们不能拥有 Turn 状态、审批状态、
tool-call ledger 或恢复算法。

## 5. 一等对象与唯一所有者

### 5.1 Thread

Thread 是可持久化会话身份，拥有有序 Turn 历史、创建时间、归档状态和默认配置引用。
Thread 不保存活动执行对象，也不直接保存模型 provider 客户端。

要求：

- 一个 Thread 可包含多个 Turn；
- follow-up 在同一 Thread 创建新 Turn；
- fork 创建新 Thread，并明确记录来源 Thread 和截止 Turn；
- `threads.active_turn_id` 与 `head_version` 通过 SQLite CAS/唯一约束保证跨进程最多一个活动 Turn；
- Thread 存在 `running`、`paused` 或 `interrupted` Turn 时，`active_turn_id` 继续被该 Turn 占用，
  不得创建 follow-up；resume 必须继续该 Turn；
- 只有 `completed`、`failed`、`cancelled` 或 `abandoned` terminal Turn 才释放 `active_turn_id` 并允许
  follow-up；用户不再恢复 interrupted Turn 时，必须显式提交 durable `abandon`/`cancel` transition；
- CLI `run` 可以隐式创建只有一个 Turn 的 Thread；
- CLI `chat` 和 SDK 连续对话必须显式复用 Thread；
- Thread ID 和 Turn ID 不得互相填充或互为别名。

workspace realpath 是 Thread 的不可变安全域，不是普通 Turn binding。同一 Thread 不得从 repo A
切换到 repo B；需要换 workspace 时必须创建新 Thread。跨 workspace 的 fork 默认继承空上下文；只有显式
export transaction 才能复制用户逐项选择且 allowlist 允许的 `user_message`。默认排除 plan、reasoning、
tool-derived summary、文件内容、工具结果、artifact、环境信息和 approval。export record 必须保存源 Item ID、
过滤策略 revision、导出内容 hash 和目标安全域，不能用“非敏感摘要”这种未定义分类自动跨域。

### 5.2 Session

Session 是某个 Thread 当前打开的、可重建的内存运行句柄，也是唯一的活执行所有者。它拥有该
Thread 的运行服务、客户端事件订阅、取消令牌和当前活动 Turn 句柄；一个 Session 同时最多运行一个
Turn。ThreadManager 只负责创建、查找、分叉和恢复 Thread，并为同一 thread_id 复用同一个 Session。

Session 不是持久化真相。进程退出后可由 Thread、Turn、Item 和绑定快照重建。

### 5.3 Turn

Turn 从用户输入开始；`paused`、`interrupted` 是仍占用 Thread active slot 的 resumable 状态，
`completed`、`failed`、`cancelled`、`abandoned` 才是 terminal 状态。
Turn 创建时冻结以下绑定：

- model alias 和 provider wire revision；
- workspace realpath；
- permission/sandbox policy revision；
- tool catalog 和 model-visible tool snapshot revision；
- context policy revision；
- knowledge/skill/MCP 配置引用。

follow-up 使用新 Turn 的新绑定；resume 恢复原 Turn 的原绑定。任何后续 `/model` 切换都不得污染旧 Turn。

“冻结”不能只保存一个后来无法解析的 revision 字符串。Rollout 必须保存 secret-free canonical manifest
及其 hash，覆盖 tool schema/effects/resources、每个 runner 的 `runner_implementation_revision`、
permission/sandbox policy、provider wire contract 和 context policy。runner revision 同时绑定 ToolSpec revision
与构建 artifact/source-tree digest，不能只绑定不变的 schema。恢复时 resolver 必须逐字段证明当前实现与
原 manifest 兼容；runner revision 不同即不兼容。若旧 revision 已不可用，已完成 operation
仍复用已存结果，但未执行或等待审批的 operation 必须使原审批失效并 fail loud/reconcile，不能用新版
runner 偷跑旧批准。

### 5.4 Item

Item 是用户与 Agent 交互的语义事实。至少支持：

- `user_message`、`agent_message`、`reasoning_summary`；
- `plan`；
- `model_request`、`model_response`；
- `tool_call`、`tool_result`；
- `approval_request`、`approval_decision`；
- `file_change`、`command_execution`、`verification`；
- `context_compaction`；
- `warning`、`error`、`final_proposal`、`completion_decision`。

Item 具有独立 ID、Thread ID、Turn ID、单调 sequence、kind、状态、父 Item、
producer、结构化 payload 和时间戳。producer 至少区分 user、model、runtime、tool、
orchestrator 和 verifier。模型消息不是唯一 Item 类型，工具和审批也不是塞进 metadata 的字符串。
模型输出只能形成 model-owned proposal/call Item，不能伪造 approval、verification、completion 或
runtime transition Item。

一次 tool operation 只有一个 canonical `tool_call` 和一个 canonical `tool_result`。
`file_change`、`command_execution` 是引用同一 operation ID 的 typed activity/projection，只保存该类型
独有的 diff、进度或显示字段，不能复制一份独立 outcome。`verification` 是 Verifier 对环境的独立观察，
不是 ToolResult 的别名。

### 5.5 Event 与 RolloutRecord

RolloutRecord 是唯一 canonical durable truth，例如 `turn_started`、`item_started`、
`item_completed`、`turn_paused`。它拥有持久 sequence，是审计和恢复依据。Thread、Turn、Item、
Approval 和 ToolOperation 表都是由 RolloutRecord reducer 生成的可重建 projection，不是第二真相。

record envelope 一经发布即冻结，至少包含 `record_id`、Thread/Turn ID、sequence、`record_type`、
`payload_schema_version`、producer、canonical payload bytes/hash 和时间戳。canonical encoding 的字段顺序、
数字/文本表示必须版本化。Reducer 是无时钟、随机数、网络、文件系统或其他 I/O 的纯函数。旧 payload
只能通过按版本注册的确定性 upcaster 读取，或通过追加 migration record 表达；禁止原位重写历史 record。
相同 record prefix 在不同新进程中必须 fold 出相同 projection hash。

每个 Thread projection 必须记录 `(thread_id, applied_thread_sequence)`；store-global projection 才记录
`applied_record_id`。两者都记录 reducer version 和 canonical hash，字段/API 不得复用含糊的
`applied_sequence`。启动、恢复、迁移与
最终验收必须能比较 projection 与 log head；不一致时禁止执行副作用，只允许在独占事务中从 canonical
records 重建。migration 只能追加 migration record 或增加确定性 upcaster 后重建 projection，禁止修改
旧 record 或只修改 projection。

Event 是向客户端发布的通知：

- durable Event 必须引用已提交的 RolloutRecord；
- 文本 token delta 等瞬时 Event 可以不持久化，但断线重连后必须能读取最终 Item 快照；
- Event 不得成为第二份状态，不得由 CLI 自己推断完成状态；
- 客户端可从任意持久 sequence 重放，不重复已确认记录。

RolloutRecord 同时充当 transactional outbox。每条记录有 store-global `record_id` 与
`(thread_id, thread_sequence)`。Thread stream 只接受包含 `thread_id + thread_sequence + store_epoch +
schema_epoch` 的版本化 opaque cursor；global maintenance tailer 只接受 `after_record_id`，两种 cursor
类型和 API 不复用。cursor 由各客户端保存，不写成 Thread 共享 ack。publisher/tailer 只发布已提交记录，
发布前崩溃由重连 replay 补齐，两个订阅者的 cursor 相互独立；epoch 不匹配的旧 cursor fail loud，并返回
可操作的重新同步指令，不能静默从错误位置继续。

### 5.6 RuntimeComposition

RuntimeComposition 是无业务状态的 composition root/factory，防止 ThreadManager 变成改名后的
AgentService。它只负责构造、共享和逆序关闭资源，不创建 Turn transition、不判断完成、不保存消息。

生命周期所有权：

- process/Agent 级：ModelControlPlane、provider client pool、LocalRuntimeManager、冻结 ToolRegistry、
  RolloutStore、MCP connection manager；
- Session 级：事件 tailer/subscription、取消 scope、当前 active Turn handle；
- Turn/Step 级：TurnContext、ContextManager view、ToolRouter snapshot、ModelOperation 和 resource claims；
- ToolRuntime 自己声明是否有 close hook，composition root 统一调用；
- 关闭顺序：停止接收新 Turn -> 中断/协调活动 operation -> 关闭 Session/subagent/MCP/tool runtime ->
  provider clients -> RolloutStore；不关闭用户自行启动的模型进程。

CLI/SDK 只能请求 composition root 打开 public Agent host；不得各自拼装 provider、工具和数据库。
未开始消费的 stream 不分配 Thread/Turn；异常、取消和正常退出都必须经过同一 bounded-close 协议。

## 6. TurnContext、StepContext 与 ContextManager

### 6.1 TurnContext

TurnContext 是 Turn 级不可变快照，包含 Session/thread 身份、Turn 身份和本 Turn 的 durable binding。
它不拥有 ContextManager、ToolRouter、ModelClient 或 ToolOrchestrator；这些服务由 Session 拥有。
任何需要改变模型、权限或工作区绑定的 follow-up 都必须创建新 Turn。

### 6.2 StepContext

StepContext 只对应一次模型 sampling request，不对应 Turn，也不对应“一批工具执行”。它至少包含：

- 当前 TurnContext；
- 当前 context version；
- 当前模型和采样设置；
- 本次请求实际暴露的工具快照；
- 取消令牌和 trace/span 身份；
- 当前资源/环境快照。

StepContext 必须一次捕获后向下传递；context、模型设置、advertised tools 和 ToolCall origin 必须来自
同一个 StepContext，禁止在 request preparation 或工具执行时分别读取可变全局状态。StepContext
不持久化为整个对象；持久化的是重建它所需的版本、hash 和 operation 事实。

### 6.3 ContextManager

ContextManager 是 provider-neutral ContextProjection 的唯一所有者。它从 Item/Rollout 投影构建消息与
context fragments，负责：

- 稳定系统前缀和动态后缀；
- user/assistant/tool-call/tool-result 的合法顺序与配对；
- token/byte 硬预算；
- 大型工具输出 artifact 化；
- compaction；
- provider-neutral context hash 和稳定顺序。

所有权分界：

- Turn 冻结 catalog revision 与 visibility policy revision；
- ToolRouter 为每个 Step 生成实际 exposed-tool snapshot 与 tool snapshot hash；deferred activation
  只能在冻结 policy 允许的集合内产生下一 Step 的新 snapshot；
- ContextManager 生成 context projection 与 context hash，不序列化 provider-specific body；
- ModelRequest 组合 context projection、tool snapshot 和模型设置；
- provider wire 拥有最终请求 body、provider serializer revision 与 wire hash。

每次 model call 持久化 context hash、tool snapshot hash 和 wire hash。ToolCall origin 绑定产生它的
Step/tool snapshot，不能用后续激活的工具集合倒推权限。

禁止：

- Provider、Tool、Memory、Skill 或 CLI 直接修改消息列表；
- 同时维护 `conversation_history`、`messages`、`turn_transcript` 三份权威历史；
- 压缩只改本地统计而不改变真实 provider request；
- 无上限注入文件、RAG 结果、Skill 文本或工具输出。

Compaction 不删除 Rollout 历史。它追加 `context_compaction` Item，记录覆盖的 sequence 范围、
保留事实、摘要、artifact 引用和 context version；后续模型投影使用该 Item 替代被覆盖内容。

必须完整保留：架构/安全约束、文件变更、验证结论、未完成事项、审批和不确定副作用状态。

## 7. Session.run_turn

Praxis 不再引入 `TurnRunner` 对象。Session 的 `run_turn(TurnContext)` 是唯一 Turn 控制流；每次循环执行：

1. Session 从 durable Turn 和当前服务捕获一个 StepContext；
2. 从该 StepContext 取得唯一 ModelRequest，调用 ModelClient；
3. 持久化 ModelResponse Item；
4. 若有 tool calls，交给 ToolRouter/ToolOrchestrator 并持久化结果；
5. 若模型提出完成，交给 CompletionGate；
6. 根据已提交结果继续、暂停或终止 Turn。

`Session.run_turn` 不得直接：

- 拼 provider payload；
- 解析 shell 风险；
- 弹审批 UI；
- 执行工具 runner；
- 操作 SQLite；
- 初始化 RAG/MCP/Skill；
- 编写 CLI 输出；
- 修改 context history；
- 根据计划文字授予权限；
- 把模型的 final answer 直接标为 completed。

## 8. ModelClient 边界

保留现有模型目录、ModelControlPlane、provider wire、usage 和 token accounting 资产。

ModelClient 只接收完全结构化的 ModelRequest，返回结构化 ModelResponse/stream。它不接收
Session、TurnContext 或 LoopState，不查询 ToolRegistry，不加载 RAG，不判断完成，不保存 checkpoint。

Harness 的每次采样统一属于 `agent_step`：同一个响应可以包含 tool calls，也可以提出最终答案。不得把
agent step 复用成旧 RAG 流水线里的 `tool_decision` 或 `final_synthesis`。模型目录声明 context window、
最大输出等能力上界；Turn policy 冻结总 token/step 预算；provider adapter 只组装并序列化二者的有效交集。
切换模型不得要求业务编排改 stage budget。

一个 logical ModelOperation 对应一个 Step，拥有多个 ModelAttempt。logical operation 保存
`active_attempt_id`、generation/version 和唯一 canonical response Item ID；attempt 使用独立状态机：

```text
prepared -> dispatched -> completed | failed | unknown
```

网络调用前先持久化 logical operation ID、attempt ID/generation、model binding、context/tool/wire hash 和
有界 request 引用；
调用成功后才持久化 usage、provider response ID 和 response Item。崩溃发生在 dispatched 后、response
提交前时进入 `unknown`：支持 idempotency key/response lookup/previous-response 的 provider 优先 reconcile；
否则把原 attempt 标为 unknown/abandoned，再以新 attempt 重试，并明确记录可能的重复模型成本。

`unknown` 只表示 transport 中断、超时、取消或进程崩溃导致无法证明 provider 是否形成了终态。provider
明确返回的 rejection、`response.incomplete`、`finish_reason=length/max_tokens` 都是已知结果：必须持久化
response status、reason 和实际 usage，然后确定性失败；不得执行可能被截断的 tool call，不得把同一请求
原样自动重试。若产品以后允许增加输出上限，必须形成新的 policy decision 和新的 canonical request，且仍受
Turn 总预算约束。

只有仍等于 logical operation 当前 `active_attempt_id/generation` 的 attempt 能通过 CAS 提交唯一 canonical
response Item。旧 attempt 在新 generation 已激活或已有 response 后迟到，只能写审计和可能成本，不能
产生 context Item、改变 Step 或覆盖新 attempt 的 usage/response。一个 Step 最终至多一个 canonical
model response。

`max_tokens_total` 是冻结的 Turn 级模型消费预算，不是事后展示字段。provider I/O 之前，
ModelClient 必须根据 durable usage 计算剩余量，并预留预计 input 与 provider output ceiling。
预留失败是已知 preflight rejection：attempt/operation 转 `failed`，usage 保持为空，不得标成
`unknown` 或重试。完成响应的实际 usage 持久化后，Session 在处理 tool call 或 final proposal 前再次
校验累计量；超额必须使 Turn 失败。后者只是 provider 计量漂移的保险，不能代替调用前预留。
刚好用尽时可接受刚收到的响应，但不得再启动下一次 provider 调用。

Provider 模型调用不得承载不可逆业务副作用；所有外部动作仍必须回到 ToolOrchestrator。敏感 header、
secret 和无界原始内容不得进入记录，大型内容通过 content hash/artifact 引用。无效模型别名在 provider
调用前失败，并保留旧 session 选择。

## 9. Tool ACI

工具边界分成四个独立职责。它们可以实现为不可变对象、纯函数或窄端口，不强制为四个长生命周期 class；
验收标准是唯一副作用所有者和可独立验证的输入输出，不是类名数量。

### 9.1 ToolRegistry

保存所有已安装工具及不可变 ToolSpec：名称、schema、效果、资源、并行能力、审批提示、runner 引用。
注册后冻结；重复名称启动即失败。

### 9.2 ToolRouter

根据 Turn/Step 快照生成：

- model-visible specs；
- 可执行 runtime 映射；
- deferred/discoverable 集合；
- tool snapshot revision。

“模型看得到”与“运行时存在”必须分离。未知或未暴露工具调用都 fail closed。

### 9.3 ToolOrchestrator

ToolOrchestrator 是唯一副作用门，固定顺序为：

```text
validate -> resolve resources/effects -> permission -> approval
         -> sandbox selection -> execute -> verify/reconcile -> persist result
```

它负责：

- 结构化参数验证；
- permission 与 hard guard；
- Approval 状态机；
- Seatbelt sandbox；
- timeout/cancellation；
- idempotency 和 operation record；
- 按资源冲突决定并行；
- 可重试错误分类；
- 执行结果截断、artifact 化和结构化错误。

### 9.4 ToolRuntime

ToolRuntime 只执行已获准的具体操作。它不能自行绕过 Orchestrator、改变权限、请求审批、
写 Rollout 或向 UI 发事件。

## 10. Approval、Sandbox 与副作用恢复

Approval 与 Turn、ToolOperation 是三套独立状态机。

Approval：

```text
pending -> approved | denied | cancelled | expired
approved -> invalidated
```

Approval 是通用 InteractionRequest 的一个子类型。Interaction 支持
`tool_approval`、`clarification`、`choice` 和 `tool_reconciliation`：

```text
pending -> resolved | cancelled | expired
```

每个 response 必须绑定 request ID；重复提交同一决定幂等，冲突决定或错误 request ID fail loud。
clarification/choice 只产生新的 user/context Item，不授予工具权限；reconciliation 决定只作用于绑定的
unknown operation。

ToolOperation：

```text
prepared -> awaiting_approval -> ready -> running
        -> succeeded | failed | denied | cancelled | superseded | unknown
unknown -> succeeded | failed | cancelled
```

Turn：

```text
running -> paused | interrupted | completed | failed | cancelled
paused/interrupted -> running
paused/interrupted -> cancelled | abandoned
```

所有参数、resolved resources、effects 和 policy/runner revision 在 operation 创建后不可原位修改。完整的
ToolOperation/Approval 收敛表如下；表外 transition 一律拒绝并记录审计：

| source | durable event | target | 唯一 owner | claim 处理 |
| --- | --- | --- | --- | --- |
| `prepared` | validation/permission denied | `denied` | ToolOrchestrator | 无 claim |
| `prepared` | permitted, no approval required | `ready` | ToolOrchestrator | 无 claim |
| `prepared` | approval required + request created | `awaiting_approval` | ToolOrchestrator | 无 claim |
| `awaiting_approval` | approval approved | `ready` | ToolOrchestrator consuming Interaction | 无 claim |
| `awaiting_approval` | approval denied | `denied` | ToolOrchestrator consuming Interaction | 无 claim |
| `awaiting_approval` | request cancelled/expired | `cancelled` | ToolOrchestrator consuming Interaction | 无 claim；Turn 保持 paused，需新请求或用户 cancel/abandon |
| `awaiting_approval`/`ready` | immutable scope revalidation mismatch | `superseded` | ToolOrchestrator | 无 claim；旧 approval 记 `invalidated`，若继续必须新建 operation/request |
| `ready` | claim CAS acquired | `running` | ToolOrchestrator | 获取 operation generation/fencing claim |
| `running` | confirmed success/known failure | `succeeded`/`failed` | ToolOrchestrator | terminal commit 后释放 |
| `running` | confirmed process-group cancellation | `cancelled` | ToolOrchestrator | 确认停止并提交后释放 |
| `running` | crash/lease loss/unprovable outcome | `unknown` | recovery owner | 保留 claim |
| `unknown` | deterministic reconcile proves outcome | `succeeded`/`failed`/`cancelled` | Reconciler through ToolOrchestrator | terminal commit 后释放 |
| `unknown` | inconclusive/user has not resolved | `unknown` | Reconciler | 保留 claim，Turn paused |

Approval 的 `pending -> approved/denied/cancelled/expired` 由 Interaction owner 以 request-version CAS 提交；
`approved -> invalidated` 只能由执行前 scope/manifest revalidation 触发。所有 Approval terminal 状态不可原位
改成另一决定。generic Interaction cancelled/expired 时，ToolOrchestrator 必须在同一事务中按上表收敛绑定的
`awaiting_approval` operation，不能留下永久悬挂状态。`unknown` reconciliation 也只能经上表产生结果，
不能直接改 Item projection。

每次副作用操作在执行前持久化 operation ID、tool call ID、参数摘要、effects、资源集合、
幂等键、reconcile 策略、claim generation 和 fencing token。

`ready -> running` 必须在副作用发生前通过独立 durable CAS 取得 operation claim。Runner 收到
operation ID、claim generation 和 fencing token；只有仍持有同一 claim 的 worker 可以提交结果。
旧 worker 的迟到结果必须被拒绝并写入审计记录。

恢复规则：

- `prepared`：尚未获准执行；恢复后必须重新经过 validate -> resolve resources/effects -> permission ->
  approval，只能转为 `awaiting_approval`、`ready` 或 `denied`，runner 调用数必须为零；
- 只有 `ready` 可以通过 durable CAS 转为 `running` 并进入 runner；
- `succeeded/failed/denied/cancelled`：复用结果，绝不重跑；
- `running` 后崩溃或 lease 失效：转为 `unknown`，绝不自动重放；
- `unknown` 且可确定 reconcile：检查真实环境后写入结果；
- `unknown` 且非幂等、不可确定：暂停并交给用户，不得自动重放；
- 原审批决定只对绑定的 operation、参数摘要、资源和 policy revision 有效。

新 worker 接管前必须证明旧本地进程已退出/被终止，或外部系统支持 fencing/idempotency/reconcile；
否则保持 `unknown` 并暂停人工协调。数据库 fencing 只能阻止 stale worker 回写，不能撤销已经发生的外部
副作用，因此本契约不对任意外部系统虚假承诺 exactly-once。只有幂等或可确定 reconcile 的具体工具，
才能声明副作用恰好一次。

审批后、执行前必须再次解析参数、realpath、symlink、资源身份、hard guards 和当前 policy。实际目标、
参数 hash、资源指纹或 policy revision 有任何变化，旧 approval 立即失效，runner 调用数保持零，并重新
进入审批或拒绝。批准路径名不等于批准后来替换到该路径的对象。

安全分类和审批只读取结构化工具事实，不读取模型为自己行动辩护的自由文本。

runner invocation 是不可越过的副作用不确定性边界：调用 runner 之前的失败可安全记为 `failed`；一旦
runner 已被调用，verify、normalize 或 result commit 任一阶段失败，只要不能证明真实结果，就必须记为
`unknown`，不得作为普通 retryable error 重试。若已证明 runner 成功但独立 verification 失败，必须持久化
真实 execution outcome，并另写 verification failure/Turn decision；不能把已经发生的执行伪装成未执行。

取消也是持久协议，不是仅设置内存 cancellation token：

- 本地命令必须向完整进程组发送取消并确认退出，之后才能记 `cancelled/failed`；
- 远端或无法确认停止的 operation 记 `unknown`，保留 resource claim 并进入 reconciliation；
- ToolOperation 状态必须先提交，Turn 才能释放 active lease/标记 interrupted；
- resume 对 unknown operation 先 reconcile，绝不重新 dispatch。

## 11. RolloutStore 与事务模型

SQLite 是本地产品的唯一权威存储。建议最小 schema：

- `threads`：Thread 当前投影、`active_turn_id`、head/fork point 和 CAS version；
- `turns`：Turn 当前投影、冻结绑定、lease/CAS/fencing version；
- `items`：Item 当前投影；
- `rollout_records`：追加式事实和全局/Thread sequence；
- `tool_operations`：副作用状态和 reconciliation 数据；
- `approvals`：审批状态；
- `interactions`：approval/clarification/choice/reconciliation 的请求与响应投影；
- `model_operations`：logical Step 调用、active attempt generation 和唯一 response 引用；
- `model_attempts`：每次 provider dispatch、request/response hash、usage/cost 和 unknown/abandoned 状态；
- `resource_claims`：跨 Thread/进程的 canonical resource read/write claim、operation 和 fencing epoch；
- `artifacts`：大型输出和 content hash 引用；
- `projection_meta`：各 projection 的 applied thread position 或 global record position、reducer version 和
  canonical hash。

Artifact 与 SQLite 引用采用 write-before-reference 协议：先在 artifact store 同文件系统写唯一临时文件，
完整写入后 flush/fsync，计算并核对 size/content hash，再以 content hash 为名称原子 rename，必要时 fsync
父目录；只有完成这些步骤，SQLite 事务才可追加引用该 immutable blob 的 RolloutRecord。SQLite commit
前崩溃产生的未引用完整 blob 由基于引用集和保留期的 GC 回收，临时/半文件绝不作为 artifact。读取、
replay、resume 与 `verify` 都必须重新核对 size/hash；缺失或不符时 verify 失败、禁止副作用与 resume，
不能把损坏引用当作空内容继续。

一次状态转换必须在同一 `BEGIN IMMEDIATE` 事务中：

1. 校验预期 version/lease/fencing token；
2. 追加 RolloutRecord；
3. 用唯一 reducer 更新 Item/Turn/Operation 投影和 projection metadata；
4. 提交；
5. 提交成功后发布 Event。

不能依靠进程内 `asyncio.Lock` 保证跨进程正确性。SQLite unique constraint、事务和 CAS
负责重复调用、lease 争用、审批重复提交和 sequence 单调性。

Resource claim 由 ToolOrchestrator 在 runner 前持久化。共享 read claim 可并存，write claim 与重叠
resource 的任何 claim 冲突；claim 绑定 operation/epoch/lease。`conflicts(a, b)` 是 Orchestrator 的冻结
策略：文件路径先解析为稳定 identity，再按相同节点或祖先/后代区间判断冲突；symlink 与 `.git`
gitdir/commondir 别名必须归一到同一 identity；无法静态确定目标的动态 write 保守 claim 整个
workspace/root namespace。`running/unknown` operation
的 claim 不因 Turn lease 过期自动释放，只有 reconcile、确认停止或人工终止后才能释放。该规则同时约束
父 Agent、subagent 和完全独立 Thread。

LangGraph 类型不得出现在新运行时接口、数据库 payload 或公共 API 中。旧 checkpoint 只通过
离线、显式、幂等的 migration command 转换；运行时没有自动 fallback。

RolloutStore 必须提供只读 `verify` 与显式 `rebuild-projections` 操作。`verify` 从 records 独立 fold
目标状态并比较 projection hash；`rebuild-projections` 需要独占数据库事务、不得调用模型或工具。

## 12. CompletionGate、Planning 与验证

模型只能产生 `final_proposal`，不能直接把 Turn 改成 completed。

CompletionGate 读取结构化 GoalSpec、可信 Item、workspace diff、verification 和 safety evidence，输出：

- `accept`：证据满足，Turn completed；
- `continue`：附结构化缺口，让模型继续；
- `pause`：需要用户决定；
- `fail`：安全违规、不可恢复或预算终止。

对于要求修改工作区的任务，至少要有：真实 workspace change、修改后的验证和最终验收证据。
`false_completion` 与 `safety_violation` 永远是一票否决。

Verification Item 只能由注册的 Orchestrator/Verifier producer 产生，必须绑定 verifier revision、
workspace snapshot、最后一次相关 change sequence、命令/检查摘要和 artifact hash。发生在最后一次修改
之前的验证自动失效；模型声称“测试通过”不能生成 Verification Item，也不能让 CompletionGate 放行。

验证必须与任务和被修改资源匹配。测试/lint/类型命令可验证工作区级代码改动；对精确文本替换，
只有修改后成功的 `read_file` 的 canonical read resource identity 与最后一次 write resource identity
相交时才可作为证据，读取无关文件不合格。该规则必须与 Tool ACI 自身的验证指引一致，且不得从
模型自由文本中推断验证成功。

Plan 是可选 Item，用于透明展示策略和进度：

- 复杂任务应产生和更新 Plan；简单任务可以跳过；
- Plan step 状态必须通过事件对客户端可见；
- Plan 不授予权限、不决定 tool visibility、不覆盖 GoalSpec、不证明完成；
- resume 不得因为计划变化而重放未知副作用。

## 13. 扩展边界

RAG、MCP、Skill 和 subagent 只能通过以下两种方式进入 Harness：

1. 向 ToolRegistry 提供 ToolSpec + ToolRuntime；
2. 向 ContextManager 提供有来源、可信度、生命周期和硬预算的 ContextFragment。

它们不得导入 Session 内部状态或 RolloutStore 实现。

特别规则：

- RAG 是显式、lazy knowledge provider，不是默认主路径；
- workspace MCP 启动命令在启动外部进程前就受 trust/approval 约束；
- Skill 文本是上下文，不是额外权限；
- subagent 拥有独立 Thread/Turn/Context，通过持久 Item/消息通信，不共享父 LoopState；
- 共享工作区的写操作通过资源声明和 Orchestrator 冲突控制。

## 14. 公共 API 目标

CLI 与 SDK 均通过 public `Agent` facade 调用 ThreadManager，不直接构造 Session 或 Store。

目标语义：

```text
agent run TASK             -> create Thread + Turn
agent chat                 -> create/resume Thread, create one Turn per message
agent resume TURN_ID       -> resume the same paused/interrupted Turn
agent models               -> inspect/switch future Turn model selection
agent knowledge            -> manage explicit knowledge providers
agent threads list/show     -> inspect durable Threads and Turns
agent threads fork/archive  -> explicit history fork and lifecycle control
```

Python SDK：

```python
agent = Agent(...)
first = await agent.run("inspect this repository")
second = await agent.run("continue", thread_id=first.thread_id)

paused = await agent.run("apply the approved change")
resumed = await agent.resume(paused.turn_id, "allow_once")
```

`AgentResult` 必须同时暴露真实 `thread_id` 与 `turn_id`。当前 `previous_turn_id` 可由一个
显式兼容适配器解析，但要标记 deprecated；适配器只转换请求，不形成第二套生命周期：

- predecessor 是该 Thread 当前 terminal head：在同一 Thread 创建 follow-up；
- predecessor 不是当前 head：按该 Turn 的历史截止点 fork 新 Thread，不能偷带后续 Turn；
- predecessor 仍 running/paused/interrupted：拒绝 follow-up，要求 resume、cancel 或 abandon 同一 Turn。

### 14.1 公共资产保留矩阵

新内核必须保持或显式改进现有公共能力，不能因为旧类型被删除而静默丢字段：

| 当前公共资产 | 新权威来源 |
| --- | --- |
| `answer`, `status`, `stop_reason` | final agent Item + Turn projection |
| `turn_id`, 新增 `thread_id` | Turn/Thread identity |
| `files`, `workspace_path` | Thread workspace + Turn attachment Items |
| `tool_calls` | canonical tool_call/tool_result Items |
| `evidence`, `citations`, `groundedness`, `insufficient_evidence` | knowledge tool results + verifier projection |
| `usage`、model/tool latency、cache accounting | ModelOperation/ToolOperation records |
| `diagnostics` | redacted warning/error RolloutRecords |
| `pause`, `needs_user_input` | pending InteractionRequest |
| `plan`, `plan_events` | plan Items/lifecycle records |
| text/tool/plan/approval/recovery/abort/budget events | committed records + transient deltas |
| `max_turns`, `max_tokens_total`, timeout | frozen Turn policy + operation budget records |

附件与临时 workspace 也有明确 owner：外部附件按 Turn 复制到 content-addressed artifact store，workspace
内附件保存相对路径、identity 和 hash；两者都可跨进程 resume。自动临时 workspace 在存在 resumable Turn
期间不得删除，只有 terminal retention policy 到期且无 Thread/artifact 引用时才能垃圾回收。

## 15. 迁移、切换与删旧

### 15.1 保留资产

- ModelCatalog、ModelSessionState、ModelPolicy、ModelControlPlane；
- provider wires、usage/token accounting、模型错误诊断；
- Tool/ToolSpec、内置工具 runner、MCP/RAG/Skill adapters；
- permission hard guards、Seatbelt 策略、安全测试；
- CLI 命令、Python `Agent` facade 和 `AgentResult` 用户体验；
- benchmark manifest、真实任务、恢复和安全 fixtures。

### 15.2 替换资产

- `AgentService` 的生命周期所有权 -> ThreadManager/Session；
- `AgentLoop` -> `Session.run_turn(TurnContext)`；
- `LoopState` -> Thread/Turn/Item + ContextManager + ToolOperation 投影；
- `ToolExecutor` -> ToolOrchestrator + 现有 ToolRuntime runners；
- TurnStore + LangGraph checkpoint -> RolloutStore；
- StreamEvent 全局 sequence -> committed rollout sequence + Item lifecycle events。

### 15.3 旧状态联合迁移

公共切换前必须完成独立迁移阶段，联合读取旧 TurnStore、LangGraph checkpoint、HumanInputRequest、
ToolExecutionRecord、RuntimeBinding、消息和附件；只迁 TurnStore 不合格。

旧历史是每个 Turn 最多一个 predecessor、但 terminal Turn 可有多个 successor 的森林。确定性映射规则：

1. 每个 legacy root 创建一个 primary Thread；
2. successor 按 `(created_at, turn_id)` 排序，第一个沿当前 Thread 继续；
3. 每个额外 successor 创建 fork Thread，只记录不可变 `forked_from_thread_id` 和
   `fork_point_turn_id`；ContextManager 沿该 fork edge 读取祖先前缀，禁止复制 `user_message`、
   `tool_result` 等 semantic/canonical Item；
4. 对每个分支递归应用同一规则，Turn ID 和原 Item 所有权保持不变；若为性能必须物化上下文，只能生成
   非语义 `fork_context_snapshot` artifact，逐项记录源 Item ID/hash，不能伪造 Item 的产生 Turn；
5. 迁移校验必须证明每个 legacy Turn 恰好映射一次、每条 predecessor 历史等价且没有跨分支偷带消息。

状态映射：

- completed/failed 保留 terminal 状态和结果；
- paused 联合迁移 Interaction、approval、operation 和 frozen manifests；兼容时可 resume，不兼容时保持
  paused 并失效旧 approval，要求 reconciliation/安全终止；
- interrupted 保留；
- running 在 maintenance quiesce 后，已过期/无存活 worker 的 Turn 转 interrupted，其 running operation
  转 unknown；无法证明 worker 已停则迁移暂停；
- migrated unknown operation 永远不自动 dispatch。

迁移工具必须支持：maintenance/exclusive lock、源数据库备份、dry-run 映射报告、schema version、单事务
写入、rollout/projection verify、失败回滚和幂等重跑。旧数据库保持只读且不删除。迁移成功与 public cutover
之间禁止旧 runtime 再写；否则迁移作废并重跑。

### 15.4 纵向切片顺序

每个切片都必须从 public CLI/SDK 穿透真实新内核，不允许只测试私有 helper：

1. Thread/Turn/Item/RolloutStore + 普通模型回答；
2. 只读工具调用 + ContextManager 配对与 replay；
3. workspace write + approval + Seatbelt + verification；
4. crash injection + operation reconciliation；
5. compaction + follow-up/resume + model/tool snapshot；
6. RAG/MCP/Skill/subagent adapters；
7. 旧状态联合迁移 dry-run/真实验证；
8. CLI/SDK 全量切换；
9. 删除旧编排、LangGraph 依赖和兼容 fallback。

迁移期间新旧代码可以在 feature branch 同时存在，但公共候选版本只能选择一条主路径。
不允许生产 runtime 自动 fallback 到旧内核。影子比较只能读取同一冻结输入，不得重复执行副作用。

## 16. 可执行验收矩阵

每条均需要直接证据；“有测试”“文件变小”“类已改名”都不是替代证据。

### ARCH：所有权与依赖

- **ARCH-01**：Thread、Session、Turn、Item、TurnContext、StepContext、ContextManager、
  ToolRegistry、ToolRouter、ToolOrchestrator、Approval、Sandbox、Event、Rollout 各有唯一所有者。
  - 证据：架构 contract test + import-linter + 真实调用图审查。
- **ARCH-02**：core/context/model/tools/extensions/storage 均不导入 Session 内部状态。
  - 证据：`uv run lint-imports` 和针对禁止导入的测试。
- **ARCH-03**：public Agent 和 CLI 调用同一 ThreadManager/Session/TurnContext/StepContext 路径。
  - 证据：两个入口捕获相同的 Thread/Turn/Item 序列。
- **ARCH-04**：RuntimeComposition 只装配/关闭资源，不创建 transition；ThreadManager 不构造 provider、
  tool runtime、MCP 或数据库 schema，也不判断完成。
  - 证据：装配 owner contract、异常/取消/未消费 stream 的 bounded-close 集成测试。

### CTX：模型可见上下文

- **CTX-01**：tool call/result 配对、顺序和 request hash 在重启前后相同。
- **CTX-02**：每类注入都有 hard cap，单个 Item 不得无限增长。
- **CTX-03**：compaction 改变实际 provider request，同时保留关键约束、diff、验证和未决状态。
- **CTX-04**：稳定前缀未发生语义变化时保持字节稳定。
  - 证据：捕获真实 provider request body 的集成测试，不接受只断言内部计数器。

### MODEL：模型调用事务

- **MODEL-01**：provider 调用前存在 prepared ModelOperation 和 request/context/tool/wire hash。
- **MODEL-02**：dispatched 后、response commit 前崩溃进入 unknown；支持查询的 provider reconcile，
  不支持的 provider 用新 attempt 重试并保留 unknown 成本诊断。
- **MODEL-03**：usage/response ID 只来自已收到终态的 attempt；unknown attempt 不伪造零 usage 或 response。
  冻结的总 token 预算在 provider I/O 前预留，并在实际 usage 提交后再次校验；preflight rejection
  没有 provider response/usage。`max_tokens/response.incomplete` 必须保存为已知 incomplete response，
  公开诊断后确定性失败，不得进入工具处理、CompletionGate 或 unknown retry。
- **MODEL-04**：provider wire 是唯一最终 body/wire-hash owner，ContextManager 与 ToolRouter 不序列化 body。
- **MODEL-05**：attempt A unknown/abandoned 后 attempt B 成功提交，A 的迟到响应只能形成审计/成本记录；
  logical operation CAS 保证最终只有一个 canonical model response Item。

### ACI：工具正确性

- **ACI-01**：未注册、未暴露、schema 错误、permission 错误全部 fail closed，runner 调用数为零。
- **ACI-02**：审批 scope 绑定 operation、参数 hash、资源与 policy revision。
- **ACI-03**：并行只发生在声明资源无冲突且工具允许并行时；一个只读失败不取消无关只读调用。
- **ACI-04**：错误、截断、artifact、retryable、operation ID 都是结构化字段。
- **ACI-05**：approval 后、runner 前重新解析 realpath/symlink/resource/policy；变化时旧 approval 失效，
  runner 调用数为零。
- **ACI-06**：两个进程/Thread 对同一 canonical resource 的冲突 write claim 只有一个进入 runner；
  兼容 read claims 可并存；parent/child path、symlink alias、gitdir/commondir alias 和 unknown dynamic
  write 均按保守冲突规则处理。
- **ACI-07**：operation 的参数、资源、effect 和 revision 不可原位修改；scope 变化使旧 operation
  superseded、旧 approval invalidated，继续执行必须创建新 operation/request。
  - 证据：ToolOrchestrator 集成测试和真实工具 runner 反例。

### SAFE：真实安全边界

- **SAFE-01**：真实 macOS Seatbelt 阻止 workspace 外写入和任意 `.git` 写入。
- **SAFE-02**：destructive/workspace-write/network 能力缺失时失败关闭，且不发生副作用。
- **SAFE-03**：提示注入不能通过模型自由文本改变 permission/approval 结果。
- **SAFE-04**：`safety_violation=0`，任何一例都阻断 cutover。
- **SAFE-05**：`pause -> 替换 symlink/.git gitdir/commondir -> resume` 必须拒绝，runner 调用数为零。
  - 证据：`uv run pytest -q tests/agent/test_run_command_safety.py` 加真实文件系统前后状态。

### REC：持久化与恢复

- **REC-01**：在 Turn 创建后崩溃可恢复或明确 interrupted。
- **REC-02**：model response 持久化后、tool dispatch 前崩溃不丢调用。
- **REC-03**：approval 后、execute 前崩溃不重复请求审批。
- **REC-04**：副作用后、result 持久化前崩溃进入 reconcile/unknown，不盲目重放。
- **REC-05**：result 持久化后、下一次模型调用前崩溃复用结果。
- **REC-06**：final commit 后客户端断线，重连可读取同一完成结果。
- **REC-07**：两个进程竞争同一 Turn 只有一个获得 lease，审批重复提交幂等。
- **REC-08**：旧 worker lease 失效但仍存活时，新 worker 不执行副作用；stale worker 不能提交结果。
- **REC-09**：projection 与 rollout records 人为制造差异后，verify 必须失败，rebuild 后 hash 一致。
- **REC-10**：取消本地进程确认整个进程组退出后才释放 claim；无法确认的远端取消转 unknown 且 resume
  不重新 dispatch。
- **REC-11**：prepare 后、approval/permission 完成前崩溃，恢复后重新经过完整门禁，runner 调用数为零；
  只有 ready operation 能 CAS 到 running。
- **REC-12**：runner 成功后分别在 verify、normalize、result commit 注入故障；不能证明结果时进入
  unknown，已证明执行成功但验证失败时保留 execution outcome 且不重复副作用。
- **REC-13**：同一冻结 record prefix 在两个 fresh process 中经版本化 upcaster/reducer replay 后得到同一
  hash；历史 records 没有被原位重写。
- **REC-14**：artifact 临时写、fsync/rename、SQLite 引用提交各边界故障注入后，不存在 record 指向
  缺失/半写 blob；篡改或删除 blob 时 verify 失败且禁止 resume/副作用，未引用 blob 可安全 GC。
  - 证据：真实 SQLite 文件、杀进程/故障注入、side-effect counter 和重开进程测试。

### EVT/API：协议一致性

- **EVT-01**：每个持久 Item 有单调 sequence 和完整 lifecycle。
- **EVT-02**：Thread 客户端从任意已确认 opaque thread cursor 重放后得到相同最终投影；跨 Thread 交错
  提交不会混淆相同 thread_sequence，epoch 不匹配 fail loud。
- **EVT-03**：publisher 在 commit 后、publish 前崩溃，重启 tailer 仍从 transactional outbox 补发；
  两个 Thread 订阅者使用独立 opaque cursor，不互相吞事件；global tailer 仅使用 `after_record_id`。
- **API-01**：run/chat/resume 的 Thread 与 Turn 语义不同且可观察。
- **API-02**：CLI 与 SDK 的 status、pause、tool calls、usage、diagnostics 和 IDs 等价。
- **API-03**：无效模型别名在 provider 前失败，列出可选项并保留旧选择。
- **API-04**：同一 Thread 的跨进程并发 Turn 创建只有一个成功；paused/running/interrupted Turn 均占用
  active slot 并阻止 follow-up，只有 durable resume/cancel/abandon 能推进或释放。
- **API-05**：从非 head predecessor 继续会 fork 到准确历史截止点，不包含后续 Turn。

### INT：人机交互

- **INT-01**：approval、clarification、choice、tool_reconciliation 都通过同一 Interaction lifecycle 持久化。
- **INT-02**：重复相同 response 幂等；错误 request ID、冲突 response 和对非 pending request 的 response
  fail loud，且不调用模型或工具。
- **INT-03**：clarification/choice 不授予权限；reconciliation 只处理绑定的 unknown operation。
- **INT-04**：approved scope 复验失效、pending request expired/cancelled、unknown deterministic reconcile
  都按冻结 transition table 收敛；terminal decision 不可改写，不留下悬挂 operation/claim。

### EXT：可选能力边界

- **EXT-01**：不配置 knowledge 时不初始化 RAG，也不暴露 RAG 工具。
- **EXT-02**：MCP 启动前完成 workspace/config trust 决策。
- **EXT-03**：Skill 只能增加上下文/工具，不能增加权限。
- **EXT-04**：subagent 有独立 Thread/Turn/Context 与持久消息，不共享父可变状态。

### ASSET：现有产品资产

- **ASSET-01**：AgentResult 的 answer/status/files/tool_calls/evidence/citations/usage/diagnostics/stop_reason/
  pause/workspace/groundedness/insufficient_evidence/plan/plan_events/needs_user_input 均有新来源和 public 回归。
- **ASSET-02**：run/chat/resume/models/knowledge/threads 的 CLI 与 SDK 等价，错误保持简短、可操作、已脱敏。
- **ASSET-03**：外部附件和临时 workspace 可跨进程 resume；resumable Turn 存在时不会被清理。
- **ASSET-04**：RAG evidence/citation、planning、clarification、max_turns/max_tokens、streaming 和资源关闭
  均通过 public-path tests，不接受直接调用内部 reducer 代替。

### MIG：旧状态联合迁移

- **MIG-01**：legacy 分叉 Turn forest 按确定规则生成 primary/fork Threads，每个旧 Turn 恰好映射一次，
  fork history 截止准确且 canonical semantic Items 不被复制或篡改所有权。
- **MIG-02**：completed/failed/paused/interrupted/running-expired 与消息、附件、binding、interaction、
  approval、tool record/checkpoint 联合迁移，无 orphan。
- **MIG-03**：migrated paused Turn 要么在精确兼容 manifest 下安全 resume，要么保持暂停并使旧 approval
  失效；绝不重放 pending/unknown 副作用。
- **MIG-04**：dry-run、backup、maintenance lock、schema version、verify、rollback 和幂等重跑有真实
  SQLite fixture 证据；旧数据库未删除、cutover 后无旧 writer。

### MAN：证据与防投机

- **MAN-01**：存在 machine-readable acceptance manifest，逐条映射本节所有 ID 到 command、artifact、
  evidence kind、source HEAD、model、task/evaluator revision 和 hash。
- **MAN-02**：manifest 有两个不可混淆的模式：`validate --schema` 允许 requirement 为
  `planned/pending`，但仍要求完整 ID、command 和 expected evidence contract；`audit --final` 才要求当前
  HEAD 的 artifact/hash 和实际执行结果。schema 通过不是完成证据，任何输出都必须显式标出模式。
- **MAN-03**：schema validator 拒绝缺 ID、重复 ID、未知 ID、无 command 或缺 expected evidence
  contract；final audit 额外拒绝 planned/pending、过期 HEAD、缺 artifact、hash 不符和只声明未执行
  benchmark 的证据。
- **MAN-04**：CompletionGate 拒绝 model/tool 伪造的 verification，以及发生在最后一次 change 之前的验证。
- **MAN-05**：最终 audit 只接受 acceptance runner 真实执行 manifest 中精确 argv 后生成的
  receipt/hash；receipt 绑定 requirement ID、command hash、source HEAD/tree、exit code 和 artifact
  path/hash，生成 achieved/missing/contradicted 状态；任何 missing/indirect/inconclusive 项都阻止完成。

`validate --schema` 只验证证据合同；`run --requirement ID` 才执行该 ID 的精确 command，并在
`PRAXIS_ACCEPTANCE_ARTIFACT_ROOT` 下核对预期 artifact 后生成 runner receipt。正式执行必须把
带 evidence 的 manifest 副本和 artifact root 放在被验收源码树之外，避免证据文件自身改变
`source_tree` 造成循环绑定。`audit --final` 重新计算当前源码身份并校验 receipt/artifact，
手工把 `state` 改成 `achieved` 不能通过。

### PROD：真实任务门禁

在同一冻结源码、任务、模型和 evaluator 上比较旧内核与新内核：

- **PROD-01**：`evals/code_agent/benchmark_v1.json` 30 个任务 release shape 有效，任务/环境/evaluator
  revision 和 hash 冻结。
- **PROD-02**：real-model first-pass success >= 70%，标记为 cross-layer 的任务 success >= 50%。
- **PROD-03**：同一冻结条件下新内核通过数不得低于旧内核。
- **PROD-04**：`safety_violation=0` 且 `false_completion=0`，任一非零都阻止 cutover。
- **PROD-05**：每个失败都有 outcome、首个错误归因、rollout 和 side-effect evidence；最终证据包含
  manifest validate、真实 run 与 release gate，不能只执行 shape validator。

manifest validation 不是 benchmark execution。最终证据必须同时包含 manifest validate、真实 run 和
release gate；具体 run/gate 命令由 acceptance manifest 固定，并记录模型、环境、源码 tree、task/evaluator
revision 和原始 rollout artifact。验证命令至少包含：

```bash
uv run python scripts/agent_code_benchmark.py validate \
  evals/code_agent/benchmark_v1.json --repository . --release
```

**PROD-06**：以下五个关键 public-path 场景分别连续三次 3/3 成功：

1. 单文件缺陷修复；
2. 跨文件功能修改；
3. 数据文件读取与产物生成；
4. 需要审批的 workspace write/command；
5. 注入崩溃后恢复并完成；测试工具必须是幂等或可确定 reconcile，并证明副作用恰好一次。

**PROD-07**：另有一个不可幂等、不可确定 reconcile 的故障 fixture：崩溃后必须暂停，保留 unknown operation/resource
claim，且 runner 不重放。该场景通过“安全暂停”而不是“强行完成”验收。

**PROD-08**：成功必须包含真实 diff/产物、修改后验证和隐藏验收，不接受 fake model 作为质量结论。

### CUT：旧内核退出

- **CUT-01**：公共 CLI/SDK 不导入或构造 `AgentService`、`AgentLoop`、`LoopState`。
- **CUT-02**：生产源码删除旧编排和双存储同步代码。
- **CUT-03**：删除 LangGraph checkpoint 依赖；新数据库 payload 无 LangGraph 类型。
- **CUT-04**：不存在第二套 Registry、Executor、Event 或运行时 fallback。
- **CUT-05**：旧 checkpoint migration 是离线显式工具，fixtures 证明幂等；运行时不自动读取旧格式。
- **CUT-06**：`rg`、import-linter、wheel smoke 和 public E2E 共同证明旧路径不可达。
- **CUT-07**：暂停 Turn 的 tool/policy/provider revision 无法精确解析或安全 reconcile 时 fail loud，
  原 approval 失效且 runner 调用数为零；不得用当前新版 runner 偷跑旧批准。
- **CUT-08**：行为 owner tests 证明只有 Rollout reducer 写 projection、只有 ToolOrchestrator 调
  ToolRuntime、只有 provider wire 生成最终 body；把旧逻辑改名搬到 HarnessService 不能通过。

## 17. 完成定义

只有以下条件同时成立，才能把本架构目标标为完成：

1. 第 16 节所有 ID 均有指向当前 HEAD 的证据；
2. 新内核是 CLI/SDK 唯一运行时；
3. 真实任务、恢复和 macOS 安全门禁全部通过；
4. 旧编排和 LangGraph checkpoint 已删除；
5. Git diff、依赖图和包构建证明没有把复杂度迁到兼容层；
6. 原始 dirty main worktree 未被修改；
7. 任何缺失、间接或仅由 fake 支持的证据都按“未完成”处理。

在这些条件之前，允许阶段性交付和暂时双代码存在，但禁止声称“全面重构完成”或“生产就绪”。
