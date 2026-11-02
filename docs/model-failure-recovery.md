# 模型错误与重试

Provider 层负责报告一次尝试的结果，Runtime 负责决定是否重试。OpenAI-compatible SDK 的
`max_retries=0`，一次 `model_attempt_dispatched` 对应一次 SDK 请求；SDK 不再在 Runtime
计数之外重复发送。模型换成其他 provider 时也遵循同一结构化错误契约。

## 错误类别和动作

| 类别 | 自动处理 | 费用处理 |
| --- | --- | --- |
| `authentication` / `permission` | 终止；静态 API key 没有可调用的刷新能力，不重复发送 | 已确认拒绝不保留 tokens 预留 |
| `invalid_request` / `quota_exhausted` | 终止；429 的结构化 quota code 优先判为额度耗尽 | 已确认拒绝不保留 tokens 预留 |
| `context_overflow` | 原有有界上下文压缩恢复；本地预检和远程 HTTP 拒绝分别记账 | 远程拒绝计一次请求，本地预检不计 provider 调用 |
| `rate_limited` | 无输出时有界退避，遵守服务端 Retry-After | HTTP 拒绝计一次请求；SSE 错误保留未知费用 |
| `server_error` | 无输出且允许重试时有界退避；501/505 和 `x-should-retry: false` 不重试 | 保留原尝试的未知费用，重试另行预留 |
| `transport_not_sent` | 有 ConnectError/ConnectTimeout 证据时有界重试 | 不计 provider 请求消耗；已开始的 retry 消耗一次重试额度 |
| `transport_unknown` | 暂停，用户显式选择 retry/abort | 保留未知费用，不自动退款 |
| `protocol_error` | 无输出时终止；有部分输出时暂停 | 保留未知费用 |

正文、reasoning、工具名或参数任一部分已开始输出，都禁止透明重放。未获得完整响应并提交
规范结果前，不执行其中的工具提议。缺失 provider 终态属于未知结果，不能按成功处理。
错误类别由 SDK 类型、HTTP status 和结构化 code/type 推出，不根据错误消息文本猜测。

`ModelFailure` 分别记录 provider outcome 和 billing outcome：例如 HTTP 503 已确认请求失败，
却不能证明没有计费。公共 SDK 的 `AgentResult.model_error` 包含类别、阶段、安全消息、
HTTP status、provider code、request id 和 Retry-After 的绝对时间。成功恢复后该字段为 None，
失败历史保留为 recovered diagnostic。CLI 和交互终端展示同一错误结构，不要求 verbose。
原始响应正文、认证 header 和异常字符串不写入规范错误消息。

## 冻结的自动重试预算

新 Turn 的 `binding_manifest.model_retry_policy` 默认：

- 每个逻辑 model operation 最多自动重试 2 次，即最多 3 次尝试。
- 从第一次 dispatch 起，120 秒内允许启动下一次自动尝试。
- 指数退避使用 full jitter，基数 500 ms，上限 20 秒。
- 服务端的 `retry-after-ms`、`Retry-After` 整数秒或 HTTP-date 会延长等待；不截短服务端建议。
  超过原截止时间就停止，不等到 deadline 后再发送。

120 秒约束自动 dispatch 的准入时间；已经发送的请求仍受冻结的 provider timeout 约束，
这不是一次逻辑调用总运行时长的硬上限。原 Turn 没有该 policy 字段时，不因升级而获得自动重试。
显式 retry 不重置已经使用的自动额度和原截止时间。取消一个已排定的自动重试也不返还自动次数。

每次失败结算和 `model_retry_scheduled` 同事务提交。绝对 due/deadline、选定 jitter、原 attempt
和 generation 都写入日志。进程重启直接恢复该时间表，不从头等待、不重新抽 jitter。
当前进程用 monotonic clock 等待，同时检查 epoch deadline；发现时钟倒退停止自动发送。
`model_retry_exhausted`、`model_retry_deadline_exceeded`、`model_retry_clock_invalid`、
`model_retry_budget_exhausted` 区分运行时停止原因，原 provider failure 仍独立保留。

恢复同一 operation 直接读取冻结请求，校验 request/context/tool/wire 四个 hash，不重做 prompt
投影，也不创建新的推理 Step。首发与恢复发送都规范化 JSON 和扩展参数顺序，实际 HTTP 请求体一致。

## 派发前的边界

准备和实际 dispatch 都验证用户输入 revision、取消状态、摘要源 hash、冻结截止时间和新预算预留。
等待期间新输入使旧请求失效；取消立即进入正常清理。摘要源改变时旧 job 标为 superseded，释放未用
continuation/摘要预留；旧的未知调用费用仍保留。Turn 失败/取消同样释放尚未使用的 job holds。

未知费用计入 exposure，不能通过自动或手动重试消除。新尝试必须重新预留 tokens、费用、请求及 retry
额度，准入失败则没有新 HTTP 请求。准备后尚未派发即取消的尝试完整释放预留，不计一次实际 retry。

写入协议升级为 payload schema 5，读取支持 1–5。SQLite writer gate 拒绝旧版本继续写入，连升级前
已打开的旧连接也受到约束。升级前按[开发文档](development.md)备份数据库与 artifacts。

## 源码对照边界

Codex 对照固定在 `openai/codex` commit `a4c9194247c6a1e79aa3e72cc6cd77238f2ed354`：
`codex-rs/model-provider-info/src/lib.rs`、`codex-rs/codex-api/src/api_bridge.rs`、
`codex-rs/protocol/src/error.rs`、`codex-rs/http-client/src/retry_after.rs` 及 turn 执行链。
借鉴明确的错误类别、单层 retry owner、绝对 Retry-After、响应与工具结果提交边界。

Claude 对照来自本地历史镜像 `brath-claude-code` commit
`276b8e6a0160939c74f04074a5e10ec1ba76982b`，`src/services/api/claude.ts` 中 SDK `maxRetries: 0`、
外层 `withRetry` 以及部分流重试警告。其版本来源和构建完整性未经确认，不能声称是当前官方完整源码。
不照搬特定产品的无限重试、额度重置或未提供能力支持的认证刷新逻辑。

本次验收仅覆盖相关 provider、预算、恢复和公共入口，不要求 RAG 等无关模块全量回归。
真实云验收以透明本地代理注入一次 HTTP 429，后续转发官方 DeepSeek API；同时核对实际请求数、
冻结请求字节、canonical replay、预算清理及模型工作区之外的功能断言。注入故障不代表真实云故障。
