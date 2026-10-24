# 同一 Turn 的恢复与权限边界

`Agent.resume(turn_id, action)` 和 `agent recover TURN_ID --action ACTION`
恢复同一个持久化 Turn。`previous_turn_id` 则为新 Turn 提供历史上下文；
`agent resume THREAD_ID` 重新打开聊天，下一条消息也会创建新 Turn。

## 关闭终端后继续聊天

`agent resume THREAD_ID`、`agent chat --continue` 和聊天内 `/resume` 都先只读展示历史，
不会因为选择了会话就调用模型。直接发送新消息时，运行时先取得该 Thread 的独占执行锁：

- 原进程仍在执行：拒绝另一执行循环，提示回到原终端，交互终端保留草稿。
- 原进程已退出：将未提交结果的模型请求记为 unknown，保留其未知费用与历史记录；
  安全结束旧 Turn，在同一 Thread 创建新 Turn，带上原历史和这次新消息。
- 尚未启动的工具调用：记录“未执行”的规范工具结果，闭合消息组，不执行旧调用。
- 工具已开始但结果未知：逐项进入结果核对，保留暂停，不自动重跑、取消或假报完成。
  已提交执行结果但缺少规范 ToolResult 的情况也需要核对。
- 待审批、待澄清的会话仍走原交互流程；打开历史不会产生授权。

执行锁由操作系统管理，进程退出即释放；新 Turn 记录该所有权协议版本。
旧版本运行中的记录没有这一承诺，只能在已有模型或工具租约过期后接管，
不能仅凭取得新锁就推断旧进程退出。锁文件不能删除，否则会破坏跨进程互斥。

普通新消息不会重试旧的压缩请求，也不会增加旧 Turn 的冻结摘要调用额度。
显式 `agent recover TURN_ID --action continue` / `retry` 仍恢复同一 Turn，受原预算、
模型、权限与未知结果限制。一次取消请求只在最近的 `turn_resumed` 之后有效；
明确恢复后仍可继续接收中途输入，并可再次取消。

这套聊天边界对照了 [Codex thread/resume 与 turn/start 的公开契约](https://developers.openai.com/codex/app-server)、
[Codex 历史重建源码](https://github.com/openai/codex/blob/8c076942cb70fba8e5c4732c23f1faabc4aed484/codex-rs/core/src/session/mod.rs#L1676)
以及 [Claude Code 的已保存会话与活跃后台会话区分](https://code.claude.com/docs/en/sessions)。
Claude Code 的公开 GitHub 仓库未提供这里所需的完整运行时源码，因此没有将其内部实现作为已核实证据。

新 Turn 对已知 `incomplete/max_output_tokens` 允许一次完整步骤重试，额度按整个 Turn 冻结，
原步数、tokens 和费用预算照常扣除。截断的正文、reasoning 和工具调用不作为后续 assistant 消息重放，
已完成工具结果保留；content_filter、未知终态和 summary 不走自动重试。重开进程不会重置额度，
缺少该策略字段的旧 Turn 保持旧契约。缺失 provider 终态仍进入 unknown 的人工恢复路径。

## 先恢复能力边界，再打开资源

新建 Turn 时，binding manifest 会记录
`mcp_policy.workspace_discovery_enabled`。恢复时，Session 在打开模型与工具资源之前
将该能力与当前 Agent 设置收紧：

| 原 Turn 的 MCP 发现 | 当前 Agent 的 MCP 发现 | 恢复时的 MCP 发现 |
| --- | --- | --- |
| 关闭 | 开启或关闭 | 关闭 |
| 开启 | 关闭 | 关闭 |
| 开启 | 开启 | 仍受原权限快照与当前权限共同约束 |

因此，原 Turn 禁用了 MCP 时，即使新进程使用构造函数默认值
`enable_workspace_mcp=True`，也不会重新解析 MCP 配置、要求配置受信任或打开服务器。
恢复过程中出现的新工作区配置不能为该 Turn 开启这项能力。
Session 调整的是 Agent 的浅拷贝，不会改写调用者的 Agent 默认值，也不会改写原 binding。

MCP 发现已开启时，发现阶段使用原 Turn 权限快照与当前有效权限的交集；
执行阶段继续执行权限、硬性检查和审批流程。原策略的禁止项与当前新增的禁止项都不能
通过恢复扩大。当前显式禁用 MCP 仍然有效。缺少该字段的旧 binding 沿用兼容路径，
不会凭空推断它曾禁用 MCP。

## 预授权与单次审批

`allow_write_tools`、`allow_execute_tools`、`allow_web_tools` 是运行时预授权，
会被记录在 Turn 的工具策略中。它们分别减少工作区写入、进程执行和指定公网读取工具
的普通审批；它们仍受当前权限上限、明确禁止项、硬性检查与工具自身约束限制。

`allow_execute_tools=True` 和 `allow_write_tools=True` 不会取消
`ToolEffect.DESTRUCTIVE` 的审批。`auto_approve_sandboxed` 仅适用于满足既定限制的
只读托管沙箱进程，也不覆盖破坏性效果。MCP action 保留其确认要求。
这些开关没有提供“无条件执行一切”的授权。

`allow_once` 针对已持久化的那次待审批工具操作。恢复时先重新解析参数、目标和权限，
比较审批范围；范围变化会使旧审批失效并返回暂停状态。单次同意不能绕过新的工具禁止、
文件系统限制或硬性检查。对于进程与网络同时存在的操作，网络可能仍需独立审批。
`deny` 拒绝该操作；`abort` 取消 Turn。

## Linux 输出与构建恢复

Linux 的候选工作区支持工作区内相对符号链接，保留普通文件 mtime，按块复制并校验指纹。
普通文件总量没有 64 MiB 上限，仍需要足够空间存放候选副本。

`run_command` 和 `execute_python` 默认不发布失败产物。显式设置
`publish_on_failure=true` 时，失败或超时产生的文件通过安全检查后可以发布，
供下一次命令增量续建。工具仍报告失败，`workspace_published` 表示是否实际发布。
取消及沙箱启动失败不发布；此选项不增加写权限，也不取消破坏性操作审批。

续建是基于已发布的文件重新执行命令，不恢复已终止进程的内存。
活跃命令的查询、等待和取消见[命令会话](macos-command-sessions.md)。

## 工具批次、取消与结果恢复

同一模型响应的原生 `tool_calls` 经 `TurnExecutor` 进入
`ToolOrchestrator.execute_batch`：先记录全部调用，再按能力及状态边界分段。
`ToolExecutor` 预检当前段，按 `max_parallel_calls` 和已解析资源冲突分批执行。
独占工具、远程未知结果和改变运行时状态的操作保持执行屏障；运行时不根据工具名、
任务文字或错误内容猜测调用依赖。后续调用依赖前项结果时，应由模型下一步提出。

独立调用的参数错误、权限拒绝、runner 异常和已确认取消的本地超时各自形成结果，
不取消健康的兄弟调用。存储提交失败属于批次故障：停止接纳并排空已启动调用，
保留原操作供恢复核对；不能用一条临时错误消息冒充已持久化的执行结果。
非幂等远程超时、进程死亡或未知副作用继续进入逐项对账，不能自动重跑。

取消后，`FINISH_CURRENT` 完成当前调用并提交结果，再传播取消，
不启动排队工具或下一模型请求。重复取消仍排空同一完成或清理任务，
不会仅因再次收到取消就把活跃 runner 记为已停止。普通执行和审批恢复共享这套接纳语义。
claim 事件投递期间、runner 尚未进入时的取消会提交确定的取消结果；runner 已完成后的
输出快照与终态记录会排空，保留已知结果，可选遥测不会阻止取消。
`agent recover TURN_ID --action continue` / `retry` 仅在明确取消后的暂停、无待处理交互和
无运行中或未知操作时继续；恢复跳过已有结果，只执行原来未开始的调用。
`abort` 会为未执行调用闭合工具消息组。审批、澄清和对账仍走各自原有流程。

工作区写的前快照在真正获准调用 runner 后拍摄。由于当前证据同时观察整个工作区树，
这些写操作互斥，避免将兄弟调用的修改归给 no-op 写；独立读保持有界并发。
该互斥限于当前批次，不提供对其他 Thread、宿主进程写入或活跃后台命令的全局事务隔离。

新 Turn 冻结 `execution_context_policy.tool_result_order=call-v1`，响应记录保存该标记。
审计账本保留实际完成顺序，provider 上下文的完整工具组按原调用顺序排列，
结果通过 `tool_call_id` 关联。旧响应缺少标记时保持原完成顺序；重开不会补默认值或
改变旧请求布局。工具组未闭合时不能发送下一模型请求。

该机制参考了 Apache-2.0 的 Codex commit `ddf04ad26789d040f9ef6a96736f76602e35a6cc`：
[能力驱动的并行接纳及调用错误边界](https://github.com/openai/codex/blob/ddf04ad26789d040f9ef6a96736f76602e35a6cc/codex-rs/core/src/tools/parallel.rs#L88)、
[顺序排空并发结果](https://github.com/openai/codex/blob/ddf04ad26789d040f9ef6a96736f76602e35a6cc/codex-rs/core/src/session/turn.rs#L2213)，
以及 [Claude 的完整并行工具结果协议](https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use)。
Python 实现复用本项目执行和持久化组件，没有复制外部源码；Claude 本地参考副本仅用于比较，
不作为已核实私有核心或可复制授权的依据。
