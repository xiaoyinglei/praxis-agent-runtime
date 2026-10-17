# 同一 Turn 的恢复与权限边界

`Agent.resume(turn_id, action)` 和 `agent recover TURN_ID --action ACTION`
恢复同一个持久化 Turn。`previous_turn_id` 则为新 Turn 提供历史上下文；
`agent resume THREAD_ID` 重新打开聊天，下一条消息也会创建新 Turn。

新 Turn 对已知 `incomplete/max_output_tokens` 允许一次完整步骤重试，额度按整个 Turn 冻结，
原步数、tokens 和费用预算照常扣除。截断的正文、reasoning 和工具调用不作为后续 assistant 消息重放，
已完成工具结果保留；content_filter、未知终态和 summary 不走自动重试。重开进程不会重置额度，
缺少该策略字段的旧 Turn 保持旧契约。缺失 provider 终态仍进入 unknown 的人工恢复路径。
具体实现、CLI/SDK 和 Linux 构建恢复验证见[修复记录](runtime-output-and-linux-repair.md)。

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

## 成熟实现的源码对照

本次对照使用本机 `reference-src/codex` 与 `reference-src/claude_code-src` 源码快照，
不将其当作上游当前版本声明，也不假定其他产品的聊天恢复与本项目的同一 Turn 恢复完全相同。

- Codex `codex-rs/core/src/thread_manager.rs:1099` 的
  `resume_thread_with_history` 分别接收当前 `Config` 与 `InitialHistory`；
  `codex-rs/core/src/session/mod.rs:732` 从配置建立审批和权限状态，
  `session/mod.rs:1415` 单独重建历史与用量。可借鉴的是配置与历史职责分离。
- Claude Code `src/utils/sessionRestore.ts:95` 的
  `restoreSessionStateFromLog` 恢复文件历史、归因和任务状态；
  `src/utils/permissions/permissionSetup.ts:872` 从当前 CLI 设置建立工具权限。
  `src/cli/print.ts:5345` 明确要求 SDK 动态 MCP 与 CLI 配置入口在连接前使用同一企业策略过滤。
  可借鉴的是所有资源入口共享权限门，历史恢复不能替代当前权限检查。

本项目据此将修复放在共享 Session 的恢复准备阶段：先应用已有冻结字段，
再沿用现有资源打开与审批流程。没有新增另一套恢复执行器或根据任务文字定制的规则。

## 验证

```bash
.venv/bin/python -m pytest tests/agent/harness/test_frozen_mcp_recovery.py -q
```

测试覆盖 SDK 与真实 Typer `recover` 入口：原 Turn 关闭 MCP，暂停后出现未经信任的工作区
配置，新进程使用默认设置仍可批准原操作并完成恢复。发现调用计数还覆盖当前禁用、
历史禁止、当前禁止与正常启用的对照；另验证当前新增的工具禁止使旧审批失效且不写文件。
