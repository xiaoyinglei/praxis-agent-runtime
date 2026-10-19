# 输出截断与 Linux 构建修复（2026-10-17）

范围：沿用交接文档和当前未提交工作树；不运行公开基准或付费模型，不提交或推送。
历史 wheel 和官方成绩保持原样。实现与无模型证据另存。

| 证据 | 当前机制 / 状态 | 成熟来源 | 最小改动 | 验证 |
| --- | --- | --- | --- | --- |
| MIPS 最后请求 output=reasoning=32768，无正文、无调用 | `turn._dispatch_prepared` 无条件失败，仍未修 | OpenAI Reasoning 文档确认可在无正文时到输出上限；并未保证 Chat 协议可继续原推理 | 新 Turn 冻结一次整步重试策略；已完成历史不变；截断步骤永不作为工具调用或正文重放；总预算和步数照常扣除 | 空正文、部分工具参数、重复截断、非输出类截断、预算不足、取消、重开数据库、CLI/SDK |
| 部分工具参数和 `length` 同时出现 | 非流式解析 JSON 早于 Runtime；流式 stop 归一化优先 tool calls，可能吞掉 MAX_TOKENS | 协议终态应优先于工具片段；本项目设计，不冒称其他产品保证 | 只有完整终态才解码/暴露工具调用 | 流式与非流式真实序列化路径 |
| OCaml configure 创建相对符号链接被拒 | Linux 候选视图禁止所有链接变更，仍未修 | Codex approvals/security 分开 OS 边界与审批；Claude checkpointing 不跟踪 Bash 改动 | 只允许工作区内相对链接，校验完整目标链及保护目录；沿用 descriptor 相对发布 | 内链、外链、绝对链接、悬空链接、父路径、保护目录、硬链接和并发替换 |
| make 超时丢失候选增量产物 | 每次复制，失败/超时丢弃，mtime 未保留 | 不把任意 shell 回滚视为 OS 隔离；选择明确发布策略 | `publish_on_failure=false` 保持默认；显式开启时，非零退出及超时后仍须完整安全校验才发布；取消丢弃；保留普通文件 mtime | 实际 Linux configure/make 两段续建、超时、失败、取消、越界；无后台进程存活或重连承诺 |
| 前窗口预算、completion-v2、MCP、指标 | 当前工作树已有修复 | 前窗口审计文档及源码 | 保留 | 相关已有回归 |

官方来源（本轮重新读取）：

- https://developers.openai.com/api/docs/guides/reasoning （输出额度包含 reasoning；可能无可见输出）
- https://learn.chatgpt.com/docs/agent-approvals-security （审批与沙箱职责分开）
- https://code.claude.com/docs/en/checkpointing （Bash 文件变更不由 rewind 跟踪）

历史参考源码快照不是当前上游声明；非官方 Claude 镜像不作为官方事实。
本轮不依赖 Responses 的 previous_response_id、推理缓存或 assistant prefill。
整步重试只适用于本地工具执行的当前 provider 协议：截断请求没有执行远端工具的能力。

实施步骤（原任务已授权直接实施）：

- [x] 先写并运行失败用例：输出截断、provider 半截参数、Linux 内链与文件时间戳。
- [x] 修复 provider 终态优先级，新 Turn 冻结重试策略，恢复与上下文共享同一处理路径。
- [x] 修复相对链接发布；增加显式失败发布选项，沿用所有权限与路径检查。
- [x] 验证真实 Linux 沙箱和公开 CLI/SDK；回归前窗口契约。
- [x] 记录命令、环境、结果、源码差异与限制。

后续修复（2026-10-17）：同步命令仍最多 600 秒，后台命令可显式设置更长秒数或无期限。
Linux 普通文件总量 64 MiB 上限已移除，改为流式复制与指纹验证，仍需要副本磁盘空间。
父 Agent 崩溃可接回原命令；worker 死亡或 OS 重启不恢复进程内存。详见[长命令与崩溃重连](macos-command-sessions.md)。
开启失败发布后也可从已发布产物重新执行增量构建；不重建已经终止进程的内存或描述符。
多文件发布不是 filesystem transaction，要求没有并发宿主写入；保留路径/身份检查，
不能宣称并发 I/O 失败时全树原子回滚。

## 实际契约与实现位置

`SessionBindingProvider` 为新 Turn 冻结 `model_response_recovery`：
`{"revision":"restart-step-v1","max_retries":1}`。额度按整个 Turn 计算，
不随成功步骤、压缩或重开进程重置。`TurnExecutor._handle_model_response` 从已提交的
`model_response` 计数；只有 `max_output_tokens` 可以从已完成历史发起一次新步骤。
重试经过普通 preparation、预算 admission、dispatch 和计费，不扩大单次输出上限，
不替换模型、不提高预算。没有该冻结字段的历史 Turn 保持旧行为；summary 不自动重试。

截断响应保存在账本用于诊断，但 `context._item_message` 只投影明确的 Runtime 反馈，
不投影部分正文、reasoning 或没有结果的调用组。成功工具记录继续正常投影。
`Session.resume` 和 `recover_committed_model_response` 处理“响应已提交但尚未重试”窗口；
已有 prepared-request 恢复负责“重试已准备但未派发”窗口。
流式 agent message 的完成事件标为 `failed`，携带 incomplete 原因；重试有独立 Item，
SDK 最终答案只使用完整响应。重试成功后保留 warning 诊断，不把原截断隐藏。

`chat`、`openai_wire`、`gateway` 和 `model_adapter` 保留 provider 终态：
输出截断优先于看似完整的工具 JSON；content_filter / 未知终态不暴露调用、不自动重试。
缺少终态视为连接结果不确定，进入原有 unknown / 人工恢复路径，不伪造成功。
这项修复来自本轮复核，不能用直接注入 HarnessResponse 的测试代替 provider 协议测试。

Linux `run_command` 与 `execute_python` 共用以下发布语义：

```json
{
  "command": "make all",
  "workspace_write": true,
  "publish_on_failure": true,
  "timeout_seconds": 600
}
```

这是 `run_command` 的工具参数。Python 工具使用相同发布字段，另外提供 code / output_paths。
`publish_on_failure` 仅影响 Linux 且仅在有写候选视图时生效；它不授予写权限、执行权限或网络权限。
`workspace_write` 的原有 destructive 审批保持生效。默认仍只发布成功命令。
显式开启时，命令非零退出/超时仍返回工具错误；`workspace_published=true/false` 明确说明
候选是否发布，非候选后端/只读执行为 null。取消、沙箱启动失败和校验失败不发布候选。
超时先终止进程并收完输出，再验证发布；后续命令从文件状态重新构建，不能重连旧进程。

链接验证在候选快照中逐组件解析，最多跟随 40 次；允许同批创建的目标、内部链和内部悬空链接。
新增/重定向的绝对链接（含绝对内部链接）、外链、循环、保护目录和不可读目标都拒绝。
父目录描述符带 `O_NOFOLLOW` 与 inode 身份检查；leaf 链接通过相对 descriptor 的
symlink + replace/link 发布，不沿链接写入目标。文件内容、mode、mtime 和 inode
都参与宿主并发修改检查；mtime-only 的候选修改也会发布。
新 schema 与执行语义对应 command v10 / Python v6 execution revision，旧待执行操作不能
在未核对版本的情况下静默升级。现有无关外链可保持原样，但修改它们仍可能因原路径权限拒绝。

## 验证和证据位置

新增产物在 `evals/public_benchmarks/artifacts/runtime-output-linux-20261017/`。
`before.log` 是修复前 7 failed / 7 passed 的复现；`provider-red-reproduced.log` 使用
HEAD 的三个 provider 文件在独立临时目录运行最终断言，4 项均因截断解析/终态误判失败。
它不会回滚或覆盖当前工作树。较早的 `provider-before.log` 含两项测试 usage fixture 错误，
不能将那两项错误当作源码失败证据。

Linux 初次 root + cap-drop ALL 不能建立 uid map；调整为 UID 1000 后，实际文件/进程检查通过。
amd64/Rosetta 下显式联网的嵌套 seccomp 报 EINVAL，保留 `linux-unprivileged.log`，
没有关闭过滤器或跳过原断言来宣称通过。最终另用原生 ARM Linux 镜像运行。
容器始终禁外网、只读挂载源码、非 root、cap-drop ALL；外层 Docker 仅放行所需 userns
syscalls 并解除重复系统路径遮罩，实际命令继续由 Bubblewrap 和其网络过滤器约束。

原生 Linux 的测试文件直接导入 `/src` 当前源码；镜像内预装包仅提供依赖。
`test_llm_gateway.py` 的旧测试依赖可选 RAG 导入，Linux core 镜像缺少 sentence_transformers，
因此该文件在宿主验证；原生 Linux 验证直接覆盖真实 Gateway + adapter + Runtime 流式路径，
不安装无关的训练/嵌入依赖。所有初始化失败日志保留，最终验证日志单独记录。

这轮只有无模型验证，不能声称 MIPS/OCaml 官方任务已在新代码上通过。
原冻结 wheel SHA256 已重新核对，仍为交接文档中的 `0cf68900...`。

最终验证：

| 检查 | 实际结果 | 日志 |
| --- | --- | --- |
| `tests/agent tests/provider tests/ui/test_cli.py`，真实宿主 Seatbelt | 2905 passed，75 skipped，19 warnings；无失败 | `regression-final.log` |
| 原生 ARM Linux，实际 Bubblewrap，CLI / SDK 和 provider wire | 131 passed；无跳过/失败 | `linux-native-verified.log` |
| 最终聚焦截断 / 构建 / provider 回归，macOS | 90 passed，17 个 Linux 专用用例跳过 | `focused-final.log` |
| Ruff 变更 Python 文件；mypy Runtime 115 个源文件 | 通过 | `ruff.log`、`mypy.log` |

`linux-environment.json` 确认 UID 1000、aarch64、所有 capabilities 为零，导入来源为 `/src`。
`source-manifest.json`、`source/` 和 `full-worktree.diff` 保存当前可复验状态；
全树 diff 包含前窗口与用户已有未提交变更，不能全部归为本轮修改。
本轮聚焦复核发现的非成功终态工具暴露问题已修复，最终增量复核未发现新的重要问题。
