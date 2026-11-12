# Praxis 执行链修复与验收

基线为 `main` 的 `313818817bfe1710abd54ce65877d3d1d72eef8b` 加上一批未提交改动。隔离检出为 `/private/tmp/praxis-runtime-audit.Yt4GzL`。当前修复由隔离检出相对该基线生成，原始 Rollout 使用 SQLite 只读方式取证。

## 变更证据

旧实现对每个 `WRITE_WORKSPACE` 调用采集写前和写后整树 SHA-256。目标文件的局部哈希已经记录，但 `StopChecks` 只接受整树布尔值。任一全树读取失败会使这个布尔值变为 `False`。公开 SDK 红测中，无关文件 `chmod 000`，补丁实际把 `before\n` 改为 `after\n`，Turn 最终仍为失败。普通工作区对照能够完成。

原生 `apply_patch` 现在用原子安装实际消费的前后字节生成回执。现有目录描述符、工作区写锁、预期源版本和创建冲突检查继续约束安装。适配器将回执作为 runner 的内部返回值交给 `ToolExecutor`，执行器清除普通输出里可伪造的证据字段，写入规范工具结果。完成门继续要求 Rollout 中有匹配的 ToolOperation、工具结果、写权限资源及因果顺序。重复失败判定和步数回执也读取同一证据。历史结果仍可使用整树布尔值。

任意可写命令保留整树观察，观察失败用 `null` 表示。显式拒绝覆盖指纹范围时跳过扫描，目录遍历错误也会让指纹不可用。已接纳的命令在写前扫描后仍进行权限重检。持久后台命令在启动回执提交后才放行 worker，启动及运行中查询复用持久化的写前基线。最终状态才做一次写后观察。恢复使用同一操作、命令摘要及 worker 身份，权限撤销后隐藏历史输出并跳过观察。任意命令的指纹差异仍表示观察区间内的工作区变化，无法区分同一时段内外部进程的写入。

完成门与交付归档继续采用各自的证据。延迟交付基线在关闭交付和基线已存在时也满足单外层事务契约。全量交付需要能归档和重新核对完整版本，遇到不可读文件仍可报告阻塞。检查点和工作区验收的整树版本检查仍有独立用途。可选 Codex 二进制适配器和没有精确提交回执的自定义写入仍走观察路径。

## 子进程、工具可见性和容量

- 前台 Shell 与托管 Python 显式使用 `DEVNULL`，与持久 worker 的非交互契约一致。保留后台查询、等待、取消、日志和按句柄恢复。需要命令输入的用例仍可由显式管道或重定向提供数据。Praxis 目前没有独立 PTY 输入会话。
- 保留 `DurableToolRouter` 的 `find_tools` 和已提交激活事实，保留执行时的权限裁决及每个 Step 冻结的工具集合。删去 `run_command` 工具描述中重复的参数说明，保留参数约束和平台差异。同模型、同一组 16 个工具的真实冻结请求中，`run_command` 的紧凑 JSON 从 5,718 字节降到 4,990 字节，工具数组从 41,484 降到 40,783 字节。原始首轮的 18 个工具共 39,170 字节，配置与本次试验不同。
- 第一次服务端容量拒绝仍走原有压缩，发送前要求实际序列化请求比被拒请求更小。第二次拒绝后，使用已有的工具结果归档投影并测量实际序列化输入。只有工具定义相同、输入字节下降且可用的 token 估计也下降，才在原恢复链中创建一个新的请求身份，最多再发一次。调用与结果配对和原始证据保留。归档提交后进程中断可恢复，原输入不可再缩减时结束。
- OpenAI Chat Completions 的标准生成文本是字符串或空值。对于兼容供应商的带类型文本分片，完整响应、SSE 和非流式回退统一拼接文本。未知分片明确报协议错误。拒绝通道、结束原因和工具调用仍由各自的协议字段提供。非流式预算在收到格式错误的响应时按已报告用量结算，缺少用量时保守计入预留量。[OpenAI Chat Completions 响应契约](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
- `workspace` 默认限于工作区。以合成文件验证 `trusted-local` 的根目录读授权、显式拒绝及管理员上限。原始两轮使用 `workspace`。权限配置语义维持原契约。

## 源码对照

- Codex 本地切片 `a4c9194247c6a1e79aa3e72cc6cd77238f2ed354` 提供 `session/turn.rs`、压缩、命令入口和 `write_stdin`。补丁实现和事件缺段用上游 `main` 对照。验证后的 patch 进入沙箱编排，runner 提交的 `AppliedPatchDelta` 进入完成事件和 `TurnDiffTracker`。参数流中的 patch 预览由另一条事件路径发出。任意命令走带操作身份的进程生命周期，PTY 和输入经独立请求和权限检查。当前上游还从注册表的 deferred 工具构建发现索引。Praxis 复用了提交事实与投影分离的设计，文件安装和 Rollout 仍使用自身实现。[补丁入口](https://raw.githubusercontent.com/openai/codex/main/codex-rs/core/src/tools/handlers/apply_patch.rs) · [工具事件](https://raw.githubusercontent.com/openai/codex/main/codex-rs/core/src/tools/events.rs) · [Turn 差异](https://raw.githubusercontent.com/openai/codex/main/codex-rs/core/src/turn_diff_tracker.rs) · [进程与输入](https://github.com/openai/codex/blob/main/codex-rs/core/src/unified_exec/process_manager.rs) · [工具发现](https://raw.githubusercontent.com/openai/codex/main/codex-rs/core/src/tools/handlers/tool_search.rs)
- Claude 使用本地历史反编译镜像 `276b8e6`，来源未独立核验。`toolOrchestration.ts` 转入 `toolExecution.ts` 的权限检查，再调用 `FileEditTool` 或 `FileWriteTool`。两者按目标记录文件历史、检查最新读版本、执行写入，再更新读状态。文件历史由会话存储按消息身份恢复。Bash 工具通过 `Shell.ts` 持有子进程 stdio 和任务输出，前后台管理单独处理。镜像中的 `services/api/claude.ts` 向支持的 API 标记 `defer_loading`，`query.ts` 的容量恢复在压缩前可尝试更廉价的投影。Praxis 保留现有 OpenAI 兼容工具发现和预算记录，复用已有的工具输出归档能力。
- Praxis 仍把审批、不可重放的执行身份、完整历史、验收和文件内容归档分别交给原有层。针对精确写入替换整树推断，对无法证明精确写集合的命令保持有界观察。没有引入新供应商依赖或第三方运行时代码。

## 验收记录

| 方式 | 场景 | 结果 |
| --- | --- | --- |
| 真实模型和公开 macOS CLI | 百炼 `deepseek-v4.1-flash-1017`，普通工作区补丁 | `turn_76d1e7b0d5fb43d090f3ad1e7b53d804` completed，CLI 退出 0，独立读取为 `after\n` |
| 真实模型和公开 macOS CLI | 同一补丁，无关文件不可读 | `turn_377b48b4654d44f6a681acbce23577f6` completed，CLI 退出 0，目标变更回执存在，完整交付因归档不可读文件而 blocked |
| 真实模型和公开 macOS CLI | 父 CLI 的输入管道中有合成文本，执行 Shell 和托管 Python | `turn_e4539c829efd4a1f89aad9aa178ac216` completed，两次子进程退出 0，分别输出 `shell-eof` 与 `python-eof` |
| 脚本模型与真实工具 | 两组补丁对照，后台命令首次及最终观察，审批撤销，回合恢复 | 通过公开 SDK、CLI 或执行器测试，取证于隔离 Rollout 和本地文件 |
| 受控供应商适配器和协议测试 | 第二次容量拒绝、可缩减与不可缩减、提交后重开、完整响应与流式回退 | 覆盖实际序列化与持久化。真实供应商的第二次容量拒绝和分片数组响应仍未观察到 |

真实运行记录在 `/private/tmp/praxis-audit-live-d8jr9yjn` 和 `/private/tmp/praxis-audit-io-nxb1x5w9`。审计交接中的用户原始长任务没有重跑。

专项回归 494 通过、7 跳过。全量 Agent 与 Provider 回归中途停止。运行到 1,873 项通过时暴露第一次容量恢复偶发请求增大，已加入发送前的实际字节检查。修正后容量、重开及会话专项回归 102 项通过。
