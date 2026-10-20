# macOS / Linux 长命令与崩溃重连

2026-10-17。macOS 使用 Seatbelt；Linux 使用 bubblewrap 和候选工作区发布。公共工具参数相同，文件语义按 OS 明确区分。Windows 没有新增执行后端。

## 使用

模型调用 `run_command`：

```json
{"command":"make all","working_dir":".","workspace_write":true,"yield_time_seconds":1,"timeout_seconds":null}
```

审批及路径、网络边界照常生效。`yield_time_seconds` 是响应等待时间，最多 60 秒；仍在执行时返回
`status=running`、`exit_code=null`、`session_id=cmd_...`，并非验收成功。后续观察同一命令：

```json
{"session_id":"cmd_...","action":"wait","wait_seconds":10}
```

`manage_command` 支持 status / wait / cancel / read。wait 期满不终止命令；cancel 等待进程组清理。
取消正在等待的调用也会取消命令。completed 表示已结束，成功还须 exit_code=0、未超时、无 sandbox_error。

超时参数严格以**秒**计，不再按数值猜测毫秒。省略时仍为 120 秒；同步调用最多 600 秒。
超过 600 秒的明确期限（例如 3600），或者 null 无执行期限，须配合 yield_time_seconds。
execute_python 为同步接口，仍最多 600 秒，超过会明确拒绝。

普通 stdout/stderr 保留 16 KB 累计头尾快照，避免塞满模型上下文。Agent Session 的完整流另有
每流 16 MiB 的磁盘预算；可通过字节偏移读取中间错误，超预算会显式报告 log_truncated：

```json
{"session_id":"cmd_...","action":"read","stream":"stderr","offset":16000,"max_bytes":16000}
```

日志接口每次最多 64000 字节，返回 log_next_offset / log_end_offset；偏移以字节而非字符计。

## CLI 与 SDK 恢复

父 Agent/CLI 被 SIGKILL 或解释器崩溃后，独立 worker 继续持有原进程与日志。
重新 `agent recover <turn_id> --action ...` 或 `Agent.resume()` 时，仅认领该 Turn 账本中已提交的原任务。
不重跑命令，不将 worker 丢失伪装成成功。正常退出、close、abort 或 Turn 预算失败仍会取消命令。

无需模型即可检查或取消崩溃前的任务，在原工作区执行：

```sh
agent commands <turn_id> <session_id> --action status --rollout-db /absolute/rollout.db
agent commands <turn_id> <session_id> --action wait --wait-seconds 10 --rollout-db /absolute/rollout.db
agent commands <turn_id> <session_id> --action read --stream stderr --offset 16000 --rollout-db /absolute/rollout.db
agent commands <turn_id> <session_id> --action cancel --rollout-db /absolute/rollout.db
```

Python SDK：`await agent.command(turn_id, session_id, action="wait", wait_seconds=10)`。
此检查入口释放客户端所有权时不取消仍在执行的任务；它与正常 Session.close 的契约不同。
同一任务不能同时由两个活 Agent 控制。保留原 session_id，不接受任意 PID 或替换命令。

## 崩溃一致性与权限

- 先准备私有 job，再将 launch ToolResult 提交到 Rollout，最后通过通用 after_commit 钩子 release。
  提交前不执行；提交后、release 前崩溃，恢复只 release 同一个准备任务。
- 独立稳定 owner.lock / run.lock，close-on-exec，不替换或删除锁文件。started 标记防止 worker 重放；
  原子持久化 result 后才释放 run.lock；有 started、无结果且锁空闲时报告 unknown。
- journal 位于工作区外的每用户临时目录，目录 0700、文件 0600；拒绝符号链接和硬链接，JSON 原子替换。
  文件工具权限及两个 OS 沙箱显式隐藏 journal，即使更宽目录有读取授权。
- worker 在命令执行前固化运行库 Python 源码到私有目录，从固定解释器以 -I 启动；工作区 cwd、
  PYTHONPATH、同名影子包，以及 editable install 后续被命令修改的源码均不能替换该 worker。
- 恢复核对工作区路径与 dev/ino、请求 digest、执行 revision、原 operation 和私有 release 关联。
  文件变更证据从可信账本恢复原授权与基线，不采用模型提供的元数据。
- 当前权限与原授权不一致时保守取消，并阻止旧输出读取。Linux worker 发布前等待当前 owner 的新 nonce 授权；
  owner 崩溃时可继续计算，但不能独自根据旧权限发布。已启动 OS 沙箱不能动态缩权；撤销通过取消落实。

## Linux 与 macOS 文件语义

Linux 保留候选副本、保护目录、路径/硬链接/特殊文件检查及宿主并发写入冲突检查。
移除了普通文件总量 64 MiB 上限：以 1 MiB 块复制与 SHA256 指纹重新校验，内存不随普通文件字节总量增长。
仍需候选副本的磁盘空间，目录元数据内存随文件数增长；不是零复制或无限资源。
扫描/复制在可取消的工作线程执行，避免阻塞事件循环，清理前等待线程停止使用 scratch。

默认只发布成功命令；publish_on_failure=true 可发布安全检查通过的非零/超时部分产物。
取消丢弃尚未发布的候选，沙箱启动失败不发布。多文件提交仍不是原子事务；提交期间 I/O 故障可能留下部分变更。
发布前的权限复查不能回滚已经提交的文件。

macOS 为授权目录直接写入；失败、超时、取消后的文件保留，publish_on_failure 无效，workspace_published=null。
两者均不提供 OS 重启后的进程内存恢复；worker 自身死亡只报告 unknown。
不保证清理主动 setsid/double-fork 脱离进程组的 daemon，不是多租户容器。

保留每 Session 一个 shell 活任务、最多 32 个内存结果；已收集结果可淘汰，未收集结果不会静默丢弃。
这些是并发与资源边界。持久日志每流有预算；历史 job/代码快照目前没有自动 GC，需要宿主按保留策略清理，
不得删除仍有 owner/worker 持锁的 job 或它引用的代码。当前不提供 stdin/PTY、远端服务或 cgroup 调度。
独立 CommandSessions 为进程内 owner；公开 Agent 使用 DurableCommandSessions。直接 Tool.run 绕过账本时不会 release 持久任务。

## 依据与验证

实际阅读固定版本官方源码，以 Python 基本组件独立实现，未复制上游代码：

- [Codex unified_exec](https://github.com/openai/codex/tree/ce0323f2f2d3b25e82806cd555237da3db4e1897/codex-rs/core/src/unified_exec)：响应等待与执行期限分离、保留原进程、确认终态、有界输出。
- [tmux server_start](https://github.com/tmux/tmux/blob/2e4a4db4e9fe6b976a0a956af88e427fcbbe44ba/server.c)：server 与 client 生命周期分离。
- [Supervisor process](https://github.com/Supervisor/supervisor/blob/abc60468ea4b78c446cf3a194f6fccb83d90f670/supervisor/process.py)：持有子进程、进程组及停止状态。
- [Codex reconnect](https://github.com/openai/codex/blob/ce0323f2f2d3b25e82806cd555237da3db4e1897/codex-rs/exec-server/src/remote.rs)：重连保持执行器身份并核对授权；本实现不引入其远端协议。
- [bubblewrap](https://github.com/containers/bubblewrap/tree/1fb0aa26d3eb66254f7742e704add7f82f41a582)：保持 OS namespace、网络过滤和 die-with-parent 边界，父进程现在是独立 worker。

主要实现：tools/builtins/{shell,command_sessions,command_worker}.py，tools/{tool,executor}.py，
harness/{session,tool_orchestrator,command_recovery,rollout,completion}.py，command_cli.py 和 Agent.command。

测试分开真实 Seatbelt、原生 Linux bubblewrap 与 fake 流程；使用短工作负载、真实父进程 SIGKILL、
脚本模型驱动公开 CLI/SDK，检查重复执行、审批、账本完整性、权限撤销、并发认领、日志中段与大工作区发布。
这些不是付费真实模型大型构建验收；未改或重跑冻结 benchmark，未提交/推送/合并。

最终验证记录：Agent/CLI 主回归 2870 passed、76 skipped、17 warnings（既有 SWIG/pty 弃用警告），
日志 `/tmp/praxis-linux-recovery-final-regression.log`。随后句柄锁释放与权限源异常收尾由 macOS 聚焦
18 passed、3 Linux-only skipped 验证，日志 `/tmp/praxis-recovery-policy-final.log`；
最终原生 Linux 集合 106 passed，日志 `/tmp/praxis-linux-native-final.log`，含真实 bubblewrap、
父进程 SIGKILL、公开 CLI/SDK、65 MiB 发布、撤权及撤权源不可读。
当前代码 Ruff、mypy（123 sources）、git diff --check 均通过，import-linter 5 contracts kept / 0 broken。
Linux 测试外层容器无网络、UID 1000、cap-drop ALL；仅放开外层 seccomp/systempaths 以运行嵌套
用户命名空间，内层产品 bubblewrap/网络 seccomp/文件策略保持执行，源码只读挂载。
