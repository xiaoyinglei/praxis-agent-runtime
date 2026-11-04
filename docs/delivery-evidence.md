# 版本化交付证据

Turn 的 `done` 表示执行结束。`AgentResult.delivery.status`（映射访问为 `result.delivery["status"]`）独立表示交付证据：

- `verified`：声明范围有当前文件版本、环境与依赖边界内的证据；不自动证明任意语义正确。
- `unverified`：结果可检查，但缺少计划、证据、最终审查，或者已有证据过期。
- `blocked`：检查失败、环境/读取权限不可用、证据损坏、子产物未处理或执行结果 unknown；`next_actions` 给出续跑动作。

默认 CLI 和终端显示状态、已验证范围、证据身份和续跑动作。后台 `status/result` 返回同样字段。读取证据不启动模型或宿主：

```sh
uv run agent delivery TURN --workspace /absolute/workspace --rollout-db /absolute/runtime.sqlite
```

Python 可用 `await agent.read_result(turn_id)` 重新计算当前新鲜度。在打开的 Session 中，`await session.delivery.inspect(turn_id)` 提供同一完整事实；`plan/record` 只允许写该 Session 的运行中 Turn。完成后要对新版本重新验证，使用正常后续 Turn（`previous_turn_id`），不改写旧记录。

## 模型和 SDK 的同一条链

1. `record_delivery(action="plan", scope=[...], commands=[精确命令], omissions=[省略理由])` 声明范围及验证计划。SDK 对应 `await session.delivery.plan(turn_id, scope=..., commands=..., omissions=...)`。
2. 使用原来的 `run_command` 执行检查，添加 `verification_mode` 与 `verification_dependencies`。
3. `record_delivery(action="record", tool_call_ids=[...], review="最终审查结论与覆盖范围")`，SDK 对应 `await session.delivery.record(turn_id, ...)`。指定旧成功不能隐藏相同精确命令的后续失败。
4. 自然 final 结束执行。未记录交付声明时，宿主仍保存最终版本与 diff，诚实显示未验证。重新读取时核对文件、计划、需求版本、依赖、未决执行和所引用 artifacts；旧证据保留，不洗成成功。

`record_delivery` 的工具文档是 ACI 合同，不授予权限。没有显式命令时，主代理依据仓库和需求自行制定计划；没有检查不等于通过。检查失败可修复后重跑，最新成功且版本一致后恢复证明。

## 命令的版本边界

`verification_mode="snapshot"` 从 D2 同一快照机制复制 tracked 与普通 untracked 文件到隔离目录，原 index/HEAD 不变；普通目录复用外部对象库，不注入 `.git`。命令使用原进程执行器与 OS 权限边界，网络关闭、输入只读、临时输出目录可写，绝不把验证目录改动发布回来源。父目录并发编辑不改变被测输入；若交付时父版本已不同，旧证明失效。

`verification_mode="workspace"` 在原 cwd 运行同一精确命令。它记录内容、输入及祖先目录的 inode/mtime/ctime、HEAD 和 index 版本；验证期间内容改后恢复或临时创建后删除也会保守失效。显式 `acceptance_command` 在交付证明中要求此模式、工作区根目录、成功终态和当前输入证据。原有停止条件继续要求精确命令及当前工作区结果；后台/写入命令可以正常结束，缺少版本证明时交付保持未验证。生成缓存改变输入目录身份时可能需要再跑一次；不能证明稳定时不能显示通过。

这两个证明模式目前限有界的前台、只读输入命令。可写构建先在普通命令/独立写环境运行，再验证最终结果。长进程仍使用既有 `run_command/manage_command` 能力，未提供稳定版本证明时保留为未验证；不把进程观察成功当成交付通过。

`verification_dependencies` 是明确的外部依赖范围：省略时不能提供 verified；`[]` 声明只依赖系统工具及冻结源码；非空列表声明绝对读取路径。声明不扩大权限，依赖按原始/解析路径检查读取权限。记录系统版本、实际 shell/所选可执行文件指纹、精确环境变量，以及显式依赖的内容和变更身份；运行中或之后依赖变化使结果失效。任意 shell 程序的完整依赖闭包仍由声明者审查，宿主不猜测导入关系。忽略文件、`.venv`、`node_modules` 不因快照创建自动复制；缺依赖会保留真实失败，不从父目录借入未声明输入。

收据保存精确命令、来源/实际 cwd、起止时间、退出码和原始 ToolResult 日志引用。命令输出遵守既有大小上限，截断标志保留；大工具输出沿既有 artifact 机制保留。文件快照上限沿用 D2 的 20,000 文件/256 MiB；超限、特殊文件或不稳定来源明确报错。

## 来源、diff 与解释任务

开始执行前保留来源快照 artifact；最终记录包含目标输入版本、计划、最终清单/归档、完整二进制 patch、子任务来源/集成记录及检查引用。基线到最终的 diff 表示本 Turn 期间的变化，用户原有 staged/unstaged/untracked 内容保留在基线及原 index 记录内，不冒称由代理创作。共享目录中外部编辑与代理编辑仍可能混合，不能根据时间差自动认领每个 hunk；GitDelivery 的选择/预览/提交边界保持独立。

解释任务可以没有命令、文件修改和测试。提供可信来源工具的 Item ID（或 call ID）、省略检查的理由、回答覆盖审查，并将确切最终回答传给 `answer`；记录绑定回答摘要。最终回答不一致、需求转向或来源工作区版本变化会使旧证明失效。宿主核对来源身份与版本，不用词语覆盖率判断答案是否正确。

所有记录使用 canonical Items/artifacts，与 Thread/Turn 共用事务及重建路径，没有第二套 Task/Run 状态机。证据缺失或损坏会阻断该交付记录。E 的本机脚本模型测试不代表真实云模型质量验收；渠道、CI/PR 后续和发布验收仍属于 F/G。

完整 Agent 产品入口默认提供模型侧交付工具；低层自组装 `Session.open` 保留原精简工具集合，可显式传 `enable_delivery_tools=True`。SDK 的 `session.delivery` 始终可用；生命周期与交付事实不因工具是否常驻而混同。
