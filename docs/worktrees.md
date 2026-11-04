# 并行会话与 Git Worktree 隔离

两个终端窗口在同一个项目目录里各开一个会话时，两个会话操作的是同一份文件：A 窗口的半成品修改会被 B 窗口读到、覆盖，验收命令跑的是两边混合后的代码。执行锁只防止"同一个会话被两个进程同时执行"，不提供"不同会话之间的工作区隔离"。

Worktree 隔离让每个会话在自己的检出目录和分支里工作，互不可见；交付仍然走原有的 `/commit` → `/pull` 链路。

## 设计对标

| 维度 | OpenAI Codex（codex-rs/worktree） | Claude Code（--worktree） | Praxis 的选择 |
| --- | --- | --- | --- |
| 位置 | 仓库外 `~/.codex/worktrees/<bucket>/<仓库名>/` | 仓库内 `.claude/worktrees/<名字>/` | 仓库外 `~/.praxis/worktrees/<bucket>/<仓库名>/`（Codex 式） |
| 分支 | 分离 HEAD（无分支） | `worktree-<名字>` | `worktree/<名字>`（命名分支，Claude 式） |
| 基准 | 默认远程默认分支 | `worktree.baseRef`：fresh（默认）/ head | 当前 HEAD；可用参数指定（Claude 的 head 语义） |
| 会话绑定 | git 元数据里的 `codex-thread.json` | 会话记录绑定 worktree | Thread 记录 workspace（现有机制），`.praxis/worktree.json` 记录溯源 |
| 隔离执行 | ——（依赖其沙箱） | 四条规则拦截写主检出/`git -C` 逃逸 | 复用现有工作区权限域：Session 的 workspace 即 worktree，工具天然被限制在其中，不新增逐工具规则 |
| 未提交内容 | 基于提交创建 | fresh/head 都基于提交 | 基于 HEAD 提交创建；创建时报告源工作区未提交数量 |
| 忽略文件 | 禁用 clean/smudge 过滤器 | `.worktreeinclude` 复制被忽略文件 | `.worktreeinclude`（Claude 式），仅复制"匹配且确实被 gitignore"的文件 |
| 清理 | 自动保留最近 15 个 | 退出时干净自动删、有工作先询问；定期清扫 | `agent worktree prune` 保留最近 15 个干净项；脏的或有未合并提交的永不自动删（Codex 式上限 + Claude 式保护） |
| 交付 | —— | 分支推 PR | `/commit` 在 worktree 内照常工作：`praxis/*` 分支继承 worktree 分支记录的真实基准，`/pull` 目标不变 |

选仓库外（Codex 式）而不是仓库内（Claude 式）的原因：主检出里的另一个会话执行文件搜索时不应看到并列的完整检出目录；仓库外还免去对 `.gitignore` 的额外要求。命名分支（Claude 式）而不是分离 HEAD（Codex 式）的原因：用户可以用普通 git 命令检视、合并这些分支，而且与 `praxis/*` 交付分支的自然堆叠一致。

## 使用

在项目目录里：

```sh
# 新窗口开一个并行会话（自动生成分支名）
agent chat --worktree

# 指定名字：分支为 worktree/fix-login
agent chat --worktree-name fix-login

# 非交互任务同样支持
agent run "修复 X" --worktree
```

创建时输出 worktree 目录、分支与基准提交。会话结束后回到该目录可以继续：

```sh
cd ~/.praxis/worktrees/<bucket>/<仓库名>
agent chat --continue
```

管理：

```sh
agent worktree list                 # 名称、分支、脏文件数、领先提交数
agent worktree remove fix-login     # 干净的直接删；有未提交/未合并提交时拒绝
agent worktree remove fix-login --force   # 确认丢弃后强制删除
agent worktree prune                # 清理最旧的干净 worktree（保留最近 15 个）
```

环境变量 `PRAXIS_WORKTREE_ROOT` 可改 worktree 根目录（CLI 各子命令也接受 `--root`）。

## 行为细节

- **单一拦截点**：`--worktree` 只在会话创建前把 workspace 指向 worktree 检出。Session、工具权限域、checkpoint、执行锁全部照旧工作在这个路径上；没有任何逐工具的隔离特例。
- **自包含状态**：worktree 会话的 Rollout 数据库放在 `<worktree>/.praxis/checkpoints.sqlite`，主工作区的会话列表不受影响；在 worktree 目录里 `--continue` 即可恢复。
- **未提交变更不进入 worktree**：基准是 HEAD 提交。创建输出会提示源工作区未提交文件数量。需要 `.env` 等被忽略的文件时，在仓库根放 `.worktreeinclude`（gitignore 语法，一行一个模式），创建时复制匹配且被忽略的文件（上限 200 个文件 / 64 MiB）。
- **交付不变**：在 worktree 里 `/commit` 照常创建 `praxis/*` 分支并提交；该分支继承 worktree 分支记录的真实上游，`/pull` 的 base 不变。交付后 worktree 分支仍在基准上，`agent worktree remove` 可以直接清理。
- **删除保护**：打开的会话持有工作区共享锁，删除必须取得排他锁，因此会话空闲等待输入时也受保护；即使 `--force` 也要先关闭会话。未结束的任务、未提交变更（包括被忽略的用户文件）或 worktree 分支上的未合并提交需要显式 `--force` 才能放弃；`.praxis/` 运行时状态不计入脏文件数。无法判定状态时同样按受保护处理，`prune` 不会强制删除。
- **仓库范围**：`list`、`remove`、`prune` 按 Git common directory 识别当前仓库；共享 worktree 根目录中的其它仓库不受影响。
- **Git 读取**：沙箱验证 linked worktree 的双向元数据指针后，只读开放对应 Git 元数据，使 `git status`、`git diff`、`git log` 正常工作；源检出的普通文件和 Git 元数据写入仍受原有权限边界限制。
- **防嵌套**：在 worktree 里再执行 `--worktree` 会拒绝，避免 worktree 套 worktree。

## 限制

- worktree 是全新检出：`node_modules`、`.venv` 等被忽略的依赖目录不会出现，需要时在 worktree 里重新安装，或用 `.worktreeinclude` 复制小体积必需文件。
- 子模块不会被递归检出；Git LFS 等仓库级过滤驱动器按 git 默认行为执行（Codex 会在复制时禁用过滤器，这里保持普通 `git worktree add` 语义）。
- 和主检出一样，worktree 依赖 `.praxis/` 在 `.gitignore` 中（本仓库已配置）。
- 主检出的未提交修改不自动带入；这是有意的：把"半成品"隔离在外正是本特性的目的。


## 子任务快照与主代理集成

上面的 `agent run/chat --worktree` 是交互会话的 HEAD 检出合同。附着子任务采用另一种来源合同，仍由同一个 WorktreeManager 分配目录、检查身份和保护回收：

1. `task` / `spawn_agent` 的 `workspace_mode=auto` 在存在文件写入或进程工具时启用写环境，`write` 显式请求写环境，`read-only` 收窄权限。新的附着子任务均从冻结来源执行；只读任务也不获得源目录写权限。
2. Git 来源保留 HEAD、原 index 字节/摘要和基于冻结 index 的 staged patch，并把实际 tracked/普通 untracked 文件原始字节冻结为基线。源码目录的 staged、unstaged、untracked、HEAD 和 index 不会被 stash/reset/clean 或刷新。Git checkout filter、export-ignore/export-subst 和 checkout hook 不改变快照。普通非 Git 目录使用仓库外的对象库，不向来源注入 `.git`。
3. 在启动模型/工具前，canonical 父 Turn Items 与子 Turn binding 记录 parent/child、source、base/tree、实际目录和来源 artifact。子 Thread 的 workspace、resident 文件工具、命令 cwd 和命令 owner 都绑定实际目录。权限仅收窄；祖先的持久撤销在重启后仍有效，权限文件在原来源目录域加载后再逐层映射，派发前会再次检查。低层 SDK 自组装工具集合保持原有能力；无法安全重绑定的自定义工作区工具或 skill runtime 明确失败，不借用源目录 runner、自动增加工具或静默删除能力。Agent 配置的产品入口会在子目录重新装配资源和 skills。
4. 父子仍共用 canonical runtime 数据库。重启沿 allocation、环境记录和产物状态恢复，不把内存 Task 表当执行事实。正常项目目录可以作为接入入口，子目录是执行现场。

快照边界是 Git 已跟踪及未被 ignore 的普通 untracked 文件。附着子任务不额外复制 `.worktreeinclude` 中的忽略文件；依赖、秘密或生成缓存需要在隔离环境中按权限准备。`.praxis`、当前 runtime 数据库及伴随目录、宿主配置和命令 journal 是明确排除的运行状态。每个快照上限 20,000 文件、256 MiB；子模块、特殊文件或不稳定来源会报错，不能静默截掉内容。子任务工作树根必须在源工作区外。

模型侧 ACI 与 SDK `session.subagents.integrate(...)` 使用同一组动作：

| 动作 | 合同 |
| --- | --- |
| `inspect` | 只接受已终止、无未知执行暴露的 owned child；保存实际 result archive/patch。按 `diff_offset` / `diff_limit` 读取，`diff_next_offset` 为下一页。必须完整读取 patch 后才能 apply。 |
| `apply` | 传入 inspect 返回的确切 `artifact_id`。按记录基线检查每个实际目标，包含 ignored 文件。冲突返回路径，双方内容不被覆盖。不会 stage 或 commit 来源文件。 |
| `resolve` | 主代理先手动解决冲突，再传相同 artifact 和 `resolution_note`；保存实际主目录 resolution diff，不自动猜测合并。 |
| `reconcile` | 集成中断后，显式核对每个路径是基线还是目标前后像。未决用户修改仍报冲突；核对后才允许继续 apply。 |
| `verify` | 在主工作区集成后运行实际 `run_command`（`verification_mode=workspace/snapshot` 提供稳定输入证明；旧写命令摘要仍作为兼容路径），必要时用 `manage_command` 等到终态，再提供该父命令的 `verification_tool_call_id`。命令必须成功且晚于集成；后续文件变化使记录失效。 |
| `discard` | 显式放弃集成，保留子目录和 artifact。它不清除未知工具结果，也不退还未知预算。 |

集成先持久写入意图，再修改文件。部分失败保留已写路径及前后像，状态为 `applying`，不会冒称整体回滚，也不会自动重放。删除、符号链接、目录/文件互换同样检查前像与实际文件版本；新建文件不覆盖同时出现的用户文件。

活跃或 unknown 子环境即使 `--force` 也不能回收。带子任务标记的现场不由 prune 自动删除；有产物的环境需先完成 canonical 集成或显式 discard，再由用户核对是否强制移除。来源 artifact、结果和集成记录仍保存在 runtime 数据库中。父完成门槛依据这些持久事实，不能用一句子摘要代替代码集成与主目录验证。

子集成验证也接受版本化命令的当前收据，复用 [交付证据](delivery-evidence.md) 的版本与环境核对。新流程无需为获取摘要而开放父目录写权限。
