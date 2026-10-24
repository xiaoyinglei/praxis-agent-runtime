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
- **删除保护**：有未提交变更、或有 worktree 分支上未合并的提交时，`remove` 拒绝执行并说明原因，需要显式 `--force`；无法判定状态时同样按受保护处理。`prune` 永不删除受保护项。
- **防嵌套**：在 worktree 里再执行 `--worktree` 会拒绝，避免 worktree 套 worktree。

## 限制

- worktree 是全新检出：`node_modules`、`.venv` 等被忽略的依赖目录不会出现，需要时在 worktree 里重新安装，或用 `.worktreeinclude` 复制小体积必需文件。
- 子模块不会被递归检出；Git LFS 等仓库级过滤驱动器按 git 默认行为执行（Codex 会在复制时禁用过滤器，这里保持普通 `git worktree add` 语义）。
- 和主检出一样，worktree 依赖 `.praxis/` 在 `.gitignore` 中（本仓库已配置）。
- 主检出的未提交修改不自动带入；这是有意的：把"半成品"隔离在外正是本特性的目的。
