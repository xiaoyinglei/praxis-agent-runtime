# Praxis 使用指南

本指南按实际使用顺序组织：配置模型，进入项目，完成任务，再按需要连接外部服务、增加资料权限和长期记忆。项目定位见 [README](../README.md)，源码与验证流程见[开发指南](development.md)。

## 安装与首次配置

普通用户在 macOS 或 glibc Linux（arm64/x86_64）执行：

```sh
curl -fsSL https://raw.githubusercontent.com/xiaoyinglei/praxis-agent-runtime/main/install.sh | sh
```

安装器下载与本机匹配的独立发行包，核对 SHA-256，再安装自带 Python 和依赖的 Agent。用户不需要安装 uv、pip、Python，也不需要创建或激活虚拟环境。程序与版本位于 `~/.local/share/praxis/versions`，命令位于 `~/.local/bin/agent`；已有其他方式安装的同名命令不会被覆盖。

需要时安装器会向 zsh/bash 的启动配置追加命令路径。子进程无法修改当前终端的 PATH，因此安装结束会显示立即可用的完整命令；重新打开终端后可以直接运行 `agent`。相同安装命令也用于更新：下载或解包失败不会替换旧启动入口，之前的版本保留在 `versions` 中。

安装后进入自己的项目，执行 `agent setup`、`agent chat`。安装器不会读取 Key 或调用模型。Linux 的命令执行另需系统沙箱后端，见 [Ubuntu 部署](../DEPLOY_UBUNTU.md)。

可通过 `PRAXIS_INSTALL_DIR`、`PRAXIS_BIN_DIR` 指定绝对安装路径，通过 `PRAXIS_VERSION=v0.1.1` 安装指定 GitHub 发行版。这些变量需传给管道右侧的 `sh` 进程，例如 `curl ... | PRAXIS_VERSION=v0.1.1 sh`。

已有 Python 环境的用户可以继续使用 PyPI 安装包 [`praxis-agent`](https://pypi.org/project/praxis-agent/)。下面只是 uv 安装示例；uv 不是普通用户的前置要求：

```bash
uv tool install --python 3.12 --link-mode copy praxis-agent
cd /path/to/my-project
agent setup
agent chat
```

后续示例中的 `uv run agent ...` 适用于源码安装；使用上面的 PyPI 安装方式时，直接运行 `agent ...`。

参与开发或使用源码示例时：

```bash
git clone https://github.com/xiaoyinglei/praxis-agent-runtime.git
cd praxis-agent-runtime
uv sync --locked --link-mode copy
uv run agent setup
uv run agent chat
```

使用复制方式安装依赖，避免工作区中的依赖文件与外部 uv 缓存形成硬链接，触发沙箱包含性检查。Linux 还需要 Bubblewrap 和 libseccomp，见 [Ubuntu 部署](../DEPLOY_UBUNTU.md)。

公开版默认使用 DeepSeek（`deepseek-flash`），首次向导直接进入默认模型的 API Key 输入。内置目录也提供已核验声明的其他服务商型号；安装包不包含开发者的凭据、个人模型注册表或 MLX 模型权重。向导验证失败不会改变当前模型选择，也不会保存本次输入的新 Key。模型验证会调用真实服务，产生实际用量。

使用目录已有型号时，运行 `agent setup --model MODEL_ID` 隐藏输入对应 Key，不需要手填上下文容量。目录外的服务商或本地模型按下文的 `agent model add` 接入；本地模型服务必须已经启动。

```bash
uv run agent setup --model MODEL_ID --no-mcp
```

macOS 将新 Key 保存到 Keychain；其他 POSIX 系统使用个人配置目录下的 `credentials` 私有目录和文件，权限分别为 `0700`、`0600`。Linux 使用受文件权限保护的本地明文存储。Agent 文件工具与子进程沙箱不能读取该凭据目录。

环境变量优先于保存的个人凭据。凭据绑定变量名和准确服务地址，切换地址不会自动使用旧地址保存的 Key。项目 `.env` 只提供已校验模型定义声明的 `api_key_env`，不会用来改变权限、MCP 或宿主配置；`PRAXIS_DISABLE_DOTENV=1` 可禁用自动加载。

个人配置默认位于 `~/.config/praxis`，可通过宿主环境变量 `PRAXIS_CONFIG_DIR` 指定。项目的模型选择和聊天记录默认位于 `.praxis/`，不应提交到 Git。

### 选择其他项目

CLI 以当前工作目录作为项目。Praxis 的源码和依赖可以放在一个目录，待处理项目放在另一个目录：

```bash
cd /path/to/my-project
uv run --project /path/to/praxis-agent-runtime agent setup
uv run --project /path/to/praxis-agent-runtime agent chat
```

上面的 `--project` 选择 uv 使用的安装项目，当前目录仍然是 Agent 的工作区。后续示例省略这部分前缀，默认在 Praxis 源码目录执行。

## 内置模型目录

主流云模型的维护源是 [`configs/model-catalog.yaml`](../configs/model-catalog.yaml)，本版核对截止日期为 **2026-10-22**。公开运行配置 `configs/agent-models.yaml` 及研究配置中的对应条目由它同步生成，安装包沿用公开配置；CLI 与 Python SDK 使用同一注册流程。维护方式与协议限制见 [`configs/model-catalog.md`](../configs/model-catalog.md)。

快照覆盖 OpenAI、Anthropic、Google、DeepSeek、Kimi、Qwen、智谱、MiniMax，并核对项目已有 Groq 型号。`chat_completions` 表示声明与当前接口匹配，不等于账号已授权或真实任务通过；`catalog_only` 保留官方容量与协议限制，不进入可运行列表。尚无明确整数的容量保留原始单位，不从 `1M` 猜数值。当前多模态消息协议未接通，目录中的厂商视觉能力不等于 Agent 已能读取图片。

例如使用已内置的 Kimi：

```bash
uv run agent setup --model kimi-k3 --no-mcp
uv run agent chat --model kimi-k3
```

## 接入模型

模型服务需要提供 OpenAI-compatible 接口，并满足配置中声明的能力：真实流式文本增量与权威完成状态、合法工具调用，以及启用时的结构化输出。模型 ID 必须与服务实际提供的 ID 一致。

已有个人注册表若与新版内置型号重名，启动会明确拒绝冲突，不会覆盖用户定义；升级前需保留个人文件并解决同名条目。内置模型定义只读；目录外的用户模型通过 CLI 登记，不需要修改源码配置：

```bash
uv run agent model list --source
uv run agent model current

uv run agent model add MODEL_ID \
  --provider openai_compatible \
  --base-url https://models.example.com/v1 \
  --api-key-env MY_MODEL_TOKEN \
  --context-window-tokens 131072

uv run agent model show MODEL_ID
uv run agent model probe MODEL_ID --level full
uv run agent model update MODEL_ID --timeout-seconds 90
uv run agent model switch MODEL_ID
uv run agent model remove MODEL_ID
```

以上地址、ID 和变量名是占位示例，实际凭据由宿主环境提供，或通过 `agent setup` 隐藏输入。注册表保存环境变量名称，不保存它解析出的值。高级定义可通过 `agent model add --from one-model.yaml` 导入；`--skip-probe` 跳过联网验证，并将模型报告为未验证。

`add`、`update` 在完整验证成功后才提交注册表。首次使用会自动初始化本地绑定信任信息，无需手工初始化；已有历史绑定时若信任密钥丢失，需要恢复原密钥。

聊天内输入 `/model` 选择模型，或输入 `/model MODEL_ID`。切换影响下一条消息，保留聊天历史；已经暂停的 Turn 继续使用原冻结绑定，不会改用新模型。

### 本地 MLX

Apple Silicon 可安装本地推理依赖：

```bash
uv sync --locked --extra local-mlx --link-mode copy
```

先在一个终端启动模型，再在另一个终端进入 Agent：

```bash
# 终端 1
uv run --extra local-mlx python -m mlx_lm.server \
  --model mlx-community/Qwen3.5-9B-4bit \
  --host 127.0.0.1 --port 8080

# 终端 2：主动接入本地服务，再选择它
uv run agent model add mlx-community/Qwen3.5-9B-4bit \
  --provider mlx \
  --base-url http://127.0.0.1:8080/v1 \
  --context-window-tokens 262144 \
  --timeout-seconds 300
uv run agent model switch mlx-community/Qwen3.5-9B-4bit
uv run agent chat --model mlx-community/Qwen3.5-9B-4bit
```

Praxis 不会自动启动或关闭模型进程。`/model` 改变 Agent 选择，不能替你切换占用同一端口的模型服务。

## 聊天、任务与恢复

```bash
uv run agent chat
uv run agent run "解释项目入口" --no-require-workspace-change
uv run agent run "修复失败的边界条件并验证"
```

`chat` 默认使用全屏终端界面；`--ui inline` 使用原生滚动历史。操作和简短结果依次显示，当前进度位于输入框上方。Ctrl+O 切换详细记录，可查看参数、长输出、原始错误和用量；再次按下返回，保留输入草稿。Ctrl+J 换行，Ctrl+C 停止当前执行；Esc 优先关闭菜单或详情，执行中可请求停止。停止请求发出后会等待清理结果，结果不确定时会明确提示。默认使用终端原生选择，F2 开启应用鼠标处理，`/copy` 请求复制最近回答。

单步失败会留在对应操作下，不代表整项任务失败；只有实际开始恢复时才显示恢复状态。“本轮已结束”也不代表任务已验收：完成摘要会区分“未运行验收”“已验证”和“验收未通过”，并展示实际验证范围。文件路径仅作为文件引用，不代表内容和排版已经验证。

同一聊天可以连续追问。执行中直接输入补充要求并按 Enter：输入先持久化，界面显示“补充已接收 · 等待应用”，进入下一次普通模型请求后显示“已应用”。模型正在生成时收到的新要求会阻止旧答复宣布完成；已启动的工具收集真实结果，尚未启动的旧动作不再执行。长工具执行期间补充保持等待应用，需要立即停止时使用 Ctrl+C；已发生的修改不会回滚。

审批时可用 Tab 切换到补充输入框，普通文本不会成为批准。提交补充后，原审批动作被作废，模型重新规划；新的动作仍遵守现有权限。若想换成独立任务，先 Ctrl+C 停止，再发送新任务。接收失败会保留草稿；预算耗尽或取消时会显示补充尚未生效。完整契约与恢复边界见[执行中补充输入](interactive-steering.md)。

`run` 默认在模型结束时结束本轮，`validation_status=not_evaluated`，不据此声明任务验收通过。`--require-workspace-change` 只要求发生真实修改；需要机器验收时，使用 `--acceptance-command "uv run pytest -q"`，SDK 对应 `acceptance_command="uv run pytest -q"`。Runtime 只认用户指定命令在工作区根目录的真实执行：最新一次必须成功，之后再次修改须重新验收；其他测试或模型自述不能替代它。验收命令沿用普通工具权限与审批，不会自动获得写入或联网能力。新命令不会改写旧 Turn 的冻结验收规则。

```bash
uv run agent run "修复边界条件" --require-workspace-change \
  --acceptance-command "uv run pytest -q" --allow-execute-tools
```

命令通过仅证明用户定义的检查通过，不能证明未覆盖的业务要求。验收脚本及其依赖应来自可信项目配置；命令文本的冻结不等于冻结所有依赖文件的内容。

退出或断开 SSH 后，可以重新打开历史：

```bash
uv run agent resume
uv run agent resume thread_ID
uv run agent chat --continue
```

聊天内使用 `/resume`，开始空上下文聊天使用 `/new`。浏览与回放历史不会执行旧工具；下一条消息继续原聊天。正在由其他进程执行的任务只显示状态，目前不支持实时接管另一个进程的终端。

“打开聊天”和“恢复暂停任务”是两个操作。后者使用原 Turn 的确认选项与冻结配置：

```bash
uv run agent recover turn_ID
uv run agent recover turn_ID --action continue
```

需要确认的任务可以在聊天界面中完成审批；非交互运行会暂停并显示恢复入口，不会自动批准。执行步骤预算使用 `--max-steps`；`--rollout-db` 指定完整事件与状态数据库。

每轮结束、失败或暂停等待审批时，默认显示本轮累计输入、输出和总 Token，以及提供方已报告的缓存命中和推理用量。缓存命中已包含在输入中，推理已包含在输出中，不会重复相加。用量包含该 Turn 的摘要和重试请求；恢复后继续累计。连接结果不确定时，已记录的 Token 会标注“用量不完整”。这些字段也可从 SDK 的 `result.usage` 读取。

## 并行会话：Worktree 隔离

多个窗口在同一个项目目录同时开会话会互相覆盖文件。`agent chat --worktree`（或 `agent run … --worktree`）在新窗口里为会话创建独立 Git worktree 检出和 `worktree/<名字>` 分支，会话只在自己的检出里读写；主窗口不受影响。`agent worktree list | remove | prune` 管理这些检出，交付照常走 `/commit` 和 `/pull`。设计细节与限制见 [Worktree 隔离](worktrees.md)。

## 检查点与回滚

一轮对话即将首次修改工作区文件时，Runtime 会先把整个工作区快照为一个**检查点**。检查点存放在用户配置目录（`~/.config/praxis/checkpoints/`，按工作区路径区分）的**影子 git 仓库**里：它只被 Praxis 读写，使用独立的 git 元数据，项目自身的 `.git` 目录始终保持只读，两者互不影响。快照范围与 Runtime 判定“工作区已修改”的范围一致：`.git`、`.praxis`、`.venv`、`node_modules` 和各类缓存目录不进入检查点。

在聊天中输入 `/rewind` 查看检查点列表（按时间倒序，含每轮的输入摘要和变更文件数），选择一项确认后：

- 修改过的文件恢复为检查点时的内容；
- 检查点之后新增的文件会被删除；
- 检查点从未见过的文件（例如你自己放在工作区、但从未被任何写入轮次快照到的文件）不受影响；
- 回滚前会先把当前状态保存为新的检查点，因此回滚本身也可以再次回滚。

回滚只还原文件，不改写聊天历史；`/rewind <序号>` 或 `/rewind <检查点ID前缀>` 可直接指定目标。回滚前必须没有仍在运行或等待输入的任务轮次。SDK 对应 `agent.list_checkpoints()` 与 `agent.rewind(checkpoint_id)`。

检查点保留约 30 天，过期后由后续会话自动清理；工作区目录被删除后，对应的影子仓库也会随之清理。该功能依赖本机 `git` 命令：缺失时检查点自动停用（不影响任何其他能力）。检查点是安全网而不是验证证据：回滚到某个状态不代表该状态通过了验收，`AgentResult.validation_status` 的语义不变。

已知边界：嵌套 git 仓库（工作区里的子仓库）按 gitlink 记录，内部文件不进入快照；文件权限位只保留可执行位；删除文件后可能留下空目录。

## 提交变更到 Git 仓库

`/commit <提交说明>` 把工作区当前变更提交到仓库的 `praxis/*` 分支。这是一次**由你显式触发的交付操作**：模型和沙箱依然无法写 `.git`，提交由 Runtime 在宿主侧执行，并且执行前会列出完整的待提交清单请你确认。

确认后的行为：

- 从当前 `HEAD` 创建 `praxis/<时间戳>` 分支（你已在该类分支上时直接续用，多次 `/commit` 会在同一分支上堆叠）；也可以用 `agent.commit(message, branch_name=...)` 指定分支名；
- `git add -A` 暂存预览中列出的全部变更——仓库自己的 `.gitignore` 决定什么算变更，你自己未提交的改动也会一并包含，所以确认前请看清单；
- 以你本机的 git 身份提交（沿用你的提交钩子与签名配置），提交后工作区文件内容不变，只有分支指针移动；
- 需要在仓库根目录执行；仓库存在未完成的 merge/rebase 等操作、没有变更、或存在未结束的任务轮次时会拒绝。

提交是普通 git 操作：想撤销用 `git reset --soft HEAD~1`，想合并回主线用你惯用的 merge/rebase 流程。SDK 对应 `agent.commit_status()`（只读预览）与 `agent.commit(message)`。

`/commit` 创建分支时会记住从哪个分支分出（存在仓库本地配置里），供 `/pull` 使用。

## 推送并创建 Pull Request

`/pull [标题]` 把当前 `praxis/*` 分支推送到 origin 并创建 Pull Request。流程同样是**预览 + 确认**：先列出将进入 PR 的提交和变更文件（工作区若有未提交变更会提示，它们不会包含在内），分支已有 PR 时直接给出链接。

- 推送用本地 `git push`，远端分支与本地提交完全一致（后续 `/commit` + `/pull` 会自然更新同一 PR）；标题缺省取最新提交说明，正文列出全部提交；基准分支默认取 `/commit` 记录的分出点，无记录时取仓库默认分支
- PR 通过 GitHub CLI（`gh`）创建，使用 `gh` 自己的登录凭据；推送需要 Git 凭据，未配置时错误信息会提示运行 `gh auth setup-git`
- 前置条件：origin 是 GitHub 仓库、已安装并登录 `gh`、当前在 `praxis/*` 分支且相对基准有提交

SDK 对应 `agent.pull_request_status()`（只读预览）与 `agent.open_pull_request(title=..., body=..., base_branch=...)`。这是对外发布操作：执行前必须经用户确认，存在未结束任务轮次时拒绝。

## 让模型请求交付（commit_changes / open_pull_request）

除了你手动执行 `/commit` 和 `/pull`，模型也可以通过两个常驻工具**自己发起交付请求**：`commit_changes`（提交到 `praxis/*` 分支）和 `open_pull_request`（推送并创建 PR）。这是 Claude Code / Codex 的同款交互：模型发起、你确认、Runtime 在宿主侧执行——模型与沙箱依然碰不到 `.git`，执行用的是你本机的 git 身份和 gh 凭据。

审批遵循和其它工具完全相同的模式语义，没有专门规则：

- **请求批准 / 帮我批准**：每次调用都会暂停等待你确认，审批面板显示提交说明或 PR 标题等参数预览；
- **完全访问**：自动放行（等同 Claude Code 的 bypass 模式）；
- 权限策略里的 `tools` 规则照常生效（例如 `commit_*` deny 可以整体禁掉模型侧交付）；`open_pull_request` 的推送受网络授权约束，域名过滤策略下会被拒绝。

两个入口（斜杠命令与模型工具）产出完全相同的结果；工具返回提交 SHA、分支名或 PR 链接，模型据此继续（例如提醒你观察 CI 或直接调用后续步骤）。

## 查看 CI 状态

`/checks [分支]` 读取当前分支（或指定分支）PR 的检查状态，`/checks --wait [秒]` 会等待到全部结束（默认最长 600 秒，Ctrl+C 取消）。全部通过、仍在进行、没有配置检查分别明示；失败时列出失败项并展示失败日志摘录。

对失败的 PR，确认后可以把失败项和日志摘录组装成一条消息**直接交给 agent 分析修复**——agent 修完后提醒你 `/commit`、`/pull`，然后再次 `/checks`，直到变绿。CI 信号就这样以普通用户消息进入对话，不引入任何模型侧的新权限。

SDK 对应 `agent.pull_request_checks(branch=..., include_failure_logs=..., wait_seconds=...)`。该功能同样依赖本机 `gh` 登录。

## 后台观察 CI：agent watch

聊天里的 `/checks --wait` 会占住当前终端。想让等待和失败分析自动进行，另开一个终端（或配合 `nohup`/`tmux`）运行：

```bash
agent watch                     # 观察当前（或最新）praxis/* 分支，最长 30 分钟
agent watch --timeout 3600 --interval 60
agent watch --branch praxis/x --thread <会话ID> --approval-mode auto
```

`agent watch` 轮询该分支 PR 的检查：全部通过即结束（退出码 0）；失败时（退出码 1）自动把失败项与日志摘录组装成一条消息，提交给该工作区**最近会话**（或 `--thread` 指定的会话）的 agent 分析修复——续写发生在原会话里，之后用 `agent resume` 打开就能看到这轮分析与工作区修改。等待超时、分支/会话无法确定、会话被其他进程占用分别明示（退出码 2）。`--no-fix` 只观察不调用模型。

边界与约定：观察进程是非交互的，默认使用与聊天一致的“帮我批准”审批模式（可用 `--approval-mode` 覆盖），沙箱与权限上限照常生效；任务中途需要审批时会暂停并提示用 `agent resume` 继续。交付动作保持用户确认——watch 只做观察与分析，`/commit`、`/pull` 仍由你在聊天中执行。

## 服务连接与按需登录

```bash
uv run agent connections
```

本机页面提供 GitHub、Notion、Linear 预设，也可填写自定义远程 HTTPS MCP 地址。可以添加、授权、检查和移除连接。页面仅监听 `127.0.0.1`，使用每次启动生成的随机会话地址；该地址相当于本次管理权限，请勿分享。

个人远程 MCP 的使用流程是：

1. 普通聊天只读取上次检查生成的本地工具目录，不连接远程服务或读取其凭据。
2. 工具实际被调用时，才连接对应服务，并核对当前策略、工具定义、效果、服务身份和凭据版本。
3. 没有工具目录、认证失败或目录需要更新时，显示连接提示；点击“打开连接管理”、按 F3 或输入 `/connections` 后再选择登录。
4. 登录并检查成功后，下一条聊天消息重新加载工具，保留历史。正在执行或恢复的 Turn 不会被替换工具快照；SDK 调用者需重新打开 Session。

已有账号可以静默刷新 Token，但不会在任务里自动打开授权页。Linux 和无本机浏览器的服务器可以使用 `--no-open` 显示地址；从远程电脑访问需自行配置 SSH 端口转发，不应把管理端口开放到公网。聊天退出会关闭由该聊天打开的管理服务。

### GitHub 与自定义服务

GitHub 预设复用本机 GitHub CLI 登录，默认只读仓库、Issue 和 PR：

```bash
gh auth login --hostname github.com
uv run agent mcp add github
uv run agent mcp check github
```

其他服务可以通过 CLI 登记：

```bash
uv run agent mcp add docs --url https://mcp.example.com/mcp --oauth
uv run agent mcp login docs
uv run agent mcp check docs
uv run agent mcp list
uv run agent mcp remove docs
```

也可使用 `--bearer-token-env-var DOCS_TOKEN`，在宿主环境中提供实际值。`list` 不连接远程服务；`check` 只验证认证并发现工具，不调用模型或执行服务工具。显式工作区或外部配置可通过 `agent mcp check NAME --config PATH --workspace PROJECT` 检查；这表示用户信任该配置文件，仍需已有网络授权。

默认只接入服务声明为只读的工具。确需写入时，注册时显式使用 `--allow-actions`，非只读动作仍需单独审批。服务的只读声明属于元数据，账号与服务端权限仍然重要。

通用浏览器 OAuth 支持 Authorization Code + PKCE、服务发现、动态客户端注册和刷新。服务要求预登记时可提供 `--oauth-client-id`、`--oauth-callback-port`；委托其他认证域名需显式提供 `--oauth-domain`。发现认证地址不会扩大模型工具的网络权限。

个人连接保存在 `~/.config/praxis/mcp_servers.yaml`；工具目录位于同一配置根目录下的 `mcp_catalog`，不含实际凭据。配置或凭据变化会使旧连接失效，旧审批不能沿用。新增加的网络授权可能还需在聊天中显式 `/permissions reload`；管理员 deny 始终有效。

本地 stdio MCP 保留原有沙箱启动方式，Linux 下保持只读、禁网络。项目提供的是 MCP 连接目录，不会下载执行任意第三方插件。移除本机连接不会撤销服务商侧的账号授权。

## 审批方式

新聊天和 `agent run` 默认使用“帮我批准”。聊天底部持续显示当前模式，按 **F4** 或点击底部的权限入口可打开选择器；`/permissions` 也保留可用。用方向键或鼠标选择，Enter 确认，Esc 返回。切换模式不需要退出当前任务，未发送的草稿会保留。

| 审批模式 | 行为 |
| --- | --- |
| 请求批准（`ask`） | 读取允许的文件；写入、进程执行和联网需要审批或已有明确授权 |
| 帮我批准（`auto`） | 自动执行权限范围内的普通文件编辑、只读且断网的沙箱命令、公开网页读取；其他动作弹出审批 |
| 完全访问（`full`） | 自动批准当前权限范围内的普通工具动作，包括命令的联网请求；明确拒绝、强制确认、管理员上限和沙箱保护仍有效 |

需要确认时，面板提供“允许一次”“本会话允许同类命令”（符合条件时显示）和“拒绝”。直接选择即可，不需要输入字母指令。普通终端的单次 `run` 也使用选择对话框；非交互输出仍保留可恢复的暂停记录。

审批面板还显示当前模式与授权范围，并提供“更改审批模式…”入口。切换后，运行时按新策略重新检查当前动作：允许则继续，需要确认则保留审批，明确拒绝则不执行。取消模式选择只返回原审批。选择“完全访问”后，当前权限允许的命令写入和联网可自动继续；强制确认和明确禁令仍然有效。

例如，Python 计算数据时可以在“帮我批准”下自动执行，但生成报告或图片需要命令写入工作区，因此仍会请求批准。可在当次审批中选择“允许一次”，或直接切换到“完全访问”以继续后续权限范围内的操作。切换模式与单次批准会分别记录，崩溃恢复不会把策略自动执行变成人工授权。

同类命令授权保留完整命令的 token 前缀，只允许追加参数，面板会展示具体范围。例如批准 `git status` 后，可在相同目录执行 `git status --short`，不会批准 `git reset`、追加管道或 shell 替换、写入或联网升级。只有只读、断网沙箱里的简单命令提供该选项；复合 shell 和解释器脚本仍需单次审批。授权绑定工具版本、其他完整参数、目录及当前权限，不能突破权限上限。

`/permissions grants` 查看临时命令授权，`/permissions revoke` 全部撤销。临时授权只属于当前打开的 Session，退出或重新打开聊天后失效；审批模式随聊天保存，切换聊天恢复其自己的模式。`/permissions show` 查看权限策略；原有 `load`、`profile`、`reload` 子命令继续可用。

也可以在启动时选择模式；显式参数覆盖所打开聊天的保存模式：

```bash
uv run agent chat --approval-mode auto
uv run agent run "检查项目测试" --approval-mode ask
uv run agent resume <thread-id> --approval-mode ask
```

Python SDK 使用同一策略。省略模式时保留既有 SDK 默认行为；同一 `Session` 内的授权可以跨多轮请求使用：

```python
import asyncio
from agent_runtime import Agent

async def main():
    agent = Agent(workspace_path=".", approval_mode="auto")
    async with agent.session() as session:
        result = await session.submit("检查项目测试")
        print(result.answer)
        # 暂停时由宿主界面展示 result.pause.options，再提交用户选择。
        # await session.resume(result.turn_id, "approve")
        # session.revoke_approval_grants()

asyncio.run(main())
```

## 权限与跨目录资料

工作区决定默认范围，权限策略决定额外允许的范围。模型工具不能通过修改策略文件为自己授权。

```bash
uv run agent permissions
uv run agent permissions --profile read-only
uv run agent chat --permissions /absolute/praxis-permissions.toml
```

| 模式 | 默认范围 |
| --- | --- |
| `workspace` | 工作区内可读写；写入和执行仍受工具授权及审批约束 |
| `read-only` | 只读，拒绝写入及不符合只读契约的动作 |
| `trusted-local` | 广泛读取本机文件；写入仍默认限制在工作区 |

例如，只允许读取另一个资料目录，并拒绝其中的私有子目录：

```toml
profile = "workspace"
approval = "on-request"

[[filesystem]]
path = "/absolute/reference"
access = "read"

[[filesystem]]
path = "/absolute/reference/private"
access = "deny"

[network]
decision = "ask"
```

文件规则覆盖路径及其子树。更具体的规则覆盖更宽泛的规则，同一路径 deny 优先；符号链接实际目标和硬链接身份也参与检查。`.git`、`.agents`、`.codex`、`.aws`、凭据目录和当前策略来源保持保护。

个人策略默认在 `~/.config/praxis/permissions.toml`，也支持 JSON。只有显式选择才加载项目策略，管理员策略另行与个人、会话策略取交集。工具 allow、手动审批和 trusted-local 模式都不能越过管理员 deny。

```text
/permissions
/permissions load /absolute/praxis-permissions.toml
/permissions profile read-only
/permissions reload
```

禁止规则会在后续检查中收紧权限，新增授权需要显式接受。撤权不能回滚已经发生的写入或远程副作用。公网读取、Shell 网络与远程 MCP 权限分别约束，不是一个全局“联网开关”。

macOS 子进程通过 `/usr/bin/sandbox-exec` 和 Seatbelt 隔离。Linux 通过 `/usr/bin/bwrap` 和 Bubblewrap 默认只读、禁网络；批准的联网要求 libseccomp，批准的工作区写入先在候选副本运行，通过检查后才发布。Linux 子进程外部目录写入暂不支持；缺少后端会明确报告不可用并拒绝执行。测试中的 fake sandbox 只用于流程验证，不是实际隔离证据。

Linux 候选视图支持工作区内相对符号链接，保留普通文件 mtime，按块复制与指纹校验，不再有普通文件总量 64 MiB 上限；仍需要候选副本磁盘空间。
`run_command` / `execute_python` 的 `publish_on_failure=true` 可显式保留通过安全检查的失败或超时产物，
方便后续命令增量续建；工具仍报告失败，`workspace_published` 说明是否发布。默认关闭，取消及沙箱启动失败不发布。
它不增加写权限或取消 destructive 审批；不恢复已终止的进程。详见[输出与构建恢复](runtime-recovery.md#linux-输出与构建恢复)。

`search_text` 使用随安装依赖提供的 ripgrep，不依赖系统 `PATH` 上的 `rg`。运行时负责读取权限、
路径与 glob 范围、资源限制和结构化输出；匹配交给 ripgrep。结果按路径顺序返回，不推测源码重要性。
`regex=true` 使用 ripgrep 的 Rust 正则语法，`regex=false` 搜索字面文本；省略时保留自动识别模式。
搜索在独立进程运行，默认 15 秒超时，取消会终止并回收搜索进程组，不阻塞聊天事件循环。
单文件最多读取 2 MB，合计 64 MB，最多枚举 20,000 个目录项；后端 JSON 与最终输出分别有
8 MB、1.5 MB 上限。返回内容或扫描范围不完整时，`truncated=true`；不能据此断言整个目录没有匹配。
只有精确文件已读取且结果未截断，才能在该次观察范围内确认文本不存在。符号链接不跟随；
显式授权的外部路径仍受原权限上限约束。缺少后端依赖会明确报告 `search_backend_unavailable`。

`inspect_data_file` 用 `openpyxl` 的只读流式接口读取 XLSX/XLSM，用标准库 `csv` 读取 CSV/TSV。
解析在独立子进程运行；取消或 60 秒工具超时会终止并回收解析进程，退出聊天不再等待解析线程。
子进程使用当前读取权限的快照，不继承宿主凭据或工作区的 Python 导入路径。
表格预览是原始网格，不推测表头；标题、单位、多层字段行和空行都留在 `tables[].rows` 中。
`row_numbers`、`column_numbers` 给出每个返回值的一起始源坐标；CSV 行号指逻辑记录，带引号字段里的换行不另计一行。
先查看原始网格，再显式选择用于分析的字段行和数据区域；转成 pandas DataFrame 时可先用 `header=None`，
确认结构后再指定 `header`，不要把位置假设当成源文件事实。

通过 `sheet_name` 选择工作表，或用 `sheet_offset`、`max_sheets` 分页；
通过 `start_row`、`start_column`、`max_rows`、`max_columns` 读取源区域，起始行列默认均为 1。
默认每张表预览 10 行、20 列，包含标题或字段行；`next_row`、`next_column` 继续当前网格，
`next_sheet_offset` 继续工作表列表。尚未扫描到文件末尾时，`row_count`、`column_count` 为 `null`，
不会把工作簿声明的尺寸当成核实过的总量。空预览或越界读取不能证明内容已检查。

单次调用最多返回 2000 个网格单元格，`max_characters` 限制合计值与格式文本（默认 8000，硬上限 32000 字符），
每个文本另限 2000 字符；`truncated_cells` 指明文本或格式被缩短的源坐标。
`cell_types` 保留文本、数字、日期、公式和错误的类型，`number_formats` 保留 Excel 格式码（空字符串表示 General 或不适用）。
数值是存储内容，不模拟 Excel 格式化显示；公式保留表达式或数据表公式属性，不执行计算，也不把缺失缓存当空白值。
需要完整长文本、合并区域等更多工作簿元数据，或需要按已确认表头转换数据时，使用 `execute_python` 中的 `openpyxl` / `pandas`。
解析成功和文件有效不等于业务分析正确。

这次原始网格契约升级为 `builtin-inspect-data-file-v2-raw-grids`，移除了推测性的 `headers` 字段，
`row_count` 现在包含全部原始行。旧工具版本的待恢复调用会按执行版本检查拒绝，需重新提交检查；已保存的历史结果不重写。

`execute_python` 使用 Praxis 的托管解释器和数据处理依赖，每次启动独立进程。
传入 `code` 执行内联代码，或传入工作区相对路径 `script_path` 执行已有脚本，两者只能选一个。
脚本以 `__main__` 执行，并可导入脚本同目录模块；内联代码可导入 `working_dir` 中的模块。
它使用托管环境，不采用项目 `.venv`；需要项目依赖时用 `run_command` 运行项目解释器。
两种执行工具共用有界输出读取和进程组超时、取消清理。Python 源码按原有语义执行，
不改写 AST、不统计断言；兼容输出字段 `verification` 为 `null`。
stdout、stderr 和退出码记录执行事实。新请求不根据工具名或命令名称自动生成验证结论；
需要机器验收时，使用明确的 `acceptance_command`，检查范围仅是该命令的真实执行结果。
旧 Turn 的完成策略仍按原记录解释，历史验证记录不会重写。读取次数不会强制模型开始修改或隐藏工具。

macOS / Linux 活跃 Session 的 `run_command` 可用 `yield_time_seconds` 先返回任务 ID，随后用
`manage_command` 查询、等待或取消同一个进程。等待期限与执行期限独立；显式
`timeout_seconds=null` 才取消执行期限，省略时仍为 120 秒。取消、abort、Turn 失败或
Session 关闭会清理命令进程组；正常关闭会保存未收集的终态供恢复读取。macOS 写入直接
保留，不做失败回滚。父 Agent 崩溃后可接回原命令；`agent commands <turn_id> <session_id>` 可无模型查询、等待、取消和读取日志。超时严格按秒，不再猜测毫秒。详见[长命令与崩溃重连契约](macos-command-sessions.md)。

## 记忆与私有知识

跨聊天记忆默认关闭，按工作区隔离。显式记住、修改、删除和查询来源不调用模型；启用自动生成后，提取与整理会使用所选模型及独立预算，产生真实用量。

```bash
uv run agent memory status
uv run agent memory configure --use --generate
uv run agent memory update
uv run agent memory remember --key test-command --title "测试命令" \
  --text "uv run pytest" --kind preference
uv run agent memory explain --key test-command
uv run agent memory delete --key test-command
uv run agent memory configure --no-use --no-generate
```

只有允许贡献的聊天会成为自动记忆来源；历史聊天不会因为打开默认设置就自动获得贡献授权。新 Turn 使用固定记忆版本，当前任务中的修改不会改变它已有的读取快照。记忆中的历史信息仍需按当前事实核对，不构成新的授权。

私有文档 RAG 是独立、可选的检索能力：

```bash
uv sync --locked --extra rag --link-mode copy
uv run --extra rag rag --help
```

`rag` 负责索引与检索服务准备，普通 Agent 启动不会初始化 Embedding、重排或向量数据库。先建立索引和对应存储服务，再显式连接 `RAGKnowledgeConfig`；其提供方边界按需初始化。引用提供可追溯性，不自动证明内容正确。

知识工具的 grounding 与证据不足标记只描述该工具的结果，可从 `tool_calls[].structured_output` 读取。
`AgentResult.groundedness` 和 `insufficient_evidence` 保留兼容，均为 `false`，表示最终答案未经此类评估，
不代表最终答案已被证明错误或正确。文档 ID 保留原有字符串、整数或空值，不强制转换为数字。

## 公网检索

`web_search` 与 `web_fetch` 默认向 CLI 提供搜索与网页阅读工具。它们只能访问通过检查的公网地址，不向工具开放任意内网。已配置的搜索会把查询发送给所选服务，读取网页会对目标站点发请求。

默认未配置搜索服务或 Key 时，`web_search` 返回 `web_search_not_configured`，不发请求、不消耗网络请求预算；Agent 仍可启动，`web_fetch` 可直接读取已知公网 URL。使用 `--web-search-provider brave|tavily` 与 `--web-search-key-file PATH` 配置 API 搜索；密钥文件应位于工作区外。仅提供 Key 时沿用 Brave。需要公开 HTML 搜索时可显式选择 `--web-search-provider bing`（不提供 Key），但它依赖页面结构，可能遇到限流或验证，不保证可用性。`--no-web-tools` 关闭默认的公网读取预授权，恢复逐次授权行为。

恢复未完成 Turn 时，搜索服务须与原工具执行版本一致。升级前默认使用 Bing 的待恢复调用，需要显式指定 `--web-search-provider bing`；新版未配置状态使用独立执行版本，不会把旧调用静默切换到其他服务。Brave/Tavily 恢复仍须提供原服务与受保护密钥文件，密钥不会写入 Turn 记录。Python SDK 对应参数为 `web_search_provider` 与 `web_search_key_file`。

网页读取保存来源、内容哈希及快照，长内容可以继续读取。成功取得文字不代表已经找到足够证据，HTTP 403、登录墙、内容截断和资源阻断应单独解释。

需要动态页面时可安装可选浏览器：

```bash
uv sync --locked --extra browser --link-mode copy
uv run --extra browser playwright install --with-deps --only-shell chromium
```

显式渲染沿用同一套 DNS、目标地址、网络权限、请求与字节预算。浏览器不能自行绕过这些边界；当前只允许受约束的 GET 请求，依赖 POST API 的页面可能无法完整读取。

## Python SDK 与扩展

多轮调用使用同一个 Session，资源在关闭时统一释放：

```python
import asyncio

from agent_runtime import Agent


async def main() -> None:
    agent = Agent(workspace_path=".")
    async with agent.session(require_workspace_change=False) as session:
        first = await session.submit("解释项目入口")
        second = await session.submit("继续说明刚才提到的数据层")
        print(first.answer)
        print(second.answer)


asyncio.run(main())
```

SDK 是异步接口。一次执行使用 `await agent.run()`，事件订阅使用 `agent.stream()`，只读结果和待确认项使用 `agent.read_result()`、`agent.pending_input()`；`agent.resume()` 恢复同一个暂停或中断 Turn。

执行中补充使用 `await session.steer(text, expected_turn_id=turn_id)`；也可用 `agent.steer()` 从另一控制端提交，无需打开模型。`expected_turn_id` 必须来自本次执行事件，避免把消息误投新任务。返回值是已提交的用户消息 Item，`session.input_status(receipt.item_id)` / `agent.input_status(...)` 返回 `pending`、`applied` 或 `unconsumed`。`applied` 表示已进入普通模型请求的 dispatch，不保证模型理解或请求成功。补充不会增加预算、放宽权限或代替审批。

接入已经准备好的私有索引：

```python
import asyncio

from agent_runtime import Agent, RAGKnowledgeConfig


async def main() -> None:
    knowledge = RAGKnowledgeConfig(
        storage_root="data/indexes/private_docs_v1",
        vector_backend="milvus",
        vector_collection_prefix="private_docs_v1",
    )
    agent = Agent(workspace_path=".", knowledge=knowledge)
    result = await agent.run("查找与问题相关的制度依据并标注来源", require_workspace_change=False)
    print(result.answer)


asyncio.run(main())
```

扩展沿用统一的工具权限、参数验证、审批、结果和事件记录。项目支持工作区 Skills、配置的 MCP 和受预算、权限约束的子智能体；它们是任务能力的扩展，不替代核心执行链路。具体接口以源码与公共测试为准，见[开发指南](development.md)。

## 项目规则与执行收尾

在工作区根目录放置 `AGENTS.md`，下一轮请求会自动加载。相同目录的 `AGENTS.override.md` 优先，空的 override 文件也会覆盖普通文件。根目录规则适用于工作区及其子目录；子目录可以提供更具体的同名规则，冲突时更深目录优先，用户明确指令优先于项目规则。

规则首次发现时会保存路径、作用目录、完整正文与 SHA-256，进入本轮持久绑定及每次模型请求；恢复任务和上下文压缩后仍会重新注入。同一轮使用已保存版本，文件修改在新一轮生效。规则总量最多 32 KiB（UTF-8），无法读取、编码错误或超过上限时会明确报错，不会截断后假装已经读完。工作区根目录是发现边界，不读取父目录或其他产品的个人配置，规则也不能授予权限。

子目录规则按文件工具的结构化目标、命令工作目录和声明的文件目标逐级发现。首次遇到尚未展示的规则，原动作会以 `project_instructions_required` 暂缓，下一次请求先展示规则，再由模型调整或重试动作。这也适用于直接写文件。运行在根目录的脚本如果间接访问其他子目录，Runtime 无法静态知道脚本的全部文件访问；需要显式指定相关工作目录或先读取对应目录的文件。项目规则是模型指导，不是文件系统权限的替代品。

默认每轮最多 128 步，CLI 的 `--max-steps`（旧名 `--max-turns`）或 SDK 的 `max_steps` 可显式覆盖。每步展示剩余预算，最后两步提醒收尾。达到上限仍未结束时，状态保持失败，提供持久化的工具结果统计和修改保留说明；不会追加未授权的模型请求或把部分工作标成完成。历史任务继续使用原来冻结的预算与恢复策略。

新任务还限制完成检查自身的连续续跑：完成建议被驳回后，最多允许 8 次没有工具调用或新用户输入的续跑；再次被驳回时，以 `completion_continuation_limit` 失败结束，保留最后一次检查的验证状态、范围和原因。期间实际验收通过即可正常完成，不会因为达到次数而拒绝成功，也不会绕过失败验收宣布完成。工具调用或新输入重置这一计数；它不判断任务进度，不限制合理的重复读取。所有续跑仍消耗同一份步数、token 和费用预算；旧 Turn 不会在恢复时被追加这一策略。

工具结果落盘后的终止检查在正常执行和恢复时走同一路径。恢复不会重跑已提交的工具结果，也不会把同一个完成建议、裁决或恢复反馈重复计算。[终止机制与成熟实现依据](runtime-termination.md)记录了设计边界。

`update_plan` 保存模型报告的步骤状态。模型标记完成不代表任务通过独立验收，运行结束也不会自动勾完剩余步骤：仍有未完成步骤时计划标记为 `needs_replan`，失败时标记为 `blocked` 并保留原步骤状态。普通 CLI 答复始终显示验证状态及范围；`not_evaluated` 表示本轮自然结束，但没有独立任务验收。

同一个缺失资源的重复读取不会因改变分页或输出上限而重置错误计数。连续确定性故障达到阈值后，会提供一次重新规划的反馈；仍然重复才停止。更换资源、修正参数、文件恢复可读或发生真实修改时，按照实际结果重新判断，命令参数不会被一概忽略。

思考强度、单次输出额度、工作中调整和 SDK 用法见[模型请求设置](model-request-settings.md)。
