# 权限策略

Praxis 把工作目录、文件访问、工具授权、网络访问和审批分开配置。标准 CLI 与 `Agent` SDK 使用同一条 `ToolExecutor` 执行链。策略决策由运行时执行，模型没有修改权限的工具。

## 配置与入口

工作目录通过终端当前目录或 SDK `Agent(workspace_path=...)` 选择；外部目录授权独立于工作目录。CLI 在项目目录中执行以下命令。

```bash
agent permissions --permissions /absolute/permissions.toml
agent permissions --permissions /absolute/permissions.toml --path /absolute/data.csv
agent permissions --permissions /absolute/permissions.toml --path /absolute/data.csv --write
agent permissions --permissions /absolute/permissions.toml --tool execute_python
agent permissions --permissions /absolute/permissions.toml --url https://example.com

agent chat --permissions /absolute/permissions.toml
agent run "分析资料" --permissions /absolute/permissions.toml --no-require-workspace-change
agent resume --permissions /absolute/permissions.toml
agent recover TURN_ID --permissions /absolute/permissions.toml --action allow_once
```

`agent permissions` 只解析配置和解释规则，不调用模型，不打开 MCP，不运行命令。工具检查显示命名规则；真正执行时还会检查实际效果和资源目标。

聊天中支持：

```text
/permissions
/permissions load /absolute/permissions.toml
/permissions profile workspace
/permissions profile read-only
/permissions profile trusted-local
/permissions reload
```

`load` 和 `profile` 是用户控制命令，更新会话策略并保存会话记录；不会修改原配置文件。`reload` 显式接受配置文件中新增加的授权。解析失败保留原策略。

个人设置目录可通过宿主环境变量 `PRAXIS_CONFIG_DIR` 选择，MCP 与权限共用该目录。标准 Agent 自动加载个人文件 `~/.config/praxis/permissions.toml`，不存在时尝试同目录 JSON。`--permissions` 或 SDK `permissions=` 替代个人配置。项目内文件只有显式选择才加载，避免仓库自行扩权。

管理员配置为 `/etc/praxis/permissions.toml`，不存在时尝试 JSON；要求 root 所有且组和其他用户不可写。管理员策略独立作为权限上限，与个人、会话策略取交集。SDK 宿主还可以传 `managed_permissions=`。普通本地授权无法突破该上限。

配置支持严格 JSON/TOML，未知字段、重复键和错误类型会被拒绝。相对文件规则按配置文件所在目录解析；SDK 对象里的相对规则按工作目录解析。`~` 可以展开，其他环境变量不展开。有效配置记录绝对来源、规范化规则和 SHA256 策略版本。

## 策略示例

```toml
profile = "workspace"
approval = "on-request"

[[filesystem]]
path = "/absolute/reference"
access = "read"

[[filesystem]]
path = "/absolute/output"
access = "write"

[[filesystem]]
path = "/absolute/reference/private"
access = "deny"

[[tools]]
pattern = "execute_python"
decision = "allow"

[[tools]]
pattern = "mcp__unapproved__*"
decision = "deny"

[network]
decision = "ask"
allowed_domains = ["example.com", "*.example.com"]
denied_domains = ["private.example.com"]
```

文件规则是精确路径及其整个子树，不支持文件 glob。更具体的规则覆盖更宽泛的规则；同一路径 `deny` 优先于 `write`，`write` 优先于 `read`。`write` 同时允许读取、创建、替换、重命名和删除，因此仍需工具级审批。文件路径的原始拼写与符号链接实际目标都要授权；硬链接不会把路径授权转移到另一个目录。大小写和 Unicode 别名按文件系统身份检查。

工具规则使用规范工具名和 `fnmatch` 模式，顺序为 `deny > ask > allow`；具体的 allow 不能覆盖宽泛 deny。命名 allow 不会越过文件边界、网络禁止、管理员上限和不可绕过的效果检查。破坏性操作仍需审批。文件读取、列表、搜索和数据检查都应用相同资源策略；递归读取不会暴露 deny 子目录和文件。

网络域名使用精确主机名或 `*.example.com`。`[[network.rules]]` 可为具体主机配置 allow/ask/deny；匹配规则按 deny > ask > allow 决策，否则沿用全局 decision。全局 deny、denied_domains 和 allowed_domains 的边界仍优先，管理员上限继续取交集。后者只覆盖子域名，不覆盖 `example.com`，也不匹配 `example.com.evil`。deny 优先；有允许列表时，未匹配目的地禁止访问。HTTP 重定向和浏览器资源请求也逐个检查目的地。网络 `allow` 可以预授权符合规则的访问；`ask` 走现有审批或用户显式的公网只读预授权；`deny` 无法靠审批解除。

### 配置方案

| profile | 行为 |
|---|---|
| `workspace` | 工作区可读写，外部资源须显式授权；写工具仍按授权/审批执行 |
| `read-only` | 工具不可修改文件、执行声明为破坏性的动作或未经只读声明的远程 MCP 动作，可增加外部 read/deny；write 声明拒绝加载 |
| `trusted-local` | 广泛读取本机文件，默认写入仍限制在工作区；适合个人本机跨目录分析 |

`.git`、`.agents`、`.codex`、`.aws` 路径组件保持写保护。当前策略来源文件也不能被模型工具或受限子进程改写，包含大小写、符号链接和硬链接别名。自动配置目录整体受写保护，即使候选配置尚不存在，也不能由模型工具创建；子进程不能通过重命名或移除来源文件的父目录替换策略。

`approval="on-request"` 可以请求额外工具授权；`approval="never"` 禁止出现审批，并拒绝仍需审批的动作。它不会扩大文件或网络范围。现有 `--allow-write-tools`、`--allow-execute-tools` 和公网读取开关继续作为预授权，但不能解除策略 deny。

## SDK 与撤权

```python
from agent_runtime import Agent, NetworkPolicy, PathRule, PermissionPolicy

agent = Agent(
    workspace_path="/absolute/project",
    permissions=PermissionPolicy(
        filesystem=(PathRule(path="/absolute/reference", access="read"),),
        network=NetworkPolicy(decision="deny"),
    ),
)

async def analyze():
    async with agent.session(require_workspace_change=False) as session:
        result = await session.submit("读取参考资料并分析")
        print(result.answer)
        print(session.permissions().explain())
        session.update_permissions(PermissionPolicy(profile="read-only"))
        session.update_tool_policy(denied_tool_names=frozenset({"execute_python"}))
```

配置文件在下一次资源检查时重新读取；新的禁止立即收紧授权，新增加的授权需要显式 reload。会话更新也作用于下一次工具调用。批量调用即使已经准备完成，在实际 runner 执行前仍会复查。已经开始运行的子进程使用启动时的 OS 沙箱；撤权不回滚已经发生的效果，也不会自动终止已开始的进程。

Turn/Step 保存权限快照供审计和恢复。重复恢复会合并重复约束，快照不会逐次嵌套增长。恢复调用使用历史范围与当前范围的交集；历史 preapproval 不会覆盖新禁止。普通恢复保留原会话授权，显式会话撤权优先于原 Step。审批绑定参数摘要、目标身份、工具版本和策略版本；发生变化后旧审批失效。子智能体继承父级限制和实时撤权，不能自行增加授权。

成功工具结果记录权限决策、原因和策略版本；被拒绝调用记录明确的错误；审批交互保留目标和授权范围。可使用 `agent history` 和 Rollout 记录检查实际调用。

## 执行边界与平台支持

macOS 的 Shell/Python 使用 Seatbelt，支持具体资源读写范围、deny 子树、具体例外和受保护元数据。命令的 `workspace_write=false` 会同时关闭显式授权的外部写入；工作目录仍在 workspace。临时运行目录由可信运行时单独创建并允许子进程使用；read-only 会话仍会保存聊天记录。

Linux 的 Shell/Python 使用 bubblewrap，默认断网、工作区只读。`network=true` 仍需网络审批及有效策略允许；系统需安装 `libseccomp2`，联网时仅开放 IP 通信并拦截 Unix/本机 IPC socket。配置了域名过滤时，任意命令仍不能联网，因为命令无法可靠声明全部目的地。

`workspace_write=true` 仍需写入审批。Linux 将普通项目文件复制到私有候选区，使用相同工作路径执行；原项目和已有受保护目录保持只读。只有退出码为 0、未超时且全部差异通过路径、保护目录、当前权限和宿主版本校验后，运行时才发布修改。执行失败、超时或取消不会发布候选修改；撤权会阻止发布。与 macOS 直接执行写入的语义不同，命令结束前其他进程看不到候选修改。

候选普通文件总量上限为 64 MiB；`.git`、`.agents`、`.codex`、`.aws`、`.venv`、`node_modules`、`.praxis` 和 `.agent_memory` 不参与可写复制。新建这些保护路径、修改符号链接或硬链接别名会被拒绝。发布逐文件使用原子替换，但多文件修改不是文件系统事务；并发宿主修改被检测后需检查项目并重试。

Linux 仍支持额外的只读资源。外部写入、全盘 read、已暴露目录内无法表达的 deny 等策略会返回 `sandbox_policy_unsupported`，不会退回未隔离执行。本地 stdio MCP 在 Linux 上继续只读、断网；本次联网能力面向 Shell/Python。

域名过滤目前落实在 HTTP/浏览器工具。任意 Shell/Python 命令无法可靠申报全部目的地，配置域名过滤后要求联网的命令会被拒绝；尚无面向任意子进程的域名代理。macOS 全盘 read 加子进程写入也会被拒绝，使用具体范围授权或文件工具写入。

标准 Agent 的本地 stdio MCP 在发现工具之前就进入只读文件沙箱；配置受信任只授权这份配置的执行，不解除资源约束。MCP 不支持本机文件写入；网络只有策略明确 allow 且没有域名过滤时才启用。恢复会话和冻结 Turn 的历史权限会在服务启动前应用。工具禁用规则在启动前/发现后应用；权限变化后旧服务不能继续调用，需要重新打开服务。失败会记录诊断，不会退回未隔离 MCP。

这些约束覆盖标准 CLI/Agent 内置工具及产品 MCP 启动。自行创建的工具 runner、直接调用底层 transport、宿主 hooks 和第三方服务仍由宿主负责落实其资源边界。模型供应商连接、凭据和 checkpoint 存储由可信运行时管理；工具网络策略不控制模型 API 连接。

远程 Streamable HTTP MCP 使用主机级网络规则，每个真实请求都检查策略和目的地；注册与认证见 [MCP 连接说明](mcp.md)。

`agent setup` 保存的个人模型凭据由可信运行时读取。POSIX 凭据目录对 Agent 文件工具和进程沙箱同时禁止读取及写入，显式子路径授权不能解除；该边界随权限快照保持有效。macOS 凭据使用 Keychain。首次配置见 [setup 说明](setup.md)。

## 验证

自动化用例覆盖路径优先级、策略上限、大小写/链接别名、策略源写保护、递归工具过滤、SDK/CLI、网络与审批组合、重定向、恢复、准备后撤权，以及实际 macOS Shell/Python/MCP 子进程。Linux CI 执行真实 bubblewrap Shell/Python 测试，验证允许/禁止联网、隔离本机 IPC、候选写入、保护目录、外部路径、失败/超时和权限撤销；其他平台没有未经验证的隔离承诺。
