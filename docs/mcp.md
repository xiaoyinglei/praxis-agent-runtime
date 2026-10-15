# MCP 连接

Praxis 支持本地 stdio 和远程 Streamable HTTP MCP。连接注册、认证和网络授权由用户控制；服务提供的工具沿用现有 ToolExecutor 的参数验证、权限决策、审批和记录。

## 本机连接目录

```bash
uv run agent connections
```

页面内置 GitHub、Notion、Linear，并提供自定义 HTTPS MCP 地址表单。选择服务后完成授权，页面会检查连接及可用工具；也可重新授权或移除。默认只读；自定义连接需要动作时，可明确勾选允许服务动作，工具调用仍受权限及审批控制。管理页面只监听 `127.0.0.1`，使用每次启动的随机会话地址；不要分享该地址。`--no-open` 只显示地址，Ctrl+C 关闭管理器并取消等待中的 OAuth 回调。

## GitHub：一次注册

```bash
uv run agent mcp add github
uv run agent mcp check github
uv run agent chat
```

`add github` 使用这台电脑已经登录的 GitHub CLI；如果尚未登录，先执行 `gh auth login --hostname github.com`。该命令注册官方 `https://api.githubcopilot.com/mcp/readonly`，只提供仓库搜索、文件读取、Issue 和 PR 查询。Token 复制到 macOS Keychain 的 `praxis.mcp.github` 项，Linux 则使用个人凭据目录中权限为 `0600` 的文件；凭据绑定准确的服务端点，项目配置和聊天输出中不包含 Token。

个人连接保存到 `~/.config/praxis/mcp_servers.yaml`，优先于工作区示例配置。注册会给个人权限配置添加该服务主机的 allow 规则，其他主机保留原决策；现有 deny、域名允许列表、管理员上限都不能被注册绕过。更新采用配置目录锁和原子文件替换；失败会回滚权限文件及本次凭据变更。

注册后新开 Agent 会话。旧会话的权限快照不会自动扩大；需要在旧聊天执行 `/permissions reload` 并退出，再重新打开聊天。冻结 Turn 继续受原来的权限上限限制。

GitHub CLI 切换账号或更新授权后，可重新认证：

```bash
uv run agent mcp login github --from-gh
```

若改为手工 Token，在本机终端隐藏输入：

```bash
uv run agent mcp login github
```

不要把 Token 发到聊天。更新凭据后重新打开 Agent 会话。

## 浏览器 OAuth

支持 MCP 标准的 Authorization Code + PKCE、授权服务发现、动态客户端注册和刷新令牌。兼容服务通常无需自己申请应用：

```bash
agent mcp add notion --url https://mcp.notion.com/mcp --oauth
agent mcp login notion
agent mcp check notion
```

`login` 在本机启动回调监听并打开服务商授权页，校验 state、回调路径及主机；明确重新登录会重新走授权流程，失败保留此前有效凭据。正常 Agent 会话只静默刷新，不会在任务中弹出登录窗口；需要重新登录时返回诊断。

服务不支持动态注册时，可通过 `--oauth-client-id` 指定预登记的公开客户端。若服务商要求固定回调端口，注册时加 `--oauth-callback-port PORT`，或登录时加 `--callback-port PORT`，在服务商登记 `http://127.0.0.1:PORT/callback`。不支持将客户端密钥放入连接配置。

发现的认证地址默认只能使用 MCP 服务域名；委托其他域名时，需用户显式提供 `--oauth-domain auth.example.com`。这只授权认证通信，不扩大模型工具的网络权限；当前策略的 deny 和管理员上限仍然生效。OAuth 请求拒绝非公网地址与 HTTP 重定向，不继承环境代理。服务需支持 HTTPS、S256 PKCE 和 Streamable HTTP。

macOS OAuth 凭据使用 Keychain；Linux/POSIX 使用个人 `credentials` 目录（`0700`）中的私有文件（`0600`），是受文件权限保护的本地明文存储。令牌、刷新令牌和客户端注册信息绑定准确端点，Agent 文件工具及沙箱不能访问凭据目录，连接 YAML 和图形页面不返回令牌。

移除本机连接及其凭据：

```bash
agent mcp remove notion
```

移除不撤销服务商侧账号授权，也不删除可能被其他连接复用的网络规则；服务商授权应在对应账号设置中撤销。

## 其他远程服务：只登记配置

```bash
uv run agent mcp add docs --url https://mcp.example.com/mcp \
  --bearer-token-env-var DOCS_TOKEN --tool search
uv run agent mcp list
uv run agent mcp check docs
```

`DOCS_TOKEN` 由启动 Agent 的宿主环境提供；不会保存到连接 YAML。`list` 不读取凭据、不连接远程服务。`check` 仅连接指定服务、认证并发现工具，不调用模型、不修改远程数据。

CLI 默认 `--read-only`：只安装服务声明为只读的工具。若服务没有提供只读声明，检查可能返回没有可用工具；可信服务确需执行动作时，可在注册时显式指定 `--allow-actions` 和相应 `--tool`。远程动作仍须审批，工具 allow 规则不能跳过该审批；`profile=read-only` 的权限上限会拒绝此类动作。

服务的只读声明是元数据。GitHub 预设还使用服务端强制的只读 URL；一般远程服务的实际访问范围仍由服务端和凭据权限决定。

通用 YAML 示例：

```yaml
servers:
  - name: docs
    transport: http
    url: https://mcp.example.com/mcp
    bearer_token_env_var: DOCS_TOKEN
    headers:
      X-Service-Option: example
    read_only: true
    tools_allowlist: [search]
    enabled: true
```

SDK 标准 `Agent` 会自动发现个人连接。显式工作区／外部 MCP 配置仍须宿主通过 `decide_mcp_config_trust` 授权其准确内容；通过 `AGENT_MCP_CONFIG` 可选择配置路径。MCP 配置与工具结果无法给自己增加权限。

`PRAXIS_CONFIG_DIR` 可指定独立的个人设置目录，MCP 与个人权限配置共同使用该目录。目录必须由当前用户所有，且组和其他用户不可写。标准运行时禁止模型工具写入该目录。Keychain 登录面向 macOS；Linux 使用上述 POSIX 私有凭据文件。注册锁使用 POSIX 文件锁，尚无完整 Windows 凭据安装器。

## 连接边界

远程 MCP 必须是公网 HTTPS 地址，不支持 URL 内凭据、query、fragment。连接前要求有效网络策略对该目的地为 allow；ask 不会被后台发现流程静默授权。每个实际 HTTP 请求（初始化、工具调用、SSE、关闭会话）都复查当前网络策略、配置版本和固定目的地；Host 必须与 URL 的 authority 一致。DNS 全部地址须为公网，实际连接固定到已验证的 IP。客户端不继承环境代理，不跟随重定向。

Token 的变化会改变重新连接后的工具执行版本，旧审批不能沿用。修改连接配置或收紧运行时权限后，旧连接停止接受新调用，需要重新打开会话。已经发送的请求不会被撤销或回滚。

本机文件沙箱只读与 GitHub 服务只读是两个范围。本地 stdio 服务继续在 OS 沙箱运行；远程服务不在本机运行，无法用本机文件沙箱约束其远程副作用。

CLI 和本机连接页面共享注册、授权、检查及删除服务。新增使用相同 MCP HTTP/OAuth/Token 协议的服务，只需填写连接信息，无需修改运行时代码。目录是随版本发布的服务预设，不会下载执行任意插件代码。

参考：[GitHub 官方远程服务配置](https://github.com/github/github-mcp-server/blob/main/docs/remote-server.md)、[Notion MCP 客户端指南](https://developers.notion.com/guides/mcp/build-mcp-client)、[Linear MCP](https://linear.app/docs/mcp)。
