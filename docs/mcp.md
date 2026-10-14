# MCP 连接

Praxis 支持本地 stdio 和远程 Streamable HTTP MCP。连接注册、认证和网络授权由用户控制；服务提供的工具沿用现有 ToolExecutor 的参数验证、权限决策、审批和记录。

## GitHub：一次注册

```bash
uv run agent mcp add github
uv run agent mcp check github
uv run agent chat
```

`add github` 使用这台电脑已经登录的 GitHub CLI；如果尚未登录，先执行 `gh auth login --hostname github.com`。该命令注册官方 `https://api.githubcopilot.com/mcp/readonly`，只提供仓库搜索、文件读取、Issue 和 PR 查询。Token 复制到 macOS Keychain 的 `praxis.mcp.github` 项，并绑定准确的服务端点；项目配置和聊天输出中不包含 Token。

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

`PRAXIS_CONFIG_DIR` 可指定独立的个人设置目录，MCP 与个人权限配置共同使用该目录。目录必须由当前用户所有，且组和其他用户不可写。标准运行时禁止模型工具写入该目录。Keychain 登录面向 macOS；注册锁使用 POSIX 文件锁。其他平台可使用环境变量认证及宿主提供的配置，尚无完整跨平台的凭据安装器。

## 连接边界

远程 MCP 必须是公网 HTTPS 地址，不支持 URL 内凭据、query、fragment。连接前要求有效网络策略对该目的地为 allow；ask 不会被后台发现流程静默授权。每个实际 HTTP 请求（初始化、工具调用、SSE、关闭会话）都复查当前网络策略、配置版本和固定目的地；Host 必须与 URL 的 authority 一致。DNS 全部地址须为公网，实际连接固定到已验证的 IP。客户端不继承环境代理，不跟随重定向。

Token 的变化会改变重新连接后的工具执行版本，旧审批不能沿用。修改连接配置或收紧运行时权限后，旧连接停止接受新调用，需要重新打开会话。已经发送的请求不会被撤销或回滚。

本机文件沙箱只读与 GitHub 服务只读是两个范围。本地 stdio 服务继续在 OS 沙箱运行；远程服务不在本机运行，无法用本机文件沙箱约束其远程副作用。

目前提供 CLI 注册／检查、环境 Token 和 macOS Keychain 认证；尚未实现图形化连接市场或通用浏览器 OAuth 注册流程。新增使用相同 HTTP／Token 方式的服务无需修改运行时代码。

参考：[GitHub 官方远程服务配置](https://github.com/github/github-mcp-server/blob/main/docs/remote-server.md)。
