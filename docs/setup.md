# 首次配置

在源码安装后运行：

```bash
uv run agent setup
uv run agent chat
```

向导分三步：选择模型，验证并保存模型连接，再按需连接只读 GitHub MCP。模型验证会发送少量实际模型请求，不会执行探测工具。模型服务需支持当前配置声明的流式输出、工具调用及结构化输出；本地服务需先由用户启动。

## 选择与验证模型

可选择已登记的模型 ID；输入 `new` 添加兼容 OpenAI 接口的新模型。添加时需要服务商给出的真实模型 ID、API 地址、上下文窗口 tokens 和 API Key 环境变量名。无认证服务可把变量名填为 `none`。

可直接指定 ID，跳过选择提示：

```bash
uv run agent setup --model openai/gpt-oss-120b --no-mcp
```

未登记的 ID 会进入自定义连接流程。向导复用 `ModelAdminService` 的完整 probe；验证失败不保存新 Key、不改变当前项目选择，自定义模型验证失败也不提交注册表。用户取消输入会退出。配置文件、模型绑定和聊天记录只保存模型定义与变量名，不保存 Key。

连接验证通过后才保存输入的新 Key。如果注册表写入成功而凭据保存发生系统错误，模型定义可能已存在；当前选择保持原值，错误会明确报告。重新运行向导可继续完成配置。

## 凭据保存与优先级

模型凭据依次从以下位置读取：

1. 指定名字的进程环境变量（包括已有 `.env` 加载逻辑）。
2. 为该变量名和准确 API 端点保存的个人凭据。

macOS 使用 Keychain 的 `praxis.model.*` 条目；其他 POSIX 系统使用个人设置目录下的 `credentials/models.json`，目录权限 `0700`，文件权限 `0600`。后者是权限保护的本地明文文件，不是加密保险库。Agent 的文件工具与进程沙箱禁止读取或修改凭据目录，保护随权限快照和管理员上限保持有效。

保存的 Key 绑定变量名、API 端点及个人设置目录；更换服务地址或切换个人设置目录不会自动把旧 Key 发给新服务。环境变量是宿主显式提供的凭据，优先于保存的值；若选择替换已有环境 Key，向导会提示在当前终端 `unset` 对应变量后再启动 Agent。向导无法修改调用它的父终端环境。

模型解析、能力探测、本地健康检查和冻结模型绑定使用同一凭据来源。本地健康检查仅对模型 API 的同源地址携带 Key。保存的 Key 轮换后，下一次解析创建新客户端；进行中的请求不会被撤销。

个人模型定义默认位于 `~/.config/praxis/models.yaml`，可通过 `PRAXIS_MODEL_REGISTRY_PATH` 指定；凭据和 MCP 个人设置默认位于 `~/.config/praxis`，可通过 `PRAXIS_CONFIG_DIR` 指定。模型选择默认写入当前项目的 `.praxis/model_session.json`，可用 `--session-path` 指定其他选择文件。

换项目后，可以指定已经登记的模型：

```bash
agent setup --model MODEL_ID --no-mcp
```

这会复用个人连接和凭据，再验证并保存该项目的选择。现有模型定义不会被重复注册或覆盖。需要更改模型参数时使用 `agent model update`。

## 可选 GitHub 连接

确认连接 GitHub 后，向导复用本机 GitHub CLI 登录；尚未登录时先执行：

```bash
gh auth login --hostname github.com
```

新连接沿用 `agent mcp add github` 的只读预设与域名授权。已有同名连接仅执行连接检查，不覆盖配置或重复授权。检查失败返回非零退出码并说明模型已配置、GitHub 尚未完成；模型配置仍可使用。其他服务通过 `agent mcp add` 登记，详见 [MCP 连接](mcp.md)。

向导尚不提供通用浏览器 OAuth、服务安装、图形化连接市场或 Windows 凭据安装器。
