# Ubuntu 上的独立模型调用服务

目标：服务器上的 Agent 在 DeepSeek 和 Groq 之间切换，不依赖 Mac、SSH 反向隧道或本地网关。供应商密钥由独立系统用户 `praxis-model` 持有，Agent 使用 `praxis-agent` 用户和单独代理凭证。该入口通过 SSH 使用 CLI，不提供公共 Agent HTTP API。

## 一次安装

先把本次代码及现有 Linux 沙箱改动部署到 Ubuntu 的受审查目录，例如 `/home/admin/praxis`。新入口使用端口 18444，避免与可能存在的旧隧道端口 18443 冲突。

需要 Ubuntu 24.04、系统 Python 3.12、`python3-venv`、systemd、Bubblewrap，以及现有项目的 Linux `.venv`。先按 `DEPLOY_UBUNTU.md` 验证真实 bwrap 可以启动。Agent 环境可以使用既有 core profile；服务本身另建最小环境，依赖版本和 wheel SHA-256 来自 `uv.lock`，安装时需要访问 PyPI。

```bash
sudo apt-get install python3-venv
cd /home/admin/praxis
sudo bash scripts/ubuntu/install-model-service.sh /home/admin/praxis
```


若 `/home/admin/praxis` 有未提交改动，保留它，使用新的部署目录，避免 `git pull` 覆盖已有工作：

```bash
git clone https://github.com/xiaoyinglei/praxis-agent-runtime.git /home/admin/praxis-server
cd /home/admin/praxis-server
```

新目录需要 Linux Python 3.12 Agent `.venv`。可以复用已验证、与当前 `uv.lock` 一致的同机环境：

```bash
cp -a /home/admin/praxis/.venv /home/admin/praxis-server/.venv
sudo bash scripts/ubuntu/install-model-service.sh /home/admin/praxis-server
```

安装器会将 Agent Python 配置重定位到系统解释器，并使用新目录的受保护源码。若没有可复用环境，按 `DEPLOY_UBUNTU.md` 的 core profile 安装。不要拷贝 Mac 的 `.venv` 到 Linux。

安装器为首次安装设计：遇到已有 `/opt/praxis`、凭证目录、同名用户/组、unit 或 launcher 就拒绝覆盖。它复制受审查源码和 Agent 虚拟环境到 root 所有的 `/opt/praxis`，独立创建密钥服务虚拟环境，拒绝 editable 导入钩子和 site-packages 符号链接。密钥服务用系统 Python 的隔离模式执行独立脚本，不加载 Agent 包、工作目录或用户 site-packages。

安装器在服务器终端隐藏提示输入 DeepSeek、Groq API key。这是一次明确的密钥托管变更：密钥将保存在该服务器。不要把值发到聊天、写进 shell 命令参数或打开 shell trace；没有自动读取或传输 Mac `.env` 的行为。两把密钥输入并验证格式后才写文件；真实账户权限还需要下述 completion probe。

供应商密钥和服务侧代理凭证在 `/etc/praxis-model`，目录 0700、文件 0400、root 所有。systemd 将它们复制给 `praxis-model`。Agent 只能读 `/var/lib/praxis-agent/proxy.token`；工作目录是 `/var/lib/praxis-agent/workspace`。密钥服务的私有状态在 `/var/lib/praxis-model`。两个用户均不加入 sudo 或管理员组。

## 验证及使用

```bash
sudo systemctl status praxis-model --no-pager
sudo ss -ltnp 'sport = :18444'
sudo -iu praxis-agent praxis-agent model list --source
sudo -iu praxis-agent praxis-agent model probe deepseek-flash --level full
sudo -iu praxis-agent praxis-agent model probe openai/gpt-oss-120b --level full
sudo -iu praxis-agent praxis-agent run "只回复：连接成功" --model deepseek-flash --no-require-workspace-change
sudo -iu praxis-agent praxis-agent chat --model deepseek-flash
```

`model list` 证明目录能加载；代理 `/v1/models` 只返回已启用路由，不访问供应商，因此不是账户鉴权证据。实际 completion / Agent turn 才能验证供应商账户和模型访问权；会产生实际用量。DeepSeek thinking 模式在短 probe 的输出预算内可能不返回文本，遇到此情形使用完整 Agent turn 检查。

进入 chat 后：

```text
/model
/model openai/gpt-oss-120b
/model deepseek-flash
```

沿用现有控制面：切换影响下一轮，保留对话历史；恢复暂停任务使用任务原来的模型绑定。错误别名不会改选项。不同模型上下文容量和工具支持仍需遵守运行时检查。

部署代码与工作目录分开。需要编辑自己的项目时，由管理员把项目复制/检出到该用户可写的工作目录；不要把供应商密钥、服务源码或配置放进 Agent 的工作区。当前 Linux shell/Python 沙箱仍只允许只读、禁网络执行；修改工作区使用现有文件工具，不能把本次模型接入当作 Linux shell 写权限支持。


## 终端交互界面

`agent chat` 已有 Markdown 会话、模型/状态栏、工具记录折叠和批准输入。它是终端界面，目前没有浏览器聊天窗口、项目列表或图形化密钥设置页。`agent run` 是单任务命令；非 TTY 或 `TERM=dumb` 会降级为纯文本。

在真实终端保留 SSH TTY：

```bash
ssh -t praxis
sudo -iu praxis-agent env TERM=xterm-256color praxis-agent chat
```

也可以安装完成后从 Mac 一条命令进入：

```bash
ssh -t praxis 'sudo -iu praxis-agent env TERM=xterm-256color praxis-agent chat'
```

在 chat 输入 `/model` 打开模型选择，`/help` 查看交互键；模型计算在云端，工具执行和工作区在服务器。新密钥服务当前会缓冲完整响应后再显示，所以等待期间没有实时 token 输出。它尚不提供网页 UI；若要浏览器操作，应另行设计带身份认证的 Agent 会话入口，而不是把密钥服务的 loopback 端口开放到公网。

## 使用限制和故障

默认并发 2、单请求 120 秒、请求体 256 KiB、响应 8 MiB、输出最多 4096 tokens。每日 UTC 窗口最多 200 个已接纳 completion 请求，且预留输出总量最多 200000 tokens。每次转发前将请求的最大输出 token 数写入 SQLite；失败、超时和没有用完的输出预留也不退款。小于已记录日期的系统时间会拒绝接纳，避免时钟回退重开额度。无效请求不访问上游，并发/体积/时间限制仍生效。

重启服务、轮换代理凭证不会重置当天额度；删除状态文件会破坏这一保证，因此 Agent 没有状态目录权限。若管理员需要改额度，用 `systemctl edit praxis-model` 覆盖 ExecStart，并保留独立用户、凭证、固定监听地址和私有 StateDirectory；不要通过 Agent 工作区配置控制网关路由。

这里限制的是调用量和输出上限，不承诺固定人民币金额。输入用量、推理 token 计费和供应商规则不同；如需金额硬上限，还需供应商账户支持的限制。已有 Agent OpenAI SDK 可能重试临时错误，每个实际转发的重试独立消耗额度，服务自身不重试，也不静默切换供应商。

JSON/SSE 响应都先在限额内缓冲并检查凭证回显，再交给 Agent；因此首段输出会等到模型完成，不是实时逐 token 显示。支持文本、工具调用及工具结果；多模态内容和未知顶层参数会被拒绝。上游重定向、错误内容/headers、异常文本不会透传。字面、JSON 转义及 SSE 分片回显检查只是额外防护，不能对抗故意编码秘密的恶意供应商。

## 轮换与停止

```bash
sudo /usr/bin/python3 -I /opt/praxis/provision-model-secrets.py --rotate-token
sudo systemctl restart praxis-model
sudo -iu praxis-agent praxis-agent chat
```

轮换后重启服务、重新启动已有 Agent 进程；旧进程持有的 token 将失效。两个 token 文件更新过程中或重启之前可能暂时拒绝调用，预算保持不变。供应商 key 需要由管理员在供应商平台撤销/更新，再在服务器隐藏输入并替换对应 root-only 文件，不通过 Agent 文件工具操作。

回退时先 `sudo systemctl disable --now praxis-model`；如有本地保留的旧 Mac 网关，可自行切回旧启动方式。完整卸载由管理员确认不再需要工作区及预算证据后，移除 unit、`/usr/local/bin/praxis-agent`、`/opt/praxis`、凭证和两个系统用户；不要自动删除用户工作区或用量证据。部分安装失败时保留私有文件，确认失败位置后再处理，安装器不会自动覆盖或删除它们。

## 安全边界和验证证据

普通 Agent/tool 进程无法读取供应商密钥，但被盗代理 token 仍有剩余额度的消费权，也能把数据发给允许的供应商。localhost 不是身份认证；每个 API 请求仍需要 token。程序固定 HTTPS 上游，不接收调用者指定的 URL、密钥或代理设置。

同机 root 或 `praxis-model` 进程被攻破仍可能获取供应商密钥；数据也会送到所选供应商。需要抵抗整台 Agent 主机失陷时，应把密钥服务放在另一台受控机器，并增加经过认证的 TLS 连接，不能仅靠同机加密或容器承诺这一点。

自动测试只用假密钥/假供应商，验证模型路由、SDK JSON/SSE/工具调用、配额并发与重启、响应泄漏防护、独立导入路径、凭证文件权限和原有模型切换。真实供应商密钥不会自动迁移；实际安装与真实 completion probe 的结果须另行记录，不能用这些测试代替。

### 2026-10-03 本次验证记录

- 本地 Agent / model-runtime 回归：1672 passed、45 skipped。新网关/部署测试包含在该次运行中。
- 新服务及 Python 部署脚本：ruff、mypy 通过；5 条 import-linter 边界全部保留。
- 只用十个 SHA-256 锁定 wheel 的独立虚拟环境：安装、import-hook 检查和独立服务 CLI 加载通过。
- `ssh praxis` 现有 Linux 沙箱测试：18 passed；该运行没有上传新源码或凭证。
- 独立源码审查已修复 dotenv、getpass 回退、初始状态目录复用、uv venv 格式及重定位问题。
- 新服务尚未部署：自动审批拒绝向 `praxis` 发送项目源码/依赖，要求明确授权 payload 和目的地。没有传输项目归档或 wheel，也没有迁移供应商密钥。`smoke-model-service.py` 的真实 systemd/独立 UID/重启配额验证和真实供应商 completion 仍待执行。
