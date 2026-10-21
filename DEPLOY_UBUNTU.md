# Ubuntu 安装与服务器使用

本文只介绍当前推荐的安装方式：普通用户运行 CLI，凭据保存在该用户的个人配置目录，通过 SSH 使用终端。项目定位与日常操作见 [README](README.md) 和[使用指南](docs/guide.md)。

## 环境准备

推荐 Ubuntu 24.04、Python 3.12，以及 Git、uv、Bubblewrap 和 libseccomp。

```bash
sudo apt-get update
sudo apt-get install -y git python3-venv bubblewrap libseccomp2
```

安装 uv 的方法见 [uv 官方文档](https://docs.astral.sh/uv/getting-started/installation/)。检查实际沙箱能否启动：

```bash
/usr/bin/bwrap --unshare-user --unshare-pid --ro-bind / / /bin/true
```

Ubuntu 的 AppArmor 可能限制非特权用户命名空间。只有确认这是阻断原因后，才由管理员审查并安装仓库提供的单程序策略：

```bash
sudo install -m 0644 scripts/ubuntu/praxis-bwrap.apparmor /etc/apparmor.d/praxis-bwrap
sudo apparmor_parser -r /etc/apparmor.d/praxis-bwrap
```

若系统已有 `/usr/bin/bwrap` 的策略，应调整现有策略，不叠加第二份。不要通过全局关闭 AppArmor 或放开所有命名空间来绕过检查。

## 安装与启动

以自己的普通账号克隆仓库、安装依赖并配置模型：

```bash
git clone https://github.com/xiaoyinglei/praxis-agent-runtime.git
cd praxis-agent-runtime
uv sync --locked --link-mode copy
uv run agent setup
uv run agent chat
```

模型 API Key 由使用者在隐藏输入中填写。Linux 将保存的凭据放在个人配置目录的私有文件中；账号配置、项目文件和聊天记录应分开管理，不把密钥放进 Git。

默认依赖包含数据与文档工具。RAG 和浏览器按需增加，MLX 不是 Linux 推理后端：

```bash
uv sync --locked --extra rag --link-mode copy
# 或安装需要的浏览器能力
uv sync --locked --extra browser --link-mode copy
uv run --extra browser playwright install --with-deps --only-shell chromium
```

如果同时需要两项能力，在同一次同步中传入两个 `--extra`；后续 uv 命令也保留所需 extras，避免同步时移除它们。

## 工作区与依赖

Agent 以命令执行时所在的目录作为工作区。源码安装目录可以与工作项目分开：

```bash
cd /path/to/project
uv run --project /path/to/praxis-agent-runtime agent chat
```

项目已在 `pyproject.toml` 中将 uv 的默认 `link-mode` 固定为 `copy`，普通 `uv sync`、`uv run` 和 `uv pip install` 的安装也使用独立文件，避免与外部缓存形成硬链接。
更改默认值不会改造已安装的旧文件。旧环境若触发 `workspace_hardlink_detected`，先核对错误路径，再在安装目录重装依赖；后续更新沿用项目默认值：

```bash
uv sync --locked --reinstall --link-mode copy
```

不要从 macOS 复制 `.venv` 到 Linux。若项目自身的依赖也在工作区内，同样应使用复制模式，或将执行环境放到工作区之外；不要删除实际的沙箱边界检查。

## Linux 执行边界

Shell 与 Python 默认在只读、禁网络的 Bubblewrap 环境中运行，只有私有临时区可写。普通文件工具的权限与子进程权限分别检查。

- 明确授权 `network=true` 后，可以使用出站 IPv4/IPv6。libseccomp 仍阻断主机 Unix IPC、监听和 io_uring；按域名限制的网络策略不能直接授予任意命令联网。
- 明确授权 `workspace_write=true` 后，子进程在私有候选工作区中运行。只有成功退出且完整变更通过当前权限、受保护路径和并发版本检查后，才发布到原工作区。
- 失败、超时和取消会丢弃候选写入。普通输入输出文件内容上限为 64 MiB；元数据、依赖和 Agent 状态保持保护。
- 单个文件替换是原子的，多文件发布不是整个文件系统的事务。子进程向外部目录写入暂不支持。

缺少 Bubblewrap、libseccomp 或无法表达当前策略时，工具拒绝执行，不退回无隔离进程。本地 stdio MCP 在 Linux 下保持只读、禁网络。完整权限示例见[使用指南](docs/guide.md#权限与跨目录资料)。

## 通过 SSH 使用

交互聊天需要保留 TTY：

```bash
ssh -t USER@SERVER
cd /path/to/praxis-agent-runtime
uv run agent chat
```

连接管理页面只监听服务器本机。需要从自己的电脑登录服务时，先建立 SSH 端口转发：

```bash
ssh -L 18765:127.0.0.1:18765 USER@SERVER
```

随后在该远程终端中启动：

```bash
cd /path/to/praxis-agent-runtime
uv run agent connections --port 18765 --no-open
```

在自己的电脑浏览器中打开命令输出的本机地址。管理地址带有本次会话权限，不要分享，也不要将管理端口暴露到公网。SSH 中的聊天 `/connections` 同样不会把页面自动转发到你的电脑，远程场景使用上述明确的管理命令。

## 更新已有安装

先检查工作区，保存自己的修改，再安全快进；不要用 reset 或 clean 覆盖项目资料：

```bash
git status --short --branch
git fetch --prune origin
git switch main
git merge --ff-only origin/main
uv sync --locked --link-mode copy
uv run agent --help
```

需要额外能力时，在同步命令中保留已选择的 extras。更新源码不要求覆盖个人模型配置、API Key 或聊天数据库。正在运行的进程继续使用它已加载的代码；完成当前任务后重新启动聊天，新的进程才使用更新版本。

已有 `praxis` 快捷启动器的机器，可以继续用该入口；它属于本机部署配置，不是新安装的前置条件。同步时先确认启动器引用的源码和 Python 环境，不复制其他用户的凭据，不重启未结束的聊天。
