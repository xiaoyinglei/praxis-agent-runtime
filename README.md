# Praxis

**让 AI 在你的项目里完成工作，而不止于回答问题。**

[![CI](https://github.com/xiaoyinglei/praxis-agent-runtime/actions/workflows/ci.yml/badge.svg)](https://github.com/xiaoyinglei/praxis-agent-runtime/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.12%20%7C%203.13-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Praxis 是面向个人本机和可信服务器工作区的开源 AI Agent。它可以阅读项目、修改代码、处理数据与文档、查询网页和连接外部服务，并将工具执行、权限审批和聊天历史保存在同一套运行时中。

以 `agent` 命令行为主要入口，也提供 `agent_runtime.Agent` Python SDK。你选择模型和项目，Praxis 负责把请求推进为可检查的行动与结果。

[快速开始](#快速开始) · [使用指南](docs/guide.md) · [开发指南](docs/development.md) · [Ubuntu 部署](DEPLOY_UBUNTU.md)

## 为什么做 Praxis

把模型接进项目后，真正困难的是后面的工作：它能读取哪些文件，哪些操作需要确认，中断后怎样继续，以及“已经完成”究竟有什么依据。

Praxis 把这些事情放进执行流程。模型根据观察规划下一步；文件、命令、网页和集成服务通过明确的工具接口使用；需要授权的动作在执行前暂停。运行结束与任务验收分别记录；需要机器验收时，由用户指定验收命令，Runtime 检查其真实执行结果。

这是一套服务真实任务的运行时，也是一份可以阅读、调试和扩展的 Agent 工程实现。

![Praxis 执行过程演示](docs/assets/praxis-demo.gif)

*100 元打八折却算成了 -1900 元：读取代码 → 复现失败 → 修复公式 → 三项测试通过。演示使用固定模型响应，工具、文件修改与测试均真实执行；画面为执行过程回放，不代表真实模型的任务能力。可通过 `scripts/render_praxis_demo.py` 重新生成。*

## 快速开始

### 一条命令安装

不需要预先安装 Python、uv 或 pip，也不需要创建、激活虚拟环境。支持 macOS 和 glibc Linux，自动选择 arm64 或 x86_64 发行包：

```sh
curl -fsSL https://raw.githubusercontent.com/xiaoyinglei/praxis-agent-runtime/main/install.sh | sh
```

安装器下载自带 Python 和依赖的[独立发行包](https://github.com/xiaoyinglei/praxis-agent-runtime/releases/latest)，校验 SHA-256 后完成安装。程序位于 `~/.local/share/praxis`，启动命令位于 `~/.local/bin/agent`。需要时会配置 shell 的命令路径；重新打开终端后可直接使用 `agent`，安装结束也会显示当前终端可立即使用的完整命令。

Linux 命令执行需要 Bubblewrap 和 libseccomp 系统后端，见 [Ubuntu 部署](DEPLOY_UBUNTU.md)。

### 首次配置与聊天

进入你要处理的项目，接入模型并开始聊天：

```sh
cd /path/to/my-project
agent setup
agent chat
```

首次 `setup` 默认接入 **DeepSeek**，按提示隐藏输入你自己的 API Key。连接检查会产生少量真实模型用量，可跳过 GitHub 等可选服务。其他云端模型和本地模型通过 `agent model add` 接入，再用 `agent model switch MODEL_ID` 切换，见[接入模型](docs/guide.md#接入模型)。MLX 和本地模型权重不会随普通安装下载。

macOS 将 Key 保存到系统 Keychain，Linux 使用个人配置目录中的私有凭据文件。个人模型配置默认位于 `~/.config/praxis`，聊天记录保存在各项目的 `.praxis/` 中。

### 更新与卸载

更新时重新执行上面的安装命令即可。安装器保留旧版本，下载或解包失败时不会替换现有启动入口；通过其他方式安装的同名 `agent` 命令也不会被覆盖。

使用默认路径独立安装后，可以这样卸载程序：

```sh
rm -f "$HOME/.local/bin/agent"
rm -rf "$HOME/.local/share/praxis"
```

个人配置、已保存凭据和项目聊天记录保留。如使用了自定义安装路径，请删除对应的程序目录和启动入口。完整安装说明见[使用指南](docs/guide.md#安装与首次配置)。

<details>
<summary>已有 Python 环境或参与源码开发</summary>

[`praxis-agent`](https://pypi.org/project/praxis-agent/) 同时提供 CLI 和 Python SDK。已有 Python **3.12 或 3.13** 环境时，可使用自己的环境管理方式安装，例如：

```bash
python -m pip install praxis-agent
```

已有 uv 时，也可以使用独立工具环境安装：

```bash
uv tool install --python 3.12 --link-mode copy praxis-agent
```

源码安装和参与本仓库开发时，使用 Git 和 uv：

```bash
git clone https://github.com/xiaoyinglei/praxis-agent-runtime.git
cd praxis-agent-runtime
uv sync --locked --link-mode copy

uv run agent setup
uv run agent chat
```

</details>

## 用它完成什么

| 场景 | 你可以提出的任务 | 执行方式 |
| --- | --- | --- |
| 理解与修改项目 | “梳理这个项目的入口，修复超时处理并验证。” | 阅读与搜索文件、补丁、命令执行、变更检查 |
| 数据与文档 | “整理这份表格，生成统计图并检查输出文件。” | 结构检查、受限 Python、文件生成与结果回读 |
| 公网检索 | “查阅官方文档，比较这两个方案并给出来源。” | 搜索、网页读取、可选 Chromium 渲染、来源快照 |
| 外部服务 | “查看这个仓库的 Issue，或查找 Notion 中的文档。” | GitHub、Notion、Linear 与自定义远程 MCP |
| 长期项目 | “继续上次的工作，沿用我确认过的项目约定。” | 聊天恢复、可选工作区记忆、可选私有知识库 RAG |

工作区以启动命令时所在的目录为准。处理其他项目时，进入对应目录后直接启动：

```bash
cd /path/to/my-project
agent chat
```

在聊天中直接描述目标即可。工具记录可以展开查看，写入和执行授权会单独呈现。

本地附件默认读取原文件，不复制到工作区。把文件拖入终端并按 Enter，输入区中的独立路径行会被识别为待发送附件；再输入分析任务即可。支持带空格的引号/转义路径、多个文件和本机 `file://` 地址；混合说明文字时把路径放在单独一行，或先使用 `/attach "/绝对路径/客户 资料.xlsx"`。`/attach` 查看待发送附件，`/detach` 清除待发送附件。不要把文件复制到项目里才能使用；SDK 的 `files=[...]` 和一次性命令 `--file` 也使用原地引用。

所选外部文件只授予读取权限，父目录和其他文件不会被开放；显式拒绝与管理员权限上限仍然生效。附件引用会随聊天保存，原文件之后修改、移动或删除会影响后续读取；旧版本已经保存的附件副本保持兼容，不会自动删除历史数据。

也可以通过 `agent run` 执行一次性任务。下面的验收命令以本仓库为例，可替换为目标项目自己的检查命令：

```bash
# 只做分析，不要求修改工作区
agent run "阅读 pyproject.toml，解释项目入口和模块职责。" \
  --no-require-workspace-change

# 修改类任务：要求实际变更，并运行用户指定的验收
agent run "修复配置加载中的异常处理，更新相关测试并验证。" \
  --require-workspace-change --acceptance-command "uv run pytest -q"
```

常用聊天命令集中在 `/help` 中：

| 命令 | 用途 |
| --- | --- |
| `/model` | 选择模型，下一条消息生效并保留历史 |
| `/permissions` | 选择请求批准、帮我批准或完全访问；`grants` 查看会话授权，`revoke` 撤销 |
| `/connections` | 打开本机服务连接管理 |
| `/resume` | 选择并打开已保存的聊天 |
| `/new` | 开始空上下文的新聊天 |
| `/copy` | 复制最近回答的原始 Markdown |

## 过程可见，权限可控

Praxis 以 Thread 保存聊天，以 Turn 表示一次请求，以 Step 推进模型与工具交互。执行事件、工具结果和恢复状态写入持久记录，界面只是这些记录的呈现。

```text
用户请求 → 模型规划与观察 → 工具执行 → 结果检查与回复
                  ↑           │
                  └─ 工具结果 ─┘
                       │
               权限检查 · 人工审批 · 持久记录
```

文件读取、工作区写入、命令执行和网络访问分别约束。可以只允许读取某个外部资料目录，也可以选择只读模式；个人授权仍受管理员策略上限限制。

新聊天默认“帮我批准”，自动处理普通工作区编辑、只读沙箱命令和公开网页读取。输入 `/permissions` 可在面板里选择审批模式；需要确认时直接选择“允许一次 / 本会话允许同类命令 / 拒绝”，支持方向键和鼠标。临时授权可以查看、撤销，重新打开聊天后失效。详见[审批方式](docs/guide.md#审批方式)。

- **命令与 Python：**macOS 使用 Seatbelt，Linux 使用 Bubblewrap。默认只读、禁网络；额外的联网或工作区写入需要授权。Linux 写入先进入私有候选工作区，成功并通过检查后才发布变更。
- **外部服务：**通过 `agent connections` 管理账号。个人远程 MCP 在需要调用时连接；缺少认证或工具目录时才显示连接提示，点击后由你选择登录，不会在普通聊天中自动弹授权页。
- **聊天与记忆：**`agent resume` 打开历史聊天；可选记忆按工作区隔离、默认关闭，来源与修改可检查。记忆生成与 RAG 是独立能力。
- **结果与验证：**运行结束不等于答案已经证实。`AgentResult.validation_status` 区分未评估、通过和失败，`validation_scope` 说明实际检查范围；只读分析可以直接回答，不制造修改证据。

操作方法和配置示例见[使用指南](docs/guide.md)。

## Python SDK

CLI 与 SDK 使用同一套运行时。在已有模型配置的项目中，可以直接嵌入你的 Python 程序：

```python
import asyncio
from pathlib import Path

from agent_runtime import Agent


async def main() -> None:
    agent = Agent(workspace_path=Path.cwd())
    result = await agent.run(
        "阅读当前项目，解释主要模块的职责。",
        require_workspace_change=False,
    )
    print(result.answer)
    print(result.validation_status)


asyncio.run(main())
```

多轮聊天使用 `async with agent.session()` 和 `session.submit()`；实时事件使用 `agent.stream()`。恢复一个暂停的 Turn 使用 `agent.resume()`，与打开历史聊天的 CLI `agent resume` 语义不同。[查看 SDK 与扩展方式](docs/guide.md#python-sdk-与扩展)。

## 适用范围与项目状态

Praxis 适合个人本机开发、可信服务器上的项目工作，以及 Agent 运行时的学习与二次开发。当前版本为 `0.1.1`，接口仍在演进，尚不提供多租户远程执行服务，也不承诺隔离任意恶意仓库或不可信操作者。

macOS 沙箱依赖 `/usr/bin/sandbox-exec`，Linux 依赖 `/usr/bin/bwrap`；需要命令联网时还依赖 libseccomp。后端缺失或无法表达当前策略时会拒绝执行，不降级为无约束进程。Linux 子进程的外部目录写入暂不支持，Windows 尚未完成支持。

CI 覆盖类型、导入边界、回归测试、沙箱和浏览器检查、安装包及公共入口。测试桩只验证测试流程，不能代替真实沙箱验证；历史评估不代表当前版本或任意任务的模型质量。任务效果还取决于模型、权限和输入资料。

## 文档与参与开发

| 文档 | 内容 |
| --- | --- |
| [使用指南](docs/guide.md) | 安装、模型、聊天、连接、权限、记忆、检索与 SDK |
| [开发指南](docs/development.md) | 代码结构、设计原则、检查命令与贡献流程 |
| [Ubuntu 部署](DEPLOY_UBUNTU.md) | Linux 沙箱、依赖安装和运行环境 |

```bash
uv run ruff check .
uv run mypy
uv run lint-imports
uv run pytest -q
```

欢迎通过 Issue 描述问题或提交 PR。问题报告请附平台、Python 版本、启动命令和脱敏错误信息；运行与恢复日志也应先检查是否含有个人资料。

采用 [MIT](LICENSE) 许可证。
