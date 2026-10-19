# Praxis 开发指南

Praxis 的公开入口以 CLI 为先，Python SDK 与其共享能力和执行语义。代码修改应同时检查这两个入口，避免把运行时行为藏在某个界面或模型适配器中。

## 代码结构

| 路径 | 职责 |
| --- | --- |
| `agent_runtime/cli.py` | 命令、交互会话和结果展示的公共入口 |
| `agent_runtime/agent.py` | Python `Agent` 公共接口 |
| `agent_runtime/harness/` | Session、Turn、Step、模型事务、上下文、工具编排和持久记录 |
| `agent_runtime/tools/` | 工具契约、参数与输出验证、权限和具体工具 |
| `agent_runtime/runtime/` | 运行资源组装、模型目录、MCP 传输与工具目录 |
| `agent_runtime/terminal_*.py` | 终端输入、Markdown、工具记录和审批界面 |
| `rag/` | 可选私有知识索引、检索与服务适配 |
| `tests/` | 公共入口、故障恢复、隔离边界与协议回归 |
| `scripts/ubuntu/` | Linux 安装、沙箱和服务器启动脚本 |

`Session` 拥有一个聊天的资源，提交请求创建 `Turn`；模型根据观察选择工具，工具调用进入统一执行器。持久事件是历史与恢复的依据。终端界面和 CLI 文本输出呈现同一批执行事实。

工具目录决定可见能力，权限决定能否执行，人工审批决定是否接受当前确认请求；这三个概念不能互相替代。MCP 只是工具来源，RAG 是可选知识提供方，子智能体也应沿用预算、权限和持久记录。

## 设计原则

保持代理程序设计简洁，用明确、完整的工具契约提高 ACI 质量；必要时直接使用基础组件，减少没有独立职责的抽象层。展示真实计划和执行状态，不把模型的一句“完成”当作验证结果。

修复问题时先定位请求、上下文、工具结果、权限与恢复边界，再增加回归验证。不围绕某个网站、任务词、模型习惯或测试案例堆叠特判；同一个通用机制应同时解释目标案例与正常反例。

## 本地检查

```bash
uv sync --locked --link-mode copy
uv run ruff check .
uv run mypy
uv run lint-imports
uv run pytest -q
uv build
```

涉及可选依赖时使用 `uv sync --locked --all-extras --link-mode copy`，并在对应命令中保留 `--all-extras`。MLX 只适用于满足平台要求的机器；RAG 和浏览器具有各自依赖，未安装时相关测试可能明确跳过。

CI 将两个受支持平台分为独立任务，每个 PR 都运行：

- `linux`（Ubuntu 24.04）保留完整依赖、全量测试、真实 Bubblewrap、浏览器读取及隔离、模型服务隔离、构建和安装后 CLI/SDK 验收；共享的 Ruff、mypy、导入约束与历史验收资产检查在此运行一次。
- `macos`（macOS 14）只安装默认运行依赖与开发工具，先确认真实 Seatbelt 可执行，再运行命令会话及崩溃恢复、公开 CLI/SDK 沙箱路径、文件权限和终端回归；不重复安装 RAG/MLX 可选依赖。测试桩不能替代这些平台边界检查。

两个平台共用同一仓库、运行时和公共协议，差异放在 OS 适配层及对应测试中。提交按功能组织：macOS 修复、Linux 修复和共享协议修改尽量各自形成可审查提交；共享协议修改始终同时通过两端 CI。无需拆分成两套运行时代码。

涉及网页或连接管理页面时，另外执行浏览器回归：

```bash
uv run --extra browser playwright install --with-deps --only-shell chromium
PRAXIS_TEST_BROWSER=1 uv run --extra browser pytest -q \
  tests/agent/test_web_browser.py tests/agent/test_connection_manager_browser.py
```

## 验证与评估

下面两个入口只检查仓库维护的场景与历史记录，不调用模型：

```bash
uv run python scripts/agent_public_path_benchmark.py validate
uv run python scripts/agent_model_quality_gate.py validate
```

真实模型验收需要明确的模型、费用预算和独立结果目录。先用 `agent_public_path_benchmark.py freeze` 固定候选源码，再运行该快照，记录版本、模型、场景、预算和结果；具体参数见脚本 `--help`。历史通过记录不代表当前源码，也不能外推为任意编程或推理能力。

不调用真实模型的公共执行演示：

```bash
uv run python scripts/agent_delivery_smoke.py --fake-model
uv run python scripts/render_praxis_demo.py --help
```

README 中的演示来自真实公共 `agent_runtime.Agent` 路径，模型响应是固定脚本。它说明执行链路怎样工作，不作为模型质量分数。

## 贡献与问题定位

修改前核对 `git status`、实际工作区和相关源码。修改后运行与影响范围匹配的检查，并在 PR 中说明触发条件、行为变化、验证结果及未覆盖的平台。权限或恢复变更需验证实际入口和失败路径，避免只验证内部辅助函数。

报告问题时附操作系统、Python 版本、运行命令、错误码和可复现步骤。不要提交 API Key、个人 `.env`、`.praxis/`、聊天数据库或完整私有任务内容。设计历史、私人评估记录和参考仓库默认保留在本地；公开使用说明位于本目录，不依赖这些本地材料。

项目采用 [MIT](../LICENSE) 许可证。
