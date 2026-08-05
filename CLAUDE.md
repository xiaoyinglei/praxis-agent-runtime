# CLAUDE.md — Praxis Coding Agent Reference

> 给在本仓库中工作的 coding agent 使用的简明入口。产品定位见
> [README](README.md)，公开生命周期见
> [Praxis 产品契约](docs/design/agent_product_contract.md)。

## 开发环境

Praxis 使用 Python 3.12 和 `uv`。当前 distribution
`praxis-agent-runtime` 从 source checkout 构建，不假设已从包索引安装。

```bash
cd /path/to/praxis-agent-runtime
uv sync --locked
```

使用 `uv run ...` 执行 Python 工具，不要绕过锁定环境。本地 `.env` 可为
模型 provider 提供密钥；不要把密钥、provider headers 或个人路径写入
源码、测试、文档或证据产物。

## 完整门禁

不忽略已知失败，也不用部分测试代替完整门禁。提交候选版应与 CI
保持一致：

```bash
uv run ruff check .
uv run mypy
uv run pytest -q
uv run lint-imports
uv build

uv run python scripts/agent_cli_smoke.py
uv run python scripts/agent_delivery_smoke.py --fake-model --verbose
uv run python scripts/agent_tool_aci_eval.py --fake-model --json
uv run python scripts/agent_code_benchmark.py validate \
  evals/code_agent/benchmark_v1.json --repository .
```

小改动先跑聚焦测试，但在声称交付或制作证据源提交前，仍要跑上述
完整门禁。模型真实运行是单独的 evidence gate；infrastructure failure 记为
INCONCLUSIVE，不伪装成质量分数。

## 运行时心智模型

Praxis 只有一个 Agent 内核，不通过多角色 Agent 转发来组装主路径：

```text
agent CLI / agent_runtime.Agent
              -> AgentService
              -> AgentLoop
              -> ToolRegistry snapshot
              -> ToolExecutor
              -> ToolResult / checkpoint / StreamEvent / AgentResult
```

- 每个用户请求是一个 Turn，对外只暴露 `turn_id`。
- `previous_turn_id` 创建后续 Turn；`resume` 只恢复已暂停或中断的原 Turn。
- 计划、工具调用、审批、checkpoint 和验证通过 canonical state/event 展示，
  不建第二套旁路。
- RAG 是显式配置的 knowledge provider，不是普通文件或代码任务的默认
  入口。

## ACI 与工具面

六个基础 coding tools 按固定顺序常驻：

1. `search_text`
2. `list_files`
3. `read_file`
4. `apply_patch`
5. `run_command`
6. `update_plan`

工具的 schema、描述、effect、target、超时、取消和输出边界都是 ACI 合同，
模型 prompt 或 README 不能覆盖它们。

可选能力层包含 knowledge、MCP、skills 和 subagent：

- 显式配置 knowledge 时安装 `search_knowledge`，RAG 资源在首次调用时懒加载。
- 存在可用 skill 时安装 skill bridge tools，仍受 catalog、policy 和资产路径
  hard guard 限制。
- 隐藏且 discoverable 的 MCP、subagent 或其他扩展通过 `find_tools` 搜索并激活。
- `find_tools` 只是 deferred discovery ACI，不是新的 registry 或 executor。

Canonical `Tool` 定义在 `agent_runtime/tools/tool.py`；所有工具来源都装入
`agent_runtime/tools/registry.py` 的 `ToolRegistry`，冻结后交给
`agent_runtime/tools/executor.py` 的唯一 `ToolExecutor` 执行卡口。写入和执行
权限分离；工具可见不等于已授权。

## 关键路径

| 责任 | 当前路径 |
| --- | --- |
| 单 Agent while-loop | `agent_runtime/loop/runtime.py` |
| Generic definition 与 system prompt | `agent_runtime/builtin/generic.py` |
| Tool 选择、deferred discovery 与激活 | `agent_runtime/tools/selection.py` |
| 结构化文件预览模型 | `agent_runtime/primitive_ops.py` |
| Canonical Tool / ToolCall / ToolResult | `agent_runtime/tools/tool.py` |
| Tool 注册与冻结 | `agent_runtime/tools/registry.py` |
| 权限到执行的唯一卡口 | `agent_runtime/tools/executor.py` |
| Agent CLI | `agent_runtime/cli.py` |
| RAG ingest/query/diagnostics CLI | `rag/cli.py` |
| 产品 runtime 装配 | `agent_runtime/runtime/builder.py` |
| 模型 alias 目录 | `configs/models.yaml` |

## 修改原则

- 优先修改现有主路径的最小 choke point，不新建 Registry、Executor、event
  system 或第二套 runtime。
- 先用小测试重现错误，再修实现；不改测试来隐藏回归。
- 工具变更必须同时维护详细文档、schema、effect/approval 边界和测试。
- 修改类 Turn 需要真实 workspace diff，并在最后一次写入后运行可识别验证；
  空文本完成不是交付证据。
- `.git`、密钥、provider 环境和 workspace 外路径都是独立安全边界。
- 嵌入模型切换后必须重建对应向量索引；表格真实值不能只从
  `sample_rows` 推断。

对外使用 `agent run` / `agent chat` / `agent resume` 和 `agent model ...`。RAG 的入库、
查询与诊断使用独立 `rag` CLI。参数和行为以当前 `--help`、公开 API 签名与
工具 schema 为准，不保留已删除的兼容旗标。
