# Praxis

> **a trusted-local workspace agent runtime**

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Distribution: source only](https://img.shields.io/badge/distribution-source%20checkout-555)](#quickstart)

Praxis turns a model's plan into controlled work on files, code, data, documents,
and private knowledge. For default modification tasks, the runtime requires a real
workspace change and post-change verification before accepting completion.
Read-only tasks explicitly opt out of that mutation contract and may answer
directly. Praxis is designed for one person operating a trusted local workspace,
with `agent` as the CLI and `agent_runtime.Agent` as the Python API.

![Praxis deterministic fake-model demo](docs/assets/praxis-demo.gif)

**DETERMINISTIC DEMO · FAKE MODEL — NOT MODEL QUALITY EVIDENCE.** Every frame is
generated from the tested public Agent path. The scripted model inspects a file,
applies a patch, runs verification, and completes. It demonstrates runtime wiring
without credentials; real-model evidence is tracked separately.

## Why Praxis

Useful agent work is more than producing an answer:

```text
workspace knowledge -> controlled action -> verifiable result
```

Praxis keeps that path visible. The model can inspect the workspace, propose and
execute bounded tool calls, pause at approval boundaries, and continue from
durable state. Default modification turns finish against diff and verification
evidence. A read-only turn using `--no-require-workspace-change`, or the SDK with
`require_workspace_change=False`, may return analysis without manufacturing a
change. RAG is an optional private-knowledge capability, not the product's default
execution path.

## Runtime architecture

```text
CLI / Python SDK
       |
       v
Turn -> Loop -> ACI / ToolExecutor -> workspace
  |       |             |
  |       |             +-> approval before risky effects
  |       +-> model observation and bounded next step
  +-> checkpoint / resume / previous_turn_id
                       |
                       +-> verification and acceptance evidence
```

- **Turn** — one user request with one public `turn_id`; a later Turn may point
  to it through `previous_turn_id`.
- **Loop** — the bounded model/tool cycle that plans, observes results, and
  decides whether to continue, pause, fail, or finish.
- **ACI** — typed, documented tool contracts for files, structured data,
  managed Python, search, patching, commands, plans, knowledge, skills, and
  integrations.
- **Approval** — write and execute capabilities remain distinct and can pause
  before a destructive effect.
- **Checkpoint** — pending and interrupted Turns persist so the same operation
  can resume instead of being replayed as a new task.
- **Verification** — a claimed workspace change is checked against the real diff
  and verification performed after the final mutation.

The runtime deliberately uses one Agent loop rather than a chain of role-playing
agents. Deeper lifecycle details are in the
[product contract](docs/design/agent_product_contract.md).

## Current evidence

| Evidence | What it establishes | Current state |
| --- | --- | --- |
| [Deterministic demo](docs/assets/praxis-demo.gif) | Public Agent wiring: inspect, patch, verify, finish | Reproducible fake-model artifact |
| [Tool-use reliability benchmark](docs/benchmark.md) | Five fixed, controlled ACI scenarios, repeated three times each | **PASSED — 15/15 scenario executions passed**; worst-trial mean tool calls 1.6 |
| [Expanded DeepSeek V4 Flash run](docs/runs/deepseek-v4-flash.md) | Approval, continuation, mutation, validation, and redacted trace | **CONCLUSIVE PASS — 3/3 approval trials completed**; 2 tool calls per trial |
| [Real non-code data ACI run](docs/runs/deepseek-v4-flash-data-aci.md) | One combined Excel, PDF, and statistical-analysis Turn with managed Python, approval, artifact read-back, and independent acceptance | **CONCLUSIVE FUNCTIONAL PASS — 3/3 artifacts accepted**; one trial, not a reliability benchmark |
| [30-task protocol](evals/code_agent/benchmark_v1.json) | Manifest shape for a broader coding-agent evaluation | Manifest validated; not run as this change's release gate |

The tool-use reliability page is generated from a clean source commit after the
local gates pass. It measures five narrow behaviors: direct file reading, search
then read, recovery from a renamed file, approval and resume, and stopping after a
confirmed failure. It does not claim broad coding ability. Infrastructure failures
are reported as **INCONCLUSIVE**, never converted into a model score. The table
reports the measured verdict and exact execution count from the linked raw report;
inspect the expanded trace for case-level evidence.

## Quickstart

### Install from a source checkout

The distribution name `praxis-agent-runtime` is local build metadata. This
project is **not published to PyPI**: use a source checkout or build a wheel
locally. There is no package-index install command implied by this README.

```bash
git clone https://github.com/xiaoyinglei/praxis-agent-runtime.git
cd praxis-agent-runtime
uv sync --frozen
```

### Connect and switch models

Praxis accepts a secret-free model definition and reads any credential through
the named environment variable at runtime. For full Agent behavior, an endpoint
must provide:

- an OpenAI-compatible model-discovery and chat interface that advertises the
  configured model identity;
- a real streamed text delta followed by an authoritative completion;
- a schema-valid forced tool call when `supports_tools` is enabled, without
  Praxis executing the probe tool;
- valid structured output when `supports_structured_output` is enabled; and
- bounded timeouts and cancellable streaming.

Built-in model IDs remain read-only. User registrations live in the versioned
user registry. Agent startup automatically initializes the local binding trust
domain on first use and reuses it afterward. Inspect or register exact model IDs
through the CLI. For explicit setup before first use, run `uv run agent model trust init`;
inspect the existing trust domain with `uv run agent model trust status`.
`--provider` selects the provider
adapter and transport; the same `MODEL_ID` is sent to that provider:

```bash
uv run agent model list --source
uv run agent model current
export MODEL_ID=provider-model-id
export PROVIDER_BASE_URL=https://provider.example/v1
export PROVIDER_CREDENTIAL_ENV=MY_PROVIDER_TOKEN
export MY_PROVIDER_TOKEN=replace-with-provider-token

uv run agent model add "$MODEL_ID" \
  --provider openai_compatible \
  --context-window-tokens 131072 \
  --base-url "$PROVIDER_BASE_URL" \
  --api-key-env "$PROVIDER_CREDENTIAL_ENV"

uv run agent model show "$MODEL_ID"
uv run agent model probe "$MODEL_ID" --level full
uv run agent model update "$MODEL_ID" --timeout-seconds 90
uv run agent model switch "$MODEL_ID"
uv run agent model remove "$MODEL_ID"
```

`agent model add` and `update` run the full probe before their compare-and-swap
registry commit. Probe failure or cancellation writes nothing. Advanced typed
definitions can use `--from <one-model.yaml>`; `--skip-probe` is an explicit
offline escape hatch and reports the model ID as unverified. Registry files store
only the environment variable name, never its resolved value.

The session selection is mutable, both outside and inside interactive chat:

```text
$ uv run agent chat
> /model
当前模型: current-model-id
可用模型:
* current-model-id  ...
  another-model-id  ...
切换: /model <model_id>
> /model another-model-id
已切换模型: another-model-id
```

Interactive `agent chat` uses a live conversation viewport with Markdown answers,
a multiline composer, and a model/status bar. Execution groups and individual
tools expand and collapse **in place**, including while output is streaming.
Click the triangle or its row to toggle it; no command is needed. Errors remain
visible even when their execution group is collapsed.

| Control | Action |
| --- | --- |
| `/model` | Open model picker; arrows select, Enter confirms, Esc cancels; clicking a model also selects it |
| Drag over answer text | Select and copy to the system clipboard on macOS |
| Right click | Paste into the composer without sending |
| `F2` | Toggle native terminal mouse selection / interactive folding and scrolling |
| Click `▸` / `▾` | Expand / collapse the execution group or individual tool |
| `Ctrl+O` | Toggle the latest execution group without leaving the composer |
| `Tab`, then arrows and `Enter` / Space | Navigate records and toggle the selected row |
| `Tab` / `Esc` while navigating records | Return to the composer |
| Mouse wheel / `PageUp` / `PageDown` | Scroll the conversation viewport |
| `Ctrl+End` | Follow the latest output again |
| `Enter` / `Alt+Enter` | Send / insert a newline |
| `Ctrl+C` | Copy a selection first; otherwise cancel execution, clear input, or exit if empty |
| `Ctrl+D` at an empty prompt | Exit |

Typing `/` offers command completion. Input editing preserves complete Chinese
characters and emoji, and bracketed multiline paste stays in the composer until
you send it. Approval questions use the same composer and show the permitted
choices. Cancelling waits for execution cleanup; existing file changes are not
rolled back. Mouse interaction requires a terminal with mouse reporting and
cursor-position reports; keyboard folding is also available.
Unchanged history reuses its layout when scrolling or refreshing the status bar.
Resizing or receiving new content invalidates the layout. Use `F2` for native
terminal selection and copy shortcuts if the terminal does not support drag reporting.

Tool calls rejected before execution (for example, invalid arguments or an
out-of-workspace path) also appear in the execution records, marked as not
executed. Whitespace-only model chunks do not start a visible answer. Three
tool results with the same tool, arguments, and deterministic error without
observed progress stop the Turn with `repeated_tool_failure`. Read-only calls
and proven no-op writes do not hide the streak; actual workspace changes,
successful recovery of the same call, or unmeasured external effects reset it.
The check uses committed history, including after a process restart. A missing
file reports `file_not_found`, rather than a generic retryable runner failure.
Model context includes the actual workspace and the model ID frozen for that Turn.

OpenAI-compatible streams request and preserve provider usage, including a
usage-only trailing chunk. Token estimates include full tool descriptions and
schemas when provider usage is absent. An incomplete stream pauses with
`model_retry` and offers `retry` / `abort`; retry creates a new model attempt
without replaying tools from the unfinished response.

The live viewport retains up to 100 conversation blocks and 256 tools per
execution group, with bounded tool details (2,000 display rows and 131,072
characters per block; 8,388,608 text characters across the viewport). Display omissions are explicitly marked; folding never
changes durable history. Tool/ACI truncation is reported separately. `/verbose`
also opens or closes retained execution details. Redirected input/output and
`TERM=dumb` retain the plain-text chat path; `agent run` and `agent resume` retain
their existing command-line output.

The next message keeps the current conversation history and creates a new Turn
bound to the selected model ID; no restart or `/new` is required. An invalid ID
prints the available IDs, keeps the previous model, and does not contact a
provider. Each completed or paused Turn retains an authenticated, immutable
definition in durable history. `agent resume` therefore continues that Turn's
original model even after the model ID is updated, removed, or selected differently
for later Turns.

### 启动本地 MLX 模型与 Agent

以下命令在项目根目录执行。Praxis 不会自动启动或关闭 MLX 服务；
先启动模型服务，再启动 Agent。下载模型权重不会自动把模型加入 Agent 目录。
当前内置的本地聊天模型为：

- `mlx-community/Qwen3.5-9B-4bit`
- `mlx-community/gemma-4-26b-a4b-it-4bit`

#### 双窗口运行

窗口 1：启动 Qwen，保持服务运行。首次加载可能需要等待权重下载或加载完成。

```bash
uv run python -m mlx_lm.server \
  --model mlx-community/Qwen3.5-9B-4bit \
  --host 127.0.0.1 --port 8080
```

窗口 2：启动 Agent。首次启动自动创建本地 trust 密钥，后续启动复用。
启动后可用 `uv run agent model trust status` 查看状态；如果已有历史绑定却丢失密钥，
程序会报错要求恢复原密钥，不会自动生成替代密钥。

```bash
uv run agent chat --model mlx-community/Qwen3.5-9B-4bit
```

运行 Gemma 时，先在模型窗口按 Ctrl+C 停止旧服务，再执行：

```bash
uv run python -m mlx_lm.server \
  --model mlx-community/gemma-4-26b-a4b-it-4bit \
  --host 127.0.0.1 --port 8080
```

在另一个窗口启动对应的 Agent：

```bash
uv run agent chat --model mlx-community/gemma-4-26b-a4b-it-4bit
```

#### 单窗口运行

先确保没有其他模型服务占用 `8080`。下面在后台启动模型，等待接口就绪后进入
Agent；日志写入 `/tmp/praxis-mlx.log`。切换启动模型时，只需修改第一行的模型 ID。

```bash
praxis_model_id=mlx-community/Qwen3.5-9B-4bit

uv run python -m mlx_lm.server \
  --model "$praxis_model_id" \
  --host 127.0.0.1 --port 8080 \
  > /tmp/praxis-mlx.log 2>&1 &
praxis_mlx_pid=$!

until curl --max-time 2 -fsS http://127.0.0.1:8080/v1/models >/dev/null 2>&1; do
  if ! kill -0 "$praxis_mlx_pid" 2>/dev/null; then
    tail -40 /tmp/praxis-mlx.log
    break
  fi
  sleep 2
done

if kill -0 "$praxis_mlx_pid" 2>/dev/null; then
  uv run agent chat --model "$praxis_model_id"
fi
```

在聊天中输入 `/exit` 退出 Agent，随后在同一个终端关闭此次后台启动的模型：

```bash
kill "$praxis_mlx_pid"
```

需要查看模型启动日志时执行：

```bash
tail -40 /tmp/praxis-mlx.log
```

#### 查看与切换模型

在 Agent 聊天输入框中输入（不是终端命令）：

```text
/model
/model current
/model mlx-community/Qwen3.5-9B-4bit
/model mlx-community/gemma-4-26b-a4b-it-4bit
```

以上两条切换命令按需选一条。`/model` 只更新 Agent 的模型选择，不负责重启 MLX。
两个本地模型共用 `8080`，切换前需要先在模型窗口停止旧服务、启动目标模型；
下一条聊天消息会使用新选择并保留对话历史。

也可以在终端保存模型选择，再进入聊天：

```bash
uv run agent model list --source
uv run agent model switch mlx-community/Qwen3.5-9B-4bit
uv run agent chat
```

手工修改模型目录后，需要退出并重新启动 Agent 才能刷新列表。

### Run a task

Run a read-only task explicitly:

```bash
uv run agent run \
  "Read pyproject.toml and explain the public entry points." \
  --no-require-workspace-change
```

Read-only tasks can answer directly because `--no-require-workspace-change`
disables only the mutation requirement; it does not invent a verification claim.

For a task that may edit files or run verification, describe the outcome instead
of scripting tool names:

```bash
uv run agent run \
  "Add a typed timeout to the public API, update its tests, and verify the change."
```

Risky tool calls are presented for approval in an interactive terminal. A
non-interactive run pauses instead of silently approving them and prints the
`agent resume` command for the pending Turn.

### Python API

```python
import asyncio
from pathlib import Path

from agent_runtime import Agent


async def main() -> None:
    agent = Agent(
        model="<model-id>",
        workspace_path=Path("."),
    )
    result = await agent.run(
        "Read pyproject.toml and summarize the package boundaries.",
        require_workspace_change=False,
    )

    print(result.answer)
    print(result.turn_id)


asyncio.run(main())
```

The Python SDK execution surface is async-only. Use `await agent.run(...)` for a
new Turn, `await agent.resume(...)` only for an existing paused or interrupted
Turn, `await agent.read_result(...)` / `await agent.pending_input(...)` for durable
state reads, and `async for event in agent.stream(...): ...` for live events.
Standalone scripts may create the event loop once at their application boundary
with `asyncio.run(main())`; async applications should directly `await` the Agent.

## Capability map

| Capability | Public route | Typical work |
| --- | --- | --- |
| **Files and code** | `agent` / `agent_runtime.Agent` | Discover, read, search, patch, inspect diffs, run bounded verification |
| **Data and documents** | `inspect_data_file` + sandboxed `execute_python` | Inspect CSV/TSV/JSON/XLSX/XLSM/PDF inputs; calculate, transform, chart, and structurally verify generated artifacts without a workspace `.venv` |
| **Private knowledge** | Explicit `RAGKnowledgeConfig` | Retrieve cited evidence from a configured local knowledge index |
| **Extensions** | Workspace Skills, configured MCP servers, and bounded subagent delegation | Add installed ACI capabilities without replacing the core loop |

Capabilities are assembled for the current workspace. Availability does not
grant permission: tool visibility, write authority, command execution, network
access, and approval are separate controls.

## Optional RAG

The `rag` package and CLI handle ingestion, retrieval, storage, and diagnostics.
They are an optional provider boundary beneath Praxis—not an alternate Agent API.
An ordinary `agent run` does not initialize embedding, reranking, vector storage,
or knowledge services.

Attach private knowledge explicitly with a lazy provider:

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
    result = await agent.run(
        "Find the relevant policy evidence and summarize it.",
        require_workspace_change=False,
    )
    print(result.answer)


asyncio.run(main())
```

The RAG runtime initializes only when the model first calls the knowledge tool.
Index maintenance stays on the optional `rag` command:

```bash
uv run rag --help
```

See the [runbook](docs/RUNBOOK.md) for service and private-document workflows.

## Safety and limitations

Praxis targets a **trusted-local macOS/Python workspace**. Its controls reduce
accidental or unapproved effects; they do not turn hostile code, a malicious
repository, or an untrusted operator into a safe workload.

- Read/execute and workspace-write capabilities are distinct. Writes and command
  execution can require approval; `.git` mutations remain outside the default
  workspace-write boundary.
- Real `run_command` and `execute_python` execution currently require macOS
  `/usr/bin/sandbox-exec` and a Seatbelt profile. If that executable is absent,
  the tool will fail closed with `sandbox_unavailable`; on other platforms it is
  unavailable, and this repository has no equivalent command-sandbox safety
  evidence. The fake sandbox fixtures are test-only and are not safety evidence.
- `inspect_data_file` returns bounded previews, structural validity, and a
  runtime-computed SHA-256. That proves which artifact was inspected; it does
  not by itself prove that a formula, statistical method, or business conclusion
  is semantically correct.
- RAG evidence quality depends on parsing, indexing, retrieval configuration, and
  the source documents. A citation is traceability, not automatic truth.
- Model and provider availability is external infrastructure. Timeouts, quota,
  authentication failures, and malformed responses are reported separately from
  task quality.
- Checkpoints may contain bounded task metadata and sanitized tool observations;
  they are local state and should be protected like the workspace.
- This repository is not a multi-tenant remote execution service, and the current
  evidence does not establish that deployment boundary.

## Development gates and deeper documentation

The repository's local and CI gates check formatting, types, imports, tests,
buildability, installed-wheel behavior, public CLI/SDK smoke paths, and the
deterministic demo. Run the same entry points from the checkout:

```bash
uv run ruff check .
uv run mypy
uv run lint-imports
uv run pytest -q
uv build
```

Focused references:

- [Runbook](docs/RUNBOOK.md) — models, services, private knowledge, and operations
- [Troubleshooting](docs/TROUBLESHOOTING.md) — common runtime and RAG failures
- [Product contract](docs/design/agent_product_contract.md) — public lifecycle and boundaries
- [Tool-use reliability benchmark](docs/benchmark.md) — plain-language scenarios, scope, and current live result
- [Expanded DeepSeek V4 Flash run](docs/runs/deepseek-v4-flash.md) — human-readable approval-continuation evidence
- [Real non-code data ACI run](docs/runs/deepseek-v4-flash-data-aci.md) — one real combined Excel/PDF/statistics Turn with independent artifact acceptance
- [Evaluation archive](docs/EVALUATION.md) — historical retrieval baselines with provenance notes
- [MIT license](LICENSE) — use and redistribution terms

Praxis is available under the [MIT](LICENSE) license.
