# Agent API Rearchitecture Design

Date: 2026-07-02

Status: superseded

> Superseded on 2026-07-18 by
> `docs/superpowers/specs/2026-07-18-agent-public-api-lifecycle-cleanup-design.md`.
> Historical examples below describe the earlier proposal and are not the
> current SDK or CLI contract.

## 1. Why This Exists

The project has grown from a RAG-heavy runtime into a general agent runtime, but
the public API still exposes too many internal assembly details. Users currently
see names and concepts such as `RAGRuntime`, `AgentService`, `AgentRunRequest`,
`storage_root`, `embedding_model`, `reranker_model`, checkpoint internals, and
runtime model registries. These are implementation details, not a product API.

This redesign makes the product surface simple:

```text
agent     runs the agent
model     selects the chat/generation model
file      attaches local input files
knowledge attaches optional knowledge bases
tool      extends the agent with explicit capabilities
```

The first priority is CLI usability. The Python SDK must expose the same mental
model and call into the same facade, so CLI and SDK behavior cannot drift.

## 2. Design Goals

- Make `agent` the primary product entrypoint.
- Provide a clean Python SDK facade that mirrors the CLI.
- Move public package naming from `rag.*` to `agent_runtime.*`.
- Keep RAG as an optional knowledge provider, not a default runtime dependency.
- Make chat model selection as easy as `--model qwen14b`.
- Keep embedding and reranking out of the main Agent API.
- Treat budgets as telemetry and guardrails by default, not user-facing task
  requirements.
- Preserve a migration path for existing tests and lower-level RAG utilities.

## 3. Non-Goals

- Do not redesign the agent loop semantics in this API phase.
- Do not remove existing RAG ingest/query functionality immediately.
- Do not implement dynamic model selection in the first API pass.
- Do not make RAG auto-start based on `.rag` existing in the workspace.
- Do not expose internal runtime objects as recommended user APIs.

## 4. Current API Problems

### 4.1 CLI Leaks Internal Concepts

The CLI has a top-level `agent` command, but its implementation still accepts
hidden RAG storage/vector/model options and tries to attach RAG at startup.
Even hidden options shape the code path and failure modes.

Bad public mental model:

```text
agent run -> maybe build RAGRuntime -> maybe resolve embedding/reranker
```

Target public mental model:

```text
agent run -> build Agent facade -> run task
```

### 4.2 Python API Is an Internal API

The current public exports include `AgentService`, `AgentRunRequest`,
`AgentRuntimePolicy`, `ToolRegistry`, `RAGRuntime`, and storage assembly types.
These are useful for framework development, but they are not a clean SDK for
users who want to run an agent.

### 4.3 Model Configuration Has Two Sources of Truth

Agent model resolution and RAG runtime model resolution are separate enough that
changing a model feels like touching multiple systems. Users want to select a
chat model. They should not think about embedding/reranker unless they are
working with a knowledge base.

### 4.4 RAG Is Too Eager

RAG currently can be attached during Agent startup. This makes startup heavy and
turns a tool capability into a default dependency. RAG must become an explicit
knowledge provider that initializes only when the model actually calls the
knowledge tool.

## 5. Target Package Structure

New primary package:

```text
agent_runtime/
  __init__.py
  agent.py
  cli.py
  config.py
  models.py
  result.py
  events.py
  tools.py
  knowledge.py
  runtime/
    __init__.py
    service.py
    loop/
    checkpointing.py
    policies.py
  workspace/
    __init__.py
    files.py
    manifest.py
    primitive_ops.py
  knowledge_providers/
    __init__.py
    rag.py
```

Existing `rag/` package remains temporarily:

```text
rag/
  # legacy compatibility and lower-level RAG subsystem
```

Long-term ownership:

- `agent_runtime.*` owns the product API and agent runtime.
- `agent_runtime.knowledge_providers.rag` adapts the existing RAG subsystem.
- `rag.*` is legacy/lower-level RAG implementation until it can be moved or
  formally scoped as a separate package.

## 6. Public CLI API

### 6.1 Core Commands

```bash
agent run "总结这个项目" --model qwen14b
agent run "分析这个 Excel" --file report.xlsx --model qwen14b
agent chat --model qwen14b
agent resume <run-id>
```

Short aliases:

```bash
agent run "分析这个 Excel" -f report.xlsx -m qwen14b
```

### 6.2 Knowledge Commands

Knowledge is explicit:

```bash
agent run "P1 响应时间是多少？" --knowledge company_docs
agent knowledge ingest company_docs ./docs
agent knowledge list
agent knowledge inspect company_docs
agent knowledge remove company_docs
```

No `--knowledge`, no knowledge tools.

### 6.3 Model Commands

```bash
agent model list
agent model current
agent model switch qwen14b
```

Model switching is runtime session state. It must not rewrite
`configs/models.yaml` or imply a global default change.

The main `agent run` help should not show embedding/reranker/storage/vector
options.

### 6.4 Advanced Commands

Advanced options are allowed, but they must be clearly separated:

```bash
agent run "..." --max-turns 20
agent run "..." --timeout 300
agent run "..." --trace
agent run "..." --workspace ./workspace
```

Budget options should not be in first-level examples. If retained, they should
live under advanced help and be named as explicit limits:

```bash
agent run "..." --max-tokens-total 100000
```

## 7. Public Python SDK API

### 7.1 Simple Agent

```python
from agent_runtime import Agent

agent = Agent(model="qwen14b")
result = agent.run("总结这个项目")

print(result.answer)
```

The same method continues context without a separate conversation object:

```python
first = agent.run("记住项目代号 ORCHID-731")
second = agent.run("项目代号是什么？", previous_turn_id=first.turn_id)
```

Omitting `previous_turn_id` always starts a root Turn. `resume(turn_id, ...)`
is reserved for continuing paused or interrupted execution of that same Turn.

### 7.2 Files

```python
result = agent.run(
    "读取这个 Excel，列出结构并给出摘要",
    files=["report.xlsx"],
)
```

### 7.3 Knowledge

```python
from agent_runtime import Agent

agent = Agent(
    model="qwen14b",
    knowledge=["company_docs"],
)

result = agent.run("P1 工单首次响应目标是多少？请给出处")
```

### 7.4 Async and Streaming

```python
async for event in agent.astream("分析这个文件", files=["report.xlsx"]):
    print(event.type, event.data)
```

### 7.5 Result Object

Public result shape:

```python
result.answer
result.status
result.files
result.tool_calls
result.citations
result.usage
result.diagnostics
result.turn_id
result.pause
```

Internal fields such as `LoopState`, raw `ToolResult`, and internal checkpoint
payloads should not be required for normal SDK usage.

## 8. Configuration API

Default config path:

```text
agent.yaml
```

Minimal config:

```yaml
default_model: qwen14b

models:
  qwen14b:
    provider: openai-compatible
    model: mlx-community/Qwen3-14B-4bit
    base_url: http://127.0.0.1:8080/v1
```

Knowledge config:

```yaml
knowledge:
  company_docs:
    provider: rag
    storage: data/company_docs
    vector_backend: sqlite
    embedding_model: qwen_embed
    reranker_model: none

models:
  qwen_embed:
    provider: mlx-embedding
    model: mlx-community/Qwen3-Embedding-4B-4bit-DWQ
```

Rules:

- `models.*` can contain chat, embedding, or reranker entries.
- `Agent(model=...)` and `agent run --model ...` resolve only chat models.
- Knowledge providers own their embedding/reranker configuration.
- Agent startup does not initialize embedding/reranker models.

## 9. Model Control Plane

This layer is not a `ModelRouter`, task classifier, or capability router. It is
a small control plane for selecting the current chat model in a run/session.

Public mental model:

```python
from agent_runtime import Agent

agent = Agent(model="qwen14b")
agent.switch_model("mimo_cloud")
result = agent.run("继续当前任务")
```

Internal responsibilities:

- `ModelSpec`: runtime-facing model declaration with provider, provider model,
  context window, tool support, structured output support, local/cloud location,
  and optional cost metadata.
- `ModelCatalog`: list and validate known chat models from `configs/models.yaml`.
- `ModelSessionState`: hold the current model id for the runtime session.
- `ModelPolicy`: review switch requests, including requests initiated by the
  agent, before mutating session state.
- `ModelControlPlane`: shared facade used by Agent, CLI, SDK, and provider
  resolution.
- `LocalRuntimeManager`: for `location: local` models, check `runtime.health_url`,
  launch exactly `runtime.launch_command` when needed, poll until ready, and
  fail on endpoint conflict if the endpoint is serving a different model.
- LLM provider construction still happens through the existing thin resolver;
  when no node-specific model is set, it resolves `session.current_model_id`.

Not responsible for:

- Auto-selecting a model based on task complexity.
- Introducing `ModelRouter`, `TaskClassifier`, or `CapabilityRouter`.
- Choosing embedding/reranker models for generation.
- Killing local model processes, changing ports, or silently selecting a
  different model.
- Initializing RAG or vector infrastructure.

## 10. Knowledge and RAG Boundary

RAG becomes an explicit knowledge provider.

Startup flow:

```text
agent run --knowledge company_docs
  -> load knowledge config metadata only
  -> register lightweight search_knowledge tool card
  -> model may call search_knowledge
  -> lazy initialize provider on first call
  -> run retrieval
```

Without `--knowledge`, `tool_search` must not return RAG tools.

Provider behavior:

- No heavy initialization at Agent construction.
- No model download or embedding service probing at Agent startup.
- Initialization happens on first tool call.
- Initialization errors are returned as tool errors with actionable messages.
- Provider exposes diagnostics in `result.diagnostics`.

## 11. Tool API

Keep a public tool decorator small:

```python
from agent_runtime import Agent, tool

@tool
def add(a: int, b: int) -> int:
    return a + b

agent = Agent(model="qwen14b", tools=[add])
```

Internal `ToolSpec`, permissions, approvals, retries, and execution records
remain in the runtime layer. Advanced users can import from
`agent_runtime.runtime.tools`, but product docs should lead with `@tool`.

## 12. Budget and Usage

Default behavior:

- Run the task.
- Track usage.
- Prevent infinite loops with `max_turns`.
- Prevent indefinite hangs with `timeout`.
- Protect model context windows with compaction or controlled failure.

Public result:

```python
result.usage.input_tokens
result.usage.output_tokens
result.usage.total_tokens
result.usage.tool_calls
result.usage.latency_ms
```

Hard budget limits are advanced configuration:

```python
agent = Agent(model="qwen14b", limits={"max_tokens_total": 100000})
```

or CLI:

```bash
agent run "..." --max-tokens-total 100000
```

The first priority is successful task completion. Budget accounting is
observability and safety, not the default user-facing control plane.

## 13. Public vs Internal API

### 13.1 Public

```python
from agent_runtime import Agent, AgentResult, AgentConfig, tool
```

CLI:

```bash
agent run
agent chat
agent resume
agent model
agent knowledge
```

### 13.2 Internal

These move behind `agent_runtime.runtime` or remain legacy under `rag`:

```text
AgentService
AgentRunRequest
AgentRuntimePolicy
AgentServiceFactory
ModelRegistry
RAGRuntime
RuntimeOverrides
StorageConfig
CapabilityAssemblyService
ToolRegistry
ToolSpec
```

They may remain importable for compatibility in the short term, but they are not
the recommended API.

## 14. Migration Strategy

### Phase 1: Add Facade, Keep Internals

- Add `agent_runtime.Agent`.
- Add `agent_runtime.AgentResult`.
- Add `agent_runtime.config` and `agent_runtime.models`.
- Change `agent` CLI to call the facade.
- Keep existing `agent_runtime.service.AgentService` internally.
- Do not move files yet unless necessary.
- Add the first Model Control Plane slice: `ModelSpec`, `ModelCatalog`,
  `ModelSessionState`, `ModelPolicy`, `ModelControlPlane`, and
  `agent model list/current/switch`.

Success:

```bash
agent run "hi" --model qwen14b
agent run "分析文件" --file README.md
```

```python
from agent_runtime import Agent
Agent(model="qwen14b").run("hi")
```

### Phase 2: Move Package Boundaries

- Move or wrap `agent_runtime.*` under `agent_runtime.runtime.*`.
- Keep compatibility imports with deprecation warnings.
- Update docs and tests to prefer `agent_runtime`.
- Make `rag` clearly a knowledge/RAG lower-level subsystem.

### Phase 3: Knowledge Provider Split

- Add `agent knowledge ...` commands.
- Remove RAG auto-attach from `agent run`.
- Implement lazy RAG provider.
- Retire hidden RAG options from `agent` implementation.

### Phase 4: Cleanup and Deprecation

- Remove main README references to `RAGRuntime` as a user-facing path.
- Keep low-level RAG docs under a dedicated advanced section.
- Remove old compatibility paths only after tests and docs fully migrate.

## 15. Acceptance Criteria

CLI help:

```text
agent run --help
```

Must show first-level options:

```text
--model
--file
--knowledge
--workspace
--max-turns
--timeout
--verbose
```

Must not show first-level options:

```text
--storage-root
--embedding-model
--reranker-model
--vector-backend
--vector-dsn
--budget
```

Python smoke:

```python
from agent_runtime import Agent

result = Agent(model="qwen14b").run("2+2 等于几？")
assert result.answer
```

File smoke:

```bash
agent run "总结 README" --file README.md --model qwen14b
```

Knowledge smoke:

```bash
agent run "P1 响应时间是多少？" --knowledge company_docs
```

The knowledge smoke must not initialize RAG until the model calls the knowledge
tool.

## 16. Known Issues To Fix During This Refactor

- `ToolCallMetrics` is missing from checkpoint msgpack allowlist.
- `--checkpoint-db` currently constructs async SQLite checkpointing from sync CLI
  code and can fail with no running event loop.
- File-heavy tasks can overflow the tool-decision context before tools run.
- Skills can be over-triggered when their descriptions are too visible or too
  broad.
- Model defaults currently drift between docs, config, and runtime behavior.

These are not all API issues, but this refactor should expose and fix them
because they affect the product path.

## 17. Documentation Changes

README should lead with:

```bash
agent run "..." --model qwen14b
agent run "..." --file report.xlsx
agent run "..." --knowledge company_docs
```

Python SDK docs should lead with:

```python
from agent_runtime import Agent
```

Advanced RAG maintenance docs should move under:

```text
docs/rag/
```

or:

```text
docs/advanced/rag.md
```

## 18. Open Decisions

1. Final public package name: `agent_runtime` is proposed.
2. Final config filename: `agent.yaml` is proposed.
3. Whether `--file` should replace `--input-file` immediately or support both
   during migration.
4. Whether `rag` CLI should remain indefinitely as low-level maintenance API or
   become `agent knowledge` only.
5. Whether compatibility imports should emit deprecation warnings immediately or
   only after the new facade is stable.

## 19. Recommendation

Proceed with Phase 1 first: add the facade and make the CLI call it. This gives
users the new API without breaking the existing runtime internals. Once the
facade is stable, move package boundaries and split RAG into a lazy knowledge
provider.
