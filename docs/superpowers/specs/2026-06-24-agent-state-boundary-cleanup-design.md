# Agent State Boundary Cleanup Design

> **历史设计记录：** 其中的 persistent-memory 方案已于 2026-07-25 从产品运行时移除；相关内容不再代表当前产品契约。

## 0. 核心矛盾

> Agent runtime 已经是通用内核，但状态模型、上下文注入层、工具合约还保留着 RAG-first 的语义惯性。

## 1. 设计原则

1. **LoopState 不理解工具输出语义。** 删除所有 RAG-era 字段。
2. **`tool_results` 是唯一工具输出 ledger。** 大对象通过 `ExternalizedToolOutput + MemoryRef` 外置。
3. **语义解释在工具层。** Tool output schema / ToolOutputFormatter / ACI contract 承担，ContextBuilder 只调度。
4. **每个事实只有一个 canonical 来源。** 消灭状态通道重复和双轨 pending。
5. **先建新通道 → 迁移调用方 → 最后删旧字段。** 不一次性删除，确保每个 PR 可独立验证。

## 2. LoopState 新边界

### 2.1 最终字段清单（27 个 → 之前 60+）

```python
class PlanState(BaseModel):
    agent_plan: AgentPlan | None = None
    plan_events: list[PlanEvent] = Field(default_factory=list)


class PersistentMemorySnapshot(BaseModel):
    """持久记忆的 bounded 快照，不存全文。所有字段有默认值，"未加载"是合法初始状态。"""
    index_ref: str = ""               # MEMORY.md 的 externalized ref
    index_digest: str = ""            # 内容摘要（≤500 chars），空字符串 = 未加载
    selected_count: int = 0           # 本轮选中的记忆数
    selected_summaries: list[str] = Field(default_factory=list)  # bounded summaries


class MemoryState(BaseModel):
    working_summary: WorkingSummary | None = None
    extracted_facts: list[ExtractedFact] = Field(default_factory=list)
    context_budget: ContextBudgetSnapshot | None = None
    memory_refs: list[MemoryRef] = Field(default_factory=list)
    memory_budget: MemoryBudgetSnapshot | None = None
    memory_warnings: list[str] = Field(default_factory=list)
    reactive_compact_used: bool = False
    persistent: PersistentMemorySnapshot = Field(default_factory=PersistentMemorySnapshot)


class DiscoveryCandidate(BaseModel):
    """工具发现候选——typed 替代 dict[str, object]"""
    tool_name: str
    description: str
    relevance_score: float = 0.0
    metadata: dict[str, object] = Field(default_factory=dict)


class DiscoveryEvent(BaseModel):
    """工具搜索历史事件——typed 替代 dict[str, object]"""
    query: str
    timestamp_iso: str
    result_count: int = 0
    selected_tools: list[str] = Field(default_factory=list)


class DeferredToolState(BaseModel):
    active_tools: list[str] = Field(default_factory=list)
    active_tool_iterations: dict[str, int] = Field(default_factory=dict)
    last_candidates: list[DiscoveryCandidate] = Field(default_factory=list)
    last_search_query: str = ""
    search_history: list[DiscoveryEvent] = Field(default_factory=list)
    pinned_tools: list[str] = Field(default_factory=list)
    capability_diagnostics: list[RuntimeDiagnostic] = Field(default_factory=list)


class FinishState(BaseModel):
    feedback: list[StopHookFeedback] = Field(default_factory=list)
    warnings: list[StopHookFeedback] = Field(default_factory=list)


class PendingToolCall(BaseModel):
    """单一 canonical pending tool call——替代 ToolCallPlan + 旧 PendingToolCall 双轨"""
    plan: ToolCallPlan
    status: Literal["pending", "approved", "denied", "running", "completed", "failed"]
    approval_request_id: str | None = None
    operation_id: str | None = None


class ToolCallLedgerEntry(BaseModel):
    """单条 tool call 的原始 transcript source——不含运行时状态。"""
    plan: ToolCallPlan
    turn: int          # 发生在第几个 model turn
    sequence: int      # 该 turn 内的序号


class ToolCallLedger(BaseModel):
    """Bounded 保留所有 tool call 的原始 plan（含 completed/failed）。
    只有当该 tool call 已不再需要参与 native transcript 重建时才可清理。
    避免 transcript 重建时因 pending_tool_calls 清空而丢失 assistant tool-call 参数。"""
    entries: list[ToolCallLedgerEntry] = Field(default_factory=list)
    max_entries: int = 128  # bounded，超限时按 (turn, sequence) FIFO 删除


class LoopState(TypedDict):
    task: str
    messages: list[BaseMessage]
    run_config: AgentRunConfig
    iteration: int
    status: LoopStatus

    pending_tool_calls: list[PendingToolCall]       # 单轨，active (pending/approved/running)
    tool_call_ledger: ToolCallLedger                # bounded 全量 ledger（含 completed/failed）
    tool_execution_records: dict[str, ToolExecutionRecord]
    tool_results: list[ToolResult]                   # 唯一工具输出 ledger

    approval_request: HumanInputRequest | None
    approval_response: HumanInputResponse | None
    approved_tool_call_ids: list[str]
    denied_tool_call_ids: list[str]

    pause: LoopPause | None
    terminal: LoopTerminal | None
    latest_transition: LoopTransition | None
    last_model_turn: ModelTurn | None

    plan_state: PlanState
    memory_state: MemoryState
    deferred_tool_state: DeferredToolState
    finish_state: FinishState

    runtime_diagnostics: list[RuntimeDiagnostic]

    final_answer: str | None
    final_output: ValidatedFinalOutput | None
    output_validation_errors: list[dict[str, object]]

    file_manifest: FileManifest | None
```

### 2.2 删除清单（14 个）

```text
retrieval_signals              → RAG tool 内部 preprocessor
retrieval_signals_debug        → RAG tool 内部
evidence                       → ToolResult.output
citations                      → ToolResult.output
evidence_refs                  → ToolResult.output (structured_observations 内嵌重复)
answer_candidates              → ToolResult.output
computation_results            → ToolResult.output
structured_observations        → ToolResult.output
context_units                  → ToolResult.output
context_bindings               → ToolResult.output
locators                       → ToolResult.output
asset_refs                     → ToolResult.output
groundedness_flag              → RAG generation tool output
insufficient_evidence_flag     → RAG generation tool output
```

### 2.3 额外删除

```text
tool_result_store              → 与 ExternalizedToolOutput + MemoryRef 重复，统一到 tool_results
loop_messages                  → 改为 provider/context builder 的 derived transcript，不进 LoopState/checkpoint
```

### 2.4 双轨合并

```text
pending_tool_calls: list[ToolCallPlan]        ─┐
pending_loop_tool_calls: list[PendingToolCall] ─┘
  → pending_tool_calls: list[PendingToolCall]   ← 单轨 canonical
```

`PendingToolCall` 不含 `attempt_count/result/error`——这些仍以 `tool_execution_records[tool_call_id]` 和 `tool_results` 为 canonical 来源。

### 2.5 子状态收敛

| 子状态 | 收敛的字段 | 边界原则 |
|--------|-----------|---------|
| `PlanState` | `agent_plan` + `plan_events` | agent 自管理规划状态 |
| `MemoryState` | `working_summary` + `extracted_facts` + `context_budget` + `memory_refs` + `memory_budget` + `memory_warnings` + `reactive_compact_used` + `persistent` | 运行中可用的记忆上下文；persistent 是 bounded snapshot 不存全文 |
| `DeferredToolState` | `active_tools` + `active_tool_iterations` + `last_candidates` + `last_search_query` + `search_history` + `pinned_tools` + `capability_diagnostics` | 按需工具发现状态；`DiscoveryCandidate` / `DiscoveryEvent` typed |
| `FinishState` | `stop_hook_feedback` + `stop_hook_warnings` | finish gate / loop 控制流（不是 memory） |

## 3. ContextBuilder 解耦

### 3.1 ToolOutputFormatter 合约

```python
class ToolOutputFormatter(Protocol):
    tool_name: str

    def format_result(self, result: ToolResult) -> ContextSection | None: ...
    def format_externalized(self, ref: ExternalizedToolOutput) -> ContextSection | None: ...
```

### 3.2 ObservationExtractor 退出 AgentLoop

当前 runtime 在每轮 tool 执行后的实际调用链为（`rag/agent/loop/runtime.py:691-698`）：

```python
batch = self._observation_extractor.extract(new_results, seen_tool_call_ids=[...])
self._merge_observations(state, batch)    # ← 写入 LoopState 的入口
self._record_plan_observations(state, batch)
```

`_merge_observations()`（`runtime.py:1196`）将 `ObservationBatch` 平铺写入 9 个 LoopState 字段：

```python
state["structured_observations"] = ...
state["answer_candidates"] = ...
state["evidence_refs"] = ...
state["computation_results"] = ...
state["context_units"] = ...
state["locators"] = ...
state["asset_refs"] = ...
state["evidence"] = ...
state["citations"] = ...
```

这是 RAG-era 语义理解层的核心残留——agent core 仍然在规范化工具输出并写入 LoopState。

**PR2 的变更：**
- 删除 `_merge_observations()` 中对 LoopState 的全部 9 个写入
- `_merge_observations()` 改为只更新 bounded in-memory ledger（不写入 LoopState dict）
- `_record_plan_observations()` 改为只读 `tool_results` 通用状态（`tool_call_id`, `tool_name`, `status`），不读工具特定语义
- `ObservationExtractor` 保留（可内部使用），但不再作为 LoopState 写入者
- `ObservationBatch` 保留（formatter 可用），但 `as_state_update()` 方法标记 deprecated

### 3.3 ContextBuilder 职责分界线

**保留（loop 控制流 / 记忆管理）：**
- `_format_task()` — task 是 loop 输入
- `_format_plan()` — plan 是 agent 自管理状态
- `_format_working_memory()` — memory 层产出
- `_format_memory_refs()` — 记忆管理基础设施
- `_format_historical_hints()` — persistent memory 注入
- `_format_message_tail()` — messages 是 loop 基础
- `_format_loop_open_decisions()` — approval + stop_hook 是 loop 控制流

**删除（迁移到 ToolOutputFormatter）：**
- `_format_evidence()` → RAG retrieval/generation tool formatter
- `_format_structured_observations()` → `_format_tool_context()` 替代
- `_format_locator()` 中的 36 个硬编码字段 → 各工具 formatter 自处理

**新增：**
- `_format_tool_context()` — 遍历 `tool_results`，查 ToolRegistry 拿 formatter，调度渲染

### 3.4 通用 fallback

工具未注册 formatter 时，使用 `_format_tool_results()` 的现有 fallback（status + output 摘要）。

## 4. 迁移规则：可丢 vs 必须降级保留

旧 checkpoint 加载时的显式规则：

| 旧字段 | 处置 | 理由 |
|--------|------|------|
| `structured_observations` | **可丢** | ToolResult.output 可重建 |
| `evidence` | **可丢** | ToolResult.output 可重建 |
| `citations` | **可丢** | ToolResult.output 可重建 |
| `evidence_refs` | **可丢** | ToolResult.output 内嵌副本 |
| `answer_candidates` | **可丢** | ToolResult.output 可重建 |
| `computation_results` | **可丢** | ToolResult.output 可重建 |
| `context_units` | **可丢** | ToolResult.output 可重建 |
| `context_bindings` | **可丢** | ToolResult.output 可重建 |
| `locators` | **可丢** | ToolResult.output 可重建 |
| `asset_refs` | **可丢** | ToolResult.output 可重建 |
| `retrieval_signals` | **可丢** | 由 RAG tool 内部重建 |
| `groundedness_flag` | **可丢** | 旧快照值无重建价值 |
| `insufficient_evidence_flag` | **可丢** | 旧快照值无重建价值 |
| `pending_loop_tool_calls` | **降级保留** | 含 assistant tool-call 参数和顺序，`tool_results` 不能可靠恢复 |
| `loop_messages` | **降级保留** | 标记 `_needs_transcript_rebuild = True`；provider 从 `tool_call_ledger + tool_results` 重建 |
| `tool_result_store` | **可丢** | 与 ExternalizedToolOutput 重复，且 run 中断后无恢复价值 |

## 5. retrieval_signals 迁移路径

当前 `retrieval_signals` 不止在 RAG tool 中使用，以下路径需迁移：

| 当前引用方 | 迁移方式 |
|-----------|---------|
| `rag/agent/core/tool_execution.py:102` | 改为从 RAG tool 内部 preprocessor 获取，不经过 LoopState |
| `rag/agent/core/llm_prompts.py:13` (`build_retrieval_hint_prompt`) | 删除此函数；retrieval hint 由 RAG tool formatter 自主渲染 |
| `rag/agent/service.py` | 删除对 `retrieval_signals` 的读写 |
| `rag/query_pipeline.py:903` | 改为 RAG tool 内部传递 RetrievalSignals |
| `rag/agent/core/llm_context.py:78-98` (`assemble_retrieval_hint`) | 删除此方法；retrieval hint 不再作为独立 LLM 调用阶段 |

结论：`retrieval_signals` 变成 RAG retrieval tool 的内部 preprocessor 状态，不暴露到 LoopState。

## 6. 实施计划（3 个 PR）

### 6.1 PR1：子状态收敛 + legacy migration（只加不删）

**原则：新通道先建立，旧字段保留，代码不断。**

- 定义 `PlanState`, `MemoryState`, `FinishState`, `DeferredToolState` Pydantic models
- 定义 `PersistentMemorySnapshot`（替代裸 `memory_index: str`）
- 定义 `DiscoveryCandidate`, `DiscoveryEvent`（替代 `list[dict[str, object]]`）
- `create_loop_state()` 同时写新子状态对象 + 旧扁平字段（dual-write）
- `_migrate_legacy_state()` 初始化：加载旧 checkpoint 时填充新子状态（从旧扁平字段读取）
- `_normalize_loaded_state()` → 调用 `_migrate_legacy_state()`
- Checkpoint 保存新子状态 + 旧扁平字段（dual-write）
- 调用方（runtime, context builder, service）可选地开始从新子状态读取（dual-read）
- `memory_index` 收敛：`service.py:882` 改为写 `PersistentMemorySnapshot`，旧 `memory_index: str` 保留到 PR3 删除
- 测试：子状态 roundtrip、legacy 迁移逻辑、dual-read/write 一致性
- **不删除任何字段，不修改 allowlist**

### 6.2 PR2：工具输出边界清理

**原则：消除 RAG 语义写入 LoopState 的路径，ContextBuilder 改 formatter。**

- 定义 `ToolOutputFormatter` Protocol
- `ToolRegistry` 新增 `register_formatter()`
- `ObservationExtractor` 不再写入 LoopState：
  - AgentLoop runtime 中删除 `_merge_observations()` 对 LoopState 的 9 个写入
  - `ObservationExtractor` 保留（仍可内部使用），但不产出 LoopState 更新
- Planner 进展跟踪改为只读 `tool_results` 通用状态
- ContextBuilder 新增 `_format_tool_context()`（调度 formatter），删除：
  - `_format_evidence()`
  - `_format_structured_observations()` → 由 `_format_tool_context()` 替代
  - `_format_locator()` 中的 36 个硬编码字段
- RAG 工具实现 formatter：
  - `vector_search`, `keyword_search`, `grounding`, `rerank`, `graph_expand`
  - 各自注册 `ToolOutputFormatter`，把 evidence/citation/locator 渲染逻辑从 ContextBuilder 搬过来
- 文件工具实现 formatter：
  - `list_files`, `read_file`, `write_file`, `run_python`, `structured_probe`
- `retrieval_signals` 迁移：
  - 删除 `rag/agent/core/llm_prompts.py` 的 `build_retrieval_hint_prompt()`
  - 删除 `rag/agent/core/llm_context.py` 的 `assemble_retrieval_hint()`
  - RAG retrieval tool 内部 preprocessor 处理 retrieval hint
  - `service.py`, `query_pipeline.py`, `tool_execution.py` 删除对 `retrieval_signals` 引用
- `generation.py` 的 `groundedness_flag/insufficient_evidence_flag` 写入 `ToolResult.output`
- 删除 14 个废弃字段的 **写入路径**（此时调用方已全部迁移）
- 保留 14 个字段的 **定义**（旧 checkpoint 加载时仍可读，`_migrate_legacy_state` 转换后丢弃）
- 测试：formatter 调度、fallback、RAG tool formatter 端到端、ContextBuilder 输出一致性
- **不修改 allowlist（保留嵌套类型序列化支持）**

### 6.3 PR3：pending 单轨 + 最终清理

**原则：双轨合并、删除旧缓存、compat 层设退出 deadline。**

- 新版 `PendingToolCall`（plan + status + approval_request_id + operation_id）
- 合并 `pending_tool_calls` 双轨：
  - 删除旧 `PendingToolCall`（PR0 版本）
  - 删除旧 `ToolCallPlan` 作 pending 容器的用法
  - `LoopState.pending_tool_calls` 只保留新版 `PendingToolCall`
- `loop_messages` 改为 provider 层 derived transcript：
  - 每轮调用前由 `messages + tool_call_ledger + tool_results` 重建
  - `tool_call_ledger` 保留所有 tool call 的原始 `ToolCallPlan`（参数、顺序），即使已完成/失败
  - `llm_providers.py` 负责重建逻辑，从 `tool_call_ledger` 读取原始参数，不再依赖 `arguments_digest`
  - 迁移时降级保留：加载旧 checkpoint 时标记 `_needs_transcript_rebuild = True`
- `tool_call_ledger` bounded 策略：
  - `max_entries=128`，超限时按 `(turn, sequence)` FIFO 删除
  - 清理条件：只有当该 tool call 已不再需要参与 native transcript 重建时才可清理
- 删除 `tool_result_store`
- 删除 14 个废弃字段的 **定义**（写入路径已在 PR2 删除）
- `_migrate_legacy_state()` 更新为最终版：
  - drop 可丢字段
  - 降级保留 `pending_loop_tool_calls` → 合并到新 `pending_tool_calls`
  - 降级保留 `loop_messages` → 标记重建
- `rag/agent/state.py` 加 `DeprecationWarning`（60 天日落）
- Checkpoint allowlist 清理：
  - 先做 roundtrip 测试，确认 `ToolResult.output` 序列化不需要这些 allowlist
  - 移除 `AnswerCitation`, `EvidenceItem`, `RetrievalSignals`, `ObservationBatch` 等
- 测试：pending 单轨 roundtrip、双轨合并逻辑、transcript 重建、allowlist roundtrip、兼容层 deprecation warning

## 7. 风险与缓解

| 风险 | 缓解 |
|------|------|
| 执行顺序破坏 runtime | 3-PR 拆分：先建新 → 迁移调用方 → 删旧；每个 PR 独立可验证 |
| 旧 checkpoint 丢失信息 | 显式"可丢/降级保留"规则；pending 参数和顺序降级保留 |
| `loop_messages` 丢失导致 transcript 断裂 | provider 重建（降级保留标记）；roundtrip 测试 |
| ContextBuilder 行为退化 | 通用 fallback 保留；formatter 注册可选 |
| allowlist 过早移除导致序列化失败 | 推迟到 PR3，先做 roundtrip 测试 |
| 外部依赖遗漏 | 每个 PR 前 grep 全量扫描 + CI 回归 |

## 8. 不复做的

- 不新增 `tool_outputs: dict[str, list[Any]]` — 这是用 Any 垃圾桶换 RAG 垃圾桶
- 不删除 `file_manifest` — 它是 run input metadata，不是工具输出语义
- `PendingToolCall` 不含 `attempt_count/result/error` — 保持 `tool_execution_records` 和 `tool_results` 为 canonical
- `ToolCallLedger` 不复用 `PendingToolCall` — 用专用的 `ToolCallLedgerEntry(plan, turn, sequence)`，避免和 pending 状态机混淆
- `stop_hook_feedback/warnings` 不进 `MemoryState` — 它们是 finish gate / loop 控制流
- `DeferredToolState` 不用 `list[dict[str, object]]` — typed all the way
- `MemoryState.memory_index` 不保留裸 `str` — 用 `PersistentMemorySnapshot` 做 bounded snapshot

## 相关文件

- 当前状态定义: `rag/agent/loop/state.py`
- ContextBuilder: `rag/agent/memory/injector.py`
- ContextAssembler: `rag/agent/core/llm_context.py`
- 工具输出模型: `rag/agent/core/observations.py`
- Checkpoint 管理: `rag/agent/core/checkpointing.py`
- 兼容层: `rag/agent/state.py`
- Prompt 构建: `rag/agent/core/llm_prompts.py`
- LLM Provider: `rag/agent/core/llm_providers.py`
- 工具执行: `rag/agent/core/tool_execution.py`
- Agent 服务: `rag/agent/service.py`
- RAG 查询管线: `rag/query_pipeline.py`
- RAG 生成: `rag/providers/generation.py`
- 诊断记忆: [[Agent Architecture Redesign Diagnosis]]
