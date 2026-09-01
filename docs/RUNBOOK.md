# Praxis 运行手册

> 从 [README.md](../README.md) 拆分出来。安装、服务管理、端到端运行命令。
> `praxis-agent-runtime` 只是本地构建的 distribution metadata；当前未发布到 PyPI，
> 所有命令均从 source checkout 运行。

## 安装

先进入已克隆的仓库，再安装依赖：

```bash
cd /path/to/praxis-agent-runtime
uv sync
```

## 接入和切换 chat 模型

无需手工编辑 `configs/models.yaml`。该文件中的内置 alias 是只读层；
`agent model add`、`update`、`remove` 管理版本化的用户注册表，并通过
compare-and-swap 防止并发覆盖。
注册表和 Turn binding 只保存 credential environment variable 的名字，不保存解析后的值。

### 接口要求

一个完整可用的 Agent chat endpoint 必须满足：

- 提供 OpenAI-compatible 模型发现和 chat 接口，并返回配置中声明的 model identity；
- 流式响应至少包含一个真实 text delta，随后给出 authoritative completion；
- 声明支持工具时，能按指定 JSON Schema 返回一个 forced tool call；probe 只校验，绝不执行它；
- 声明支持结构化响应时，能返回可验证的 structured output；
- 遵守配置的 timeout，并允许取消仍在进行的 stream。

`connectivity` 只检查连接、鉴权和模型身份；`stream` 继续检查真实流式文本和完成信号；
`full` 再按显式 capability flags 检查 tool call 与 structured output。探测证据是当次结果，
不会反向猜测或改写 capability flags。

### 初始化与注册

首次启动新 Turn 前初始化本地 HMAC trust domain；命令只显示 domain/key id，不显示 key material：

```bash
uv run agent model trust init
uv run agent model trust status
uv run agent model list --source
uv run agent model current
export MODEL_ALIAS=my-model-alias
export PROVIDER_MODEL_ID=provider-model-id
export PROVIDER_BASE_URL=https://provider.example/v1
export PROVIDER_CREDENTIAL_ENV=MY_PROVIDER_TOKEN

uv run agent model show "$MODEL_ALIAS"
uv run agent model add "$MODEL_ALIAS" \
  --provider openai_compatible \
  --provider-model "$PROVIDER_MODEL_ID" \
  --base-url "$PROVIDER_BASE_URL" \
  --api-key-env "$PROVIDER_CREDENTIAL_ENV"

uv run agent model probe "$MODEL_ALIAS" --level full
uv run agent model update "$MODEL_ALIAS" --timeout-seconds 90
uv run agent model remove "$MODEL_ALIAS"
```

`add/update` 的事务顺序是：严格解析和规范化 → 读取预期 registry revision → full probe
→ CAS commit。失败或取消发生在 commit 前，因此注册表不变。高级字段（例如无 shell 的
launch argv）使用 `--from <one-model.yaml>`；离线登记必须显式写 `--skip-probe`，输出会标记
`unverified`。`update --unset <nullable-field>` 清除支持的可空字段；规范化 no-op 不 probe、
不写文件、也不增加 revision。

### 切换与持久化语义

查看和切换当前 Agent model session：

```bash
uv run agent model list --source
uv run agent model current
uv run agent model switch "$MODEL_ALIAS"
```

`agent model switch` 只更新 `.praxis/model_session.json`，不改注册表定义。

交互式 `agent chat` 复用同一个 catalog、policy 和 session state，不维护第二套
alias 或路由：

```text
uv run agent chat
> /model
当前模型: current-alias
可用模型:
* current-alias  ...
  another-alias  ...
切换: /model <alias>
> /model another-alias
已切换模型: another-alias
```

`agent chat` 使用 Unicode-aware Composer，中文和 emoji 的退格、Delete、光标移动按
完整字符处理。工具与命令按同一个 Item 生命周期显示；长输出默认保留头尾并明确显示
省略行数，不再在字典或句子中间硬切。可在两个 Turn 之间输入 `/verbose`，让后续工具
结果完整展开。若工具或 ACI 本身已经丢弃超预算内容，CLI 会另行警告；这与 UI 折叠
不同，verbose 不能恢复上游未保留的数据。

也可写 `/model switch <alias>`；`/model current` 只看当前详情，`/model list`
只列 catalog。成功切换后，当前聊天的下一条消息继续使用原来的
`previous_turn_id` 历史，但新 Turn 会绑定新 alias，因此不需要退出、重启或
`/new`。输入不存在或被 policy 拒绝的 alias 时，CLI 会显示错误和所有可用
alias，原选择保持不变，而且校验阶段不会发起 provider 请求。

session alias 是未来 Turn 的可变偏好；每个已创建 Turn 则持久化经过 HMAC 认证、
content-addressed archive 校验的完整不可变定义。更新或删除 alias 只影响未来 Turn。
`agent resume` 恢复同一个暂停/中断 Turn，必须使用其原定义；`previous_turn_id` 创建
新 Turn，才会读取当前 session 选择。历史 replay 只读 durable history，不要求 provider 在线。

早期版本写在 `.rag/` 中的 `agent_checkpoints.sqlite` 和
`agent_model_session.json` 不迁移、不读取、也不删除；新的运行从 `.praxis/`
中的全新状态开始，RAG 知识数据则继续留在 `.rag/`。

## RAG 服务准备

下面是知识库 embedding/rerank 的运维示例，与 chat alias 注册表相互独立。

先检查是否已经有同模型服务，避免重复常驻占内存：

```bash
ps aux | rg -i 'embedding-service|rerank-service|Qwen3-Embedding|Qwen3-Reranker|mlx_lm|vllm|ollama|uvicorn' \
  | rg -v 'rg -i|exec_command'

lsof -nP -iTCP -sTCP:LISTEN \
  | rg ':(8080|8081|8000|8001|9090|9091|9092|11434|19530|5432|6379)\b' || true
```

启动 embedding 服务。内存紧张时用 `--batch-size 1`，更稳；内存充足时可以调到 `2/4/8`：

```bash
screen -S rag_embedding_9090 -X quit >/dev/null 2>&1 || true
screen -dmS rag_embedding_9090 zsh -lc '
cd /path/to/praxis-agent-runtime
uv run rag embedding-service \
  --model mlx-community/Qwen3-Embedding-8B-4bit-DWQ \
  --port 9090 \
  --batch-size 1
'

export RAG_EMBEDDING_SERVICE_URL="http://127.0.0.1:9090"
```

rerank 是可选服务。需要重排时再启动。注意 `9091` 被 Milvus 占用，rerank 用 `9092`：

```bash
screen -S rag_rerank_9092 -X quit >/dev/null 2>&1 || true
screen -dmS rag_rerank_9092 zsh -lc '
cd /path/to/praxis-agent-runtime
uv run rag rerank-service \
  --model Qwen/Qwen3-Reranker-4B \
  --port 9092 \
  --batch-size 4 \
  --max-length 1024
'
```

健康检查：

```bash
curl -sS http://127.0.0.1:9090/health
curl -sS http://127.0.0.1:9092/health
screen -ls
```

关闭服务：

```bash
screen -S rag_embedding_9090 -X quit >/dev/null 2>&1 || true
screen -S rag_rerank_9092 -X quit >/dev/null 2>&1 || true
```

## 私有文档端到端运行手册

先准备 embedding 服务；rerank 默认不开，需要时再按"常用开关"打开。chat 使用
当前 session 选中的 alias；先用 `agent model current` 确认其 endpoint 与 credential
environment variable 已可用。

### 统一变量

入库和 Agent 的 `RAGKnowledgeConfig` 必须指向同一套 `STORAGE_ROOT / VECTOR_PREFIX`。Milvus 连接信息只通过 `AGENT_VECTOR_DSN` 注入 Agent，不写入 knowledge config 或 Turn binding。切换 embedding 模型或想重建干净索引时，换新的 `STORAGE_ROOT` 和 `VECTOR_PREFIX`。

```bash
cd /path/to/praxis-agent-runtime

# 数据位置：按实际数据改这两个变量。
export INPUT_PATH="/absolute/path/to/one-file.docx"
export INPUT_DIR="/absolute/path/to/private-docs"

# 索引位置：同一批入库和 knowledge config 必须保持一致。
export STORAGE_ROOT="data/indexes/private_docs_v1"
export VECTOR_DSN="http://127.0.0.1:19530"
export VECTOR_PREFIX="private_docs_v1"
export AGENT_VECTOR_DSN="$VECTOR_DSN"
export AGENT_KNOWLEDGE_CONFIG="$STORAGE_ROOT/agent-knowledge.yaml"

cat > "$AGENT_KNOWLEDGE_CONFIG" <<EOF
storage_root: $STORAGE_ROOT
vector_backend: milvus
vector_collection_prefix: $VECTOR_PREFIX
EOF

# 复用常驻 embedding 服务，避免每条命令重新加载 embedding 模型。
export RAG_EMBEDDING_SERVICE_URL="http://127.0.0.1:9090"

# 默认省内存：不开 rerank。
unset RAG_RERANK_SERVICE_URL
```

### 入库

入库会做：解析文档 -> 切分 section / asset -> 生成摘要 -> embedding -> 写 Milvus。入库阶段不需要 rerank。

单个文档：

```bash
unset RAG_RERANK_SERVICE_URL

uv run python scripts/ingest_private_documents.py \
  --input "$INPUT_PATH" \
  --storage-root "$STORAGE_ROOT" \
  --batch-size 1 \
  --embedding-batch-size 1 \
  --strict-summary-generation \
  --vector-backend milvus \
  --vector-dsn "$VECTOR_DSN" \
  --vector-collection-prefix "$VECTOR_PREFIX" \
  --output "$STORAGE_ROOT/ingest_result.json"
```

批量目录：

```bash
unset RAG_RERANK_SERVICE_URL

uv run python scripts/ingest_private_documents.py \
  --input "$INPUT_DIR" \
  --storage-root "$STORAGE_ROOT" \
  --batch-size 1 \
  --embedding-batch-size 1 \
  --strict-summary-generation \
  --vector-backend milvus \
  --vector-dsn "$VECTOR_DSN" \
  --vector-collection-prefix "$VECTOR_PREFIX" \
  --output "$STORAGE_ROOT/ingest_result.json"
```

入库结果检查：

```bash
cat "$STORAGE_ROOT/ingest_result.json"

uv run python - <<'PY'
import os
from pymilvus import connections, utility

prefix = os.environ["VECTOR_PREFIX"]
connections.connect(alias="check", uri=os.environ["VECTOR_DSN"])
try:
    print([name for name in utility.list_collections(using="check") if name.startswith(prefix)])
finally:
    connections.disconnect("check")
PY
```

### Agent 查询已入库知识

日常查询只调用 Agent。不要手动判断 retrieval profile；传入 `--knowledge-config "$AGENT_KNOWLEDGE_CONFIG"` 后，Agent 会把 `search_knowledge` 暴露给模型，由模型自己决定是否调用。没有该配置时，环境变量不会暗中启用 RAG。

普通制度/流程问答：

```bash
unset RAG_RERANK_SERVICE_URL

uv run agent run \
  "单笔国内差旅报销金额超过 12000 元需要谁审批？请给出处" \
  --knowledge-config "$AGENT_KNOWLEDGE_CONFIG" \
  --verbose
```

Excel / 表格 / PPT 表格 / 图片 OCR 这类已入库资产问题也直接问 Agent：

```bash
unset RAG_RERANK_SERVICE_URL

uv run agent run \
  "日提货总量是多少？请检查相关表格并给出处" \
  --knowledge-config "$AGENT_KNOWLEDGE_CONFIG" \
  --verbose
```

需要看底层 evidence / diagnostics 时，才临时用 `rag query --json` 做检索诊断；这不是日常用户入口。JSON 重点字段：

- `answer.answer_text`
- `answer.answer_sections[].evidence_ids`
- `answer.citations`
- `context.evidence`
- `retrieval_diagnostics.operator_plan`
- `retrieval_diagnostics.rerank_skipped`
- `generation_attempts`

### 批量检索评测

适合已有 golden JSONL 时看 doc / section 命中、MRR 和 misses。不开 rerank：

```bash
unset RAG_RERANK_SERVICE_URL

uv run python scripts/evaluate_private_retrieval.py \
  --golden-path data/eval_private/golden_eval_dataset.jsonl \
  --storage-root "$STORAGE_ROOT" \
  --retrieval-profile auto \
  --top-k 10 \
  --retrieval-pool-k 20 \
  --neighbor-radius 1 \
  --no-rerank \
  --vector-backend milvus \
  --vector-dsn "$VECTOR_DSN" \
  --vector-collection-prefix "$VECTOR_PREFIX" \
  --output "$STORAGE_ROOT/retrieval_eval_no_rerank.json" \
  --misses-output "$STORAGE_ROOT/retrieval_misses_no_rerank.jsonl"
```

开启 rerank 时，先启动 rerank 服务，再改两处：设置 `RAG_RERANK_SERVICE_URL`，把 `--no-rerank` 换成 `--rerank`。

### Agent 测试

`agent run` 默认是纯 Agent + workspace tools，不会因为当前环境已有 `STORAGE_ROOT`、`AGENT_VECTOR_DSN` 或 embedding/reranker 配置就自动启动 RAG。需要知识库证据时显式传入一个 `--knowledge-config` JSON/YAML 文件；RAG 会作为 lazy knowledge provider 注册，并在模型首次调用 `search_knowledge` 时初始化。

普通制度问答：

```bash
unset RAG_RERANK_SERVICE_URL

uv run agent run \
  "单笔国内差旅报销金额超过 12000 元需要谁审批？请给出处" \
  --knowledge-config "$AGENT_KNOWLEDGE_CONFIG" \
  --verbose
```

已入库资产问题：

```bash
uv run agent run \
  "日提货总量是多少？请检查相关表格并给出处" \
  --knowledge-config "$AGENT_KNOWLEDGE_CONFIG" \
  --verbose
```

本地文件直接分析，不需要先入 RAG：

```bash
uv run agent run \
  "读取这个 Excel，汇总关键指标，并写一个简短摘要" \
  --file "/absolute/path/to/report.xlsx" \
  --verbose
```

期望工具链：

```text
本地文件：
  list_files / search_text -> read_file -> final answer
  必要时由 apply_patch 或受限 run_command 处理或写回 workspace

知识库问题：
  search_knowledge -> final answer with evidence / citations
```

交互式：

```bash
uv run agent chat
```

### 常用开关

| 需求 | 做法 |
| --- | --- |
| 关闭 rerank 省内存 | `unset RAG_RERANK_SERVICE_URL`；只在显式 `--knowledge-config` 的知识库路径中相关 |
| 开启 HTTP rerank | 启动 `rag_rerank_9092`，`export RAG_RERANK_SERVICE_URL=http://127.0.0.1:9092` |
| 看 evidence / diagnostics | 先用 `agent run --verbose`；需要检索调试时才用 `rag query --json` |
| 普通制度问答 | 直接问 `agent run` |
| 已入库的文档证据问题 | `agent run ... --knowledge-config <path>`，模型会按需调用 `search_knowledge` |
| Agent 直接读本地文件 | `agent run ... --file "/path/to/file.xlsx"` |
| 查看/切换当前 chat 模型 | chat 外用 `agent model list --source`、`current`、`switch <alias>`；chat 内用 `/model` 与 `/model <alias>`；都是 session state，不改注册表定义 |
| 一次性指定模型 | `agent run --model <alias> ...`，只影响该次新 Turn |
| 恢复常驻 embedding | `export RAG_EMBEDDING_SERVICE_URL=http://127.0.0.1:9090` |

### 快速 smoke 测试

```bash
export STORAGE_ROOT="data/smoke_milvus"
export VECTOR_PREFIX="smoke_milvus_v1"
export AGENT_KNOWLEDGE_CONFIG="$STORAGE_ROOT/agent-knowledge.yaml"

cat > "$AGENT_KNOWLEDGE_CONFIG" <<EOF
storage_root: $STORAGE_ROOT
vector_backend: milvus
vector_collection_prefix: $VECTOR_PREFIX
EOF

uv run rag ingest \
  --storage-root "$STORAGE_ROOT" \
  --vector-backend milvus \
  --vector-dsn "$VECTOR_DSN" \
  --vector-collection-prefix "$VECTOR_PREFIX" \
  --source-type plain_text \
  --location memory://smoke/support-sla \
  --title "示例客服 SLA Smoke" \
  --owner smoke \
  --content "示例客服 SLA：P1 工单首次响应目标为 30 分钟，解决目标为 4 小时。"

uv run agent run \
  "P1 工单首次响应目标是多少？请给出处" \
  --knowledge-config "$AGENT_KNOWLEDGE_CONFIG" \
  --verbose
```

说明：CLI 默认 metadata 仍可用本地 metadata repo 做快速验证。正式端到端可以使用下面的 `Postgres + parquet object + Milvus` runtime 配置。

## 真实 Postgres + Milvus 端到端

用正式链路时，显式构造 `StorageConfig`：

```python
from pathlib import Path

from rag import AssemblyRequest, CapabilityRequirements, RAGRuntime, StorageComponentConfig, StorageConfig
from rag.ingest.pipeline import IngestRequest
from rag.models.assembly_adapter import to_assembly_overrides
from rag.models.runtime import resolve_runtime_config
from rag.retrieval.models import QueryOptions
from rag.schema.core import SourceType
from rag.utils.text import load_env_file

load_env_file(".env")

run_id = "manual_run_v1"
root = Path("data/manual_pq_milvus") / run_id
schema = f"rag_{run_id}"
collection_prefix = f"rag_{run_id}"

cfg = resolve_runtime_config()
storage = StorageConfig(
    backend="postgres",
    root=root,
    metadata=StorageComponentConfig(
        backend="postgres",
        dsn="postgresql://user:password@127.0.0.1:5432/postgres",
        namespace=schema,
    ),
    vectors=StorageComponentConfig(
        backend="milvus",
        dsn="http://127.0.0.1:19530",
        collection=collection_prefix,
    ),
    cache=StorageComponentConfig(
        backend="redis",
        dsn="redis://127.0.0.1:6379/0",
        namespace=collection_prefix,
    ),
    object_store=StorageComponentConfig(backend="local"),
)

request = AssemblyRequest(
    requirements=CapabilityRequirements(
        require_chat=True,
        require_rerank=False,
        allow_degraded=False,
    ),
    overrides=to_assembly_overrides(cfg),
)

with RAGRuntime.from_request(storage=storage, request=request) as runtime:
    runtime.insert(
        IngestRequest(
            location="memory://demo/support-sla",
            source_type=SourceType.PLAIN_TEXT,
            owner="demo",
            title="示例客服 SLA",
            content_text="示例客服 SLA：P1 工单首次响应目标为 30 分钟，解决目标为 4 小时。",
        )
    )

    runtime.insert(
        IngestRequest(
            location="/absolute/path/to/sample_sales.xlsx",
            source_type=SourceType.XLSX,
            owner="demo",
            title="示例销售明细",
            file_path=Path("/absolute/path/to/sample_sales.xlsx"),
        )
    )

    result = runtime.query_public(
        "请计算示例销售明细中华北区域 2026-05 的销售额合计是多少？",
        options=QueryOptions(retrieval_profile="asset", top_k=6, retrieval_pool_k=12),
    )
    print(result.answer.answer_text)
```

检查真实后端：

```bash
uv run python - <<'PY'
from pymilvus import connections, utility

prefix = "rag_manual_run_v1"
connections.connect(alias="check", uri="http://127.0.0.1:19530")
try:
    print([name for name in utility.list_collections(using="check") if name.startswith(prefix)])
finally:
    connections.disconnect("check")
PY
```

## 运行注意事项

- 入库和查询必须使用同一个 embedding space；切换 embedding 模型后必须重建 Milvus collection。
- 每次真实实验建议使用新的 `STORAGE_ROOT` 和 Milvus collection prefix，避免不同 embedding 维度或旧 schema 污染结果。
- `9091` 被 Milvus 占用，rerank 服务使用 `9092`。
- RAG 是 Agent 的一个显式 knowledge provider，不是所有文件任务的默认入口。本地文件分析优先用 `--file` 和 canonical workspace tools；需要知识库证据时再加 `--knowledge-config`。
- 对表格真实值问题，不要信任 `sample_rows`；正确路径是资产 inspect/read/analyze 或本地 Python 计算。
- OpenAI-compatible chat provider 的结构化输出能力依赖后端；降级必须可见，不能静默吞掉失败。
- 批量入库脚本支持 `--summary-provider none`，可跳过 LLM 摘要生成，直接用原文进入 summary index；质量会低于严格摘要链路。
- Agent CLI 只有一条 generic Agent runtime 装配链，不再接受 `--agent` 或历史角色名。

DOCX 图形转换可选配置：

```bash
export DOCLING_LIBREOFFICE_CMD="/Applications/LibreOffice.app/Contents/MacOS/soffice"
```

Excel 入库耗时诊断：

```bash
uv run python scripts/diagnose_ingest_timing.py "$INPUT_PATH"
```
