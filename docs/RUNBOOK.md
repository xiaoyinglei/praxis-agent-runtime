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

如果要临时走云端模型，再准备 `.env`；默认本地 Qwen 不需要 API key：

```bash
cat > .env <<'EOF'
MIMO_API_KEY=your_mimo_key
DEEPSEEK_API_KEY=your_deepseek_key_optional
EOF
```

确认基础设施：

```bash
lsof -nP -iTCP:19530 -sTCP:LISTEN
lsof -nP -iTCP:5432 -sTCP:LISTEN
lsof -nP -iTCP:6379 -sTCP:LISTEN
```

默认本地端口：

| 服务 | 端口 | 说明 |
| --- | ---: | --- |
| Milvus | `19530` | 向量索引 |
| Milvus Web/metrics | `9091` | 已被 Milvus docker 占用，不要给 rerank 用 |
| Postgres | `5432` | metadata |
| Redis | `6379` | cache |
| Optional local Qwen generation service | `8080` | 本地 OpenAI-compatible chat |
| Embedding service | `9090` | `mlx-community/Qwen3-Embedding-4B-4bit-DWQ` |
| Rerank service | `9092` | `BAAI/bge-reranker-v2-m3` |

## 模型服务管理

当前默认模型配置在 `configs/models.yaml`：

| 能力 | 默认别名 | 实际模型 / 服务 |
| --- | --- | --- |
| 生成 / 摘要 / Agent tool decision | `qwen3_8b_mlx_4bit` | `mlx-community/Qwen3-8B-4bit`，OpenAI-compatible，`127.0.0.1:8080` |
| Embedding | `qwen3_embedding_4b_4bit_dwq` | `mlx-community/Qwen3-Embedding-4B-4bit-DWQ`，HTTP service，`127.0.0.1:9090` |
| Rerank | `bge_reranker_v2_m3` | `BAAI/bge-reranker-v2-m3`，HTTP service，`127.0.0.1:9092` |

内存策略：

- 默认 chat 走本地 Qwen 服务。`agent run --model qwen3_8b_mlx_4bit`
  会先检查 `runtime.health_url`，未启动时按 `runtime.launch_command`
  自动拉起 `127.0.0.1:8080` 的 OpenAI-compatible server。
- 入库和查询需要 embedding；建议启动 embedding HTTP 服务，避免每条命令重复加载模型。
- rerank 是可选服务，默认省内存时关闭。
- 切换 embedding 模型后必须换新的 Milvus collection prefix，旧向量不能混用。
- chat 模型的当前选择是 Agent session state，不是 `configs/models.yaml`
  的全局改写。
- 指定什么 chat 模型就必须用什么模型；不会 silent fallback，不会自动换端口，也不会自动杀已有进程。如果 `8080` 已经跑着别的模型，会报 endpoint conflict。

查看和切换当前 Agent 模型 session：

```bash
uv run agent model list
uv run agent model current
uv run agent model switch mimo_cloud
```

`agent model switch` 写入 `.praxis/` 目录中的 `model_session.json`。临时只跑一次其他模型时，用
`agent run --model mimo_cloud ...`，不要改 `configs/models.yaml`。

早期版本写在 `.rag/` 中的 `agent_checkpoints.sqlite` 和
`agent_model_session.json` 不迁移、不读取、也不删除；新的运行从 `.praxis/`
中的全新状态开始，RAG 知识数据则继续留在 `.rag/`。

先检查是否已经有同模型服务，避免重复常驻占内存：

```bash
ps aux | rg -i 'embedding-service|rerank-service|Qwen3-Embedding|bge-reranker|mlx_lm|vllm|ollama|uvicorn' \
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
  --model mlx-community/Qwen3-Embedding-4B-4bit-DWQ \
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
  --model BAAI/bge-reranker-v2-m3 \
  --port 9092 \
  --batch-size 4 \
  --max-length 1024
'
```

可选：手动预热默认本地 Qwen chat 服务。通常不需要；Agent 会按
`configs/models.yaml` 的 runtime 配置自动启动。手动预热适合提前加载模型、
减少第一次请求等待：

```bash
screen -S rag_qwen_8080 -X quit >/dev/null 2>&1 || true
screen -dmS rag_qwen_8080 zsh -lc '
cd /path/to/praxis-agent-runtime
uv run python -m mlx_lm.server \
  --model mlx-community/Qwen3-8B-4bit \
  --host 127.0.0.1 \
  --port 8080 \
  --chat-template-args '"'"'{"enable_thinking": false}'"'"'
'
```

健康检查：

```bash
curl -sS http://127.0.0.1:9090/health
curl -sS http://127.0.0.1:9092/health
curl -sS http://127.0.0.1:8080/v1/models
screen -ls
```

关闭服务：

```bash
screen -S rag_qwen_8080 -X quit >/dev/null 2>&1 || true
screen -S rag_embedding_9090 -X quit >/dev/null 2>&1 || true
screen -S rag_rerank_9092 -X quit >/dev/null 2>&1 || true
```

## 私有文档端到端运行手册

先准备 embedding 服务；rerank 默认不开，需要时再按"常用开关"打开。默认 chat 走 `configs/models.yaml` 中的 `qwen3_8b_mlx_4bit`，Agent 会按 runtime 配置自动检查和启动本地 chat 服务。

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
| 查看/切换当前 chat 模型 | `agent model list/current/switch <model_id>`；这是 session state，不改 YAML |
| 一次性指定模型 | 默认不需要；如要临时走云端可用 `--model mimo_cloud` |
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
