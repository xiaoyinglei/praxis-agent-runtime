# Praxis 故障排查

> 从 [README.md](../README.md) 拆分出来。常见问题和处理顺序。

| 现象 | 主要原因 | 处理 |
| --- | --- | --- |
| `Component backend requires a DSN/URI` | `--vector-backend milvus` 但没有传 DSN，或 `VECTOR_DSN` 为空 | `echo "$VECTOR_DSN"`，命令中补 `--vector-dsn "$VECTOR_DSN"` |
| Milvus `vector dimension mismatch` | 查询 embedding 和入库 embedding 不是同一模型/维度 | 换新的 `VECTOR_PREFIX` 和 `STORAGE_ROOT` 重新入库，不要混用旧 collection |
| `Embedding service health check failed: Connection refused` | embedding 服务没启动或端口不对 | `curl http://127.0.0.1:9090/health`，失败就重启 `rag_embedding_9090` |
| `Rerank service health check failed` | 查询要求 rerank，但 rerank 服务没启动；或入库阶段误带 rerank | 入库前 `unset RAG_RERANK_SERVICE_URL`；关闭 rerank 查询时传 `--reranker-model none` |
| `document summary generation returned empty output` | 本地 Qwen3 thinking 未关闭，或生成服务异常 | 用 `enable_thinking=false` 重启 Qwen；或改回默认 Mimo 云模型 |
| 入库 10 分钟进度条不动 | 模型首次加载、Excel 解析大表、embedding batch 太大、内存压力高 | 先 `curl` 检查服务；用 `--batch-size 1 --embedding-batch-size 1`；Excel 用 `scripts/diagnose_ingest_timing.py` 定位 |
| DOCX DrawingML / VML 图片警告 | 缺少 LibreOffice 转换器或文档内图片引用缺失 | 正文通常仍可入库；如果必须抽图，安装 LibreOffice 并设置 `DOCLING_LIBREOFFICE_CMD` |
| RAG 表格答案没读到明显数字 | 没走 asset profile、没触发 compute、或问题口径不清 | 用 `--retrieval-profile asset --json` 看 `context.evidence` 和 diagnostics |
| Agent 没有工具结果 | 模型没有产生工具调用，或 deferred tool 未激活 | 用 `--verbose` 看可见工具；必要时让任务明确先 `tool_search` 或先列 workspace 文件 |
| Agent 表格问题乱选 sheet/产品/口径 | 用户问题有歧义，或 Agent 未做结构探测 | 正确行为是列候选或请求澄清；不要用硬编码业务关键词修某张表 |
