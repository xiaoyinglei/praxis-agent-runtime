# Praxis 历史检索评测

> 从 [README.md](../README.md) 拆分出来。这是 RAG 子系统的历史基线快照，
> 不是当前 Praxis 模型质量门禁；当前证据见 [benchmark](benchmark.md)。

## 历史基线结果与实验配置

下面是之前已经跑过的基线结果，保留用于和当前 `configs/models.yaml` 默认链路做横向对比。这里的模型和存储配置是历史实验快照，不代表当前默认运行配置；切换 embedding 模型后必须重建对应索引。

### 公开数据：MedicalRetrieval mini

| 基线 | 向量后端 | embedding | rerank | Recall@10 | MRR@10 | NDCG@10 | avg_latency_ms |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `BAAI/bge-m3 + sqlite` | SQLite | `BAAI/bge-m3` | on | 0.776667 | 0.690972 | 0.712199 | 2472.225 |
| `BAAI/bge-m3 + milvus` | Milvus | `BAAI/bge-m3` | on | 0.670000 | 0.588259 | 0.608173 | 563.793 |
| `qwen3-embedding:8b + milvus` | Milvus | `qwen3-embedding:8b` | on | 0.820000 | 0.705854 | 0.733644 | 695.559 |

公开 qwen8b baseline 当时的复现实验配置：

| 项 | 值 |
| --- | --- |
| storage root | `data/benchmarks/medical_retrieval/index/mini-milvus-qwen8b-v1` |
| Milvus collection prefix | `medical_retrieval_mini_qwen8b_v1` |
| vector backend | `milvus` |
| vector dsn | `http://127.0.0.1:19530` |
| embedding provider / model | `ollama` / `qwen3-embedding:8b` |
| summary provider / model | `local-hf` / `Qwen/Qwen3-8B-MLX-4bit` |
| chunk window | `480` tokens, overlap `64` tokens |
| retrieval profile | `auto` |
| evaluate top_k / evidence_top_k | `10` / `20` |
| rerank provider / model | `local-bge` / `BAAI/bge-reranker-v2-m3` |

### 私有数据：公司制度/销售中心资料

黄金测试集：

- query_count：`329`
- 题型分布：
  - `single_section_fact`：163
  - `rule_condition`：105
  - `process_step`：61

整体结果：

| 配置 | top_k | rerank | doc_mrr | section_mrr | doc_hit@1 | doc_hit@3 | doc_hit@5 | doc_hit@10 | section_hit@1 | section_hit@3 | section_hit@5 | section_hit@10 |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sqlite rerank | 10 | on | 0.9300 | 0.7588 | 0.8967 | 0.9574 | 0.9757 | 0.9818 | 0.6535 | 0.8602 | 0.8875 | 0.9027 |
| sqlite no rerank | 10 | off | 0.9127 | 0.7200 | 0.8663 | 0.9574 | 0.9757 | 0.9818 | 0.6049 | 0.8359 | 0.8723 | 0.9027 |
| sqlite no rerank | 20 | off | 0.9127 | 0.7200 | 0.8663 | 0.9574 | 0.9757 | 0.9818 | 0.6049 | 0.8359 | 0.8723 | 0.9027 |

按题型结果，开启 rerank：

| 题型 | query_count | doc_mrr | section_mrr | doc_hit@1 | doc_hit@3 | doc_hit@10 | section_hit@1 | section_hit@3 | section_hit@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `process_step` | 61 | 0.9727 | 0.7865 | 0.9508 | 1.0000 | 1.0000 | 0.7049 | 0.8689 | 0.9344 |
| `rule_condition` | 105 | 0.9221 | 0.7929 | 0.8857 | 0.9524 | 0.9810 | 0.6762 | 0.8952 | 0.9429 |
| `single_section_fact` | 163 | 0.9192 | 0.7265 | 0.8834 | 0.9448 | 0.9755 | 0.6196 | 0.8344 | 0.8650 |

结论：

- 文档级命中稳定，`doc_hit@10` 达到 0.9818。
- section 级 top10 命中达到 0.9027。
- rerank 对 section 级排序有明显帮助，`section_mrr` 从 0.7200 提升到 0.7588。
- `top_k=20` 对当前私有数据没有带来额外 top20 命中，后续优化重点在候选生成和 section top1 排序。

## 已验证端到端结果

下面是历史真实链路验证，保留用于说明系统能力边界。当前模型、索引和评测结果以实际命令输出为准。

- Postgres schema：`rag_e2e_20260516_150131`
- Milvus collection prefix：`rag_e2e_20260516_150131`
- Milvus collections：
  - `rag_e2e_20260516_150131__doc_summary__default`
  - `rag_e2e_20260516_150131__section_summary__default`
  - `rag_e2e_20260516_150131__asset_summary__default`
- 表格对象：`data/e2e_agent_pq_milvus/20260516_150131/objects/*.parquet`
- `sqlite_vector_index_used: false`

验证问题：

| 类型 | 问题 | 结果 |
| --- | --- | --- |
| RAG 制度问答 | 单笔国内差旅报销金额超过 12000 元需要谁审批？ | 命中制度原文，回答为业务线 VP |
| RAG SLA 问答 | P0 客户生产故障的首次响应目标和恢复目标分别是多少？ | 命中 `15 分钟` 和 `2 小时` |
| 表格计算 | 请计算华东区域 Q1 的开票量合计是多少？ | 触发 DuckDB SQL，返回 `375` |

表格计算执行证据：

```sql
SELECT SUM("开票量") FROM sheet WHERE "区域"='华东' AND "季度"='Q1'
```

返回：

```text
TABLE_COMPUTE_RESULT
sum("开票量") = 375
```
