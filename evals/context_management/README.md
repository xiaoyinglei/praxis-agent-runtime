# 上下文管理与工具并行验收

本目录只保留可复跑脚本和[历史结果索引](evidence_index.json)。原始请求、报告、生成代码、
数据库和工作区留在本地，新的运行输出自动忽略，不继续堆入 Git。

## 当前证据与限制

以下均为 2026-09-28 的 DeepSeek Flash 记录，未因清理而重新判定：

| 场景 | 结果 | 限制 |
| --- | --- | --- |
| 两阶段真实代码任务，跨 SQLite 重开 | 两轮完成；外部功能检查、实际验证命令、三路并行读取通过；第二轮仅一次摘要 | 原总判定仍 false：第一轮未触发语义摘要，不满足原来每阶段必须压缩的条件 |
| 三轮连续压缩与重启 | 每轮一次摘要；最新决策、理由、未实施、待验证状态正确 | 合成历史回归，不替代真实代码任务 |
| 原归档检索场景 | 答对但未调用检索，原判定 false | 目标码仍在保留的前缀内，不能证明检索能力 |
| 隐藏目标检索场景 | 发送前确认目标码不在上下文；三次真实 read_context 调用后答对 | 单独的加强覆盖用例，不改写旧失败记录 |

真实代码任务预算固定为 14,000 字节、16 步、100,000 token/Turn，实际分别使用
46,387 和 60,015 token。五次代码任务共 66 个请求的 tokenizer 估算/usage 比例为
0.918–1.198，最大低估 270 token：仅是已测范围，不是服务端精确计数或未来误差上界。

核心修复：上下文压缩使用 `context_compaction`，默认继承 Agent 输入容量和普通摘要输出额度，
避免误用通用 16K 摘要输入上限。原失败现场同一历史的预留从 126,559 降到 43,531 token。
工具结果的可见前缀与原文归档分离，批次按整个请求容量计量；实现说明见
[RUNBOOK](../../docs/RUNBOOK.md)。

## 复跑

从仓库根目录执行。以下两条调用真实 Flash，读取本地 `.env` 凭证，产生费用；
脚本只发送合成任务工作区，不发送项目源码。不要用无关模型替换后仍称为同一验收。

```bash
uv run python -m evals.context_management.session_code_live
uv run python -m evals.context_management.functional_live --thinking disabled
```

仅检查隐藏目标检索：

```bash
uv run python -m evals.context_management.functional_live --thinking disabled --case archived_detail_recall_hidden
```

`audit_token_usage.py` 只读分析本地运行目录的 `session.db` 和 `report.json`。
`replay_summary_budget.py` 在数据库备份上试算并阻断模型派发；`--legacy-summary-input`
复现旧摘要输入策略。数据库本来就未纳入 Git，这两个离线脚本需要保留的本地数据库或新运行结果。

## 历史追溯

成功与失败报告都进入索引，包含原始路径、SHA-256、判据和用量摘要。未经记录的失败不能被
一次成功覆盖。完整历史文件保留在提交
[`2660257`](https://github.com/xiaoyinglei/praxis-agent-runtime/tree/2660257ae9f479929e045f1ffb2d8b15a9042fa6/evals/context_management)，
可通过 `git show 2660257:<索引中的路径>` 读取。
此前的设计比对和失败分析也在该提交的 `docs/design/`，不再占用当前文档入口。

参考源码的固定版本为 Codex `ddf04ad26789d040f9ef6a96736f76602e35a6cc` 与
Claude 镜像 `276b8e6a0160939c74f04074a5e10ec1ba76982b`；后者来源/发布版本未经官方核实，
不把缺失实现或项目自主策略冒充官方机制。完整比对见该提交的
`docs/design/context_parallel_source_audit_2026-09-21.md`。
