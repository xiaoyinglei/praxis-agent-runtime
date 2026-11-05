# F 实施与验证记录

沿用已批准的 2026-11-03 完整 Code Agent 设计第 9/10 节，不更改 Thread/Turn 与 E 证据合同。正常入口 `/Users/leixiaoying/LLM/RAG学习`，实现/命令 workdir `/private/tmp/praxis-code-agent-f`，分支 `codex/code-agent-f`，基线 `89f82f5d9dab3379314220fef5b73906f600f95b`。保留原 outputs 和所有旧工作树；没有真实对外发送或发布授权。

- [x] canonical 渠道投影、原生认证、精确 binding/grant、inbox/收据/host command 同事务。
- [x] 缓存/claim/dispatch 当前授权复核，正文/附件/证据与控制权限分开。
- [x] durable outbox、有限退避、绝对期限、unknown/correlation/明确重试与撤销。
- [x] CLI/SDK/Webhook 到原宿主，GitHub 与飞书原生链，本机 ACK 与杀进程反例。
- [x] CI head/app/check 身份、持久轮询/预算、有限新 Turn、晚到旧 head 不通过。
- [x] push/PR/review 单项 intent、GitDelivery preview 与范围、未知结果远端核对。
- [x] 完整 Agent 回归已执行：4184 passed、93 skipped、2 failed；两失败在未改 main 复现。Ruff、mypy 292 源文件、6 条依赖合同与独立审查通过。本机验收与日志归档在 `artifacts/code-agent-f-2026-11-04/`（本地 ignored，未发布）。
- [ ] 真实 provider/云模型与 G 发布验收：不在本次具体外发授权范围内，保留正式交付门槛。

新 facts 使用 schema 8。旧库可只读审查或在本版本加法迁移；旧 writer 不能写入新库。发布级备份、恢复、降级支持矩阵属于 G，不能用本批 schema fence 代替。

独立子代理完成 provider/CI 初稿，主代理集成时修复 bot 正文丢失、CI fixture API、schema reader、排队撤销、旧 head 回滚和重复 journal 等问题。所有状态以当前源码和最终验证文件为准；本文件不宣称完整产品成熟。
