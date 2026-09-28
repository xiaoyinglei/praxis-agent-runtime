# 上下文真实模型验证

## 当前状态（2026-09-28）

已修复上下文压缩误用通用摘要 16K 输入上限的问题：压缩默认继承 Agent 输入容量，
保留原摘要输出额度及模型窗口、累计预算检查。旧配置无需迁移；任务预算与通过条件未改。
同一失败现场从 9 次摘要、预留 126,559 token，变成 1 次摘要加续跑、预留 43,531 token。
离线复算：[修复前](offline-summary-plan-before-20260928.json)、[修复后](offline-summary-plan-after-20260928.json)。
重现命令（不调用模型，只在数据库备份上试算）：

```bash
uv run python -m evals.context_management.replay_summary_budget evals/context_management/deepseek-session-code-20260928-132742 --legacy-summary-input --output /tmp/plan-before.json
uv run python -m evals.context_management.replay_summary_budget evals/context_management/deepseek-session-code-20260928-132742 --output /tmp/plan-after.json
```

| 证据 | 实际结果 | 边界 |
| --- | --- | --- |
| [两阶段真实代码任务](deepseek-session-code-20260928-133836/report.json) | 两轮均完成，外部功能检查、实际验证命令、三路并行、重启恢复均通过；第二轮仅一次摘要 | 原总判定仍 false：第一轮没有触发语义摘要。不修改此门槛 |
| [合成连续压缩与重启](deepseek-functional-20260928-134110-disabled/report.json) | 三轮各一次摘要，最新决策、理由、未实施及待验证状态正确；并行编辑场景通过 | 同报告旧归档场景答对但未调用检索，仍为 false，目标码在保留前缀中 |
| [新增隐藏目标检索](deepseek-functional-20260928-183217-disabled/report.json) | 首次请求发送前断言目标码缺失；三次真实 read_context 调用找回目标码，全部检查通过 | 单独加强覆盖的合成用例，不改写旧失败报告 |

仅使用 DeepSeek Flash，无本地模型推理。代码任务每轮仍为 14,000 字节、16 步、100,000 token；
第一阶段实际 46,387 token，第二阶段 60,015 token。没有凭运行状态冒充功能正确，旧失败样本保留。
[计量审计](offline-token-audit-20260928.json) 覆盖五次代码任务的 66 个请求：估算/usage 约
0.918–1.198，最大低估 270 token，仅代表已测范围。
详细依据与修改边界见[调用链复查第 5 节](../../docs/design/context_root_findings_2026-09-23.md)。

## 历史状态（2026-09-23）：完整验收未通过

下方 09-21 结果是历史合成场景记录，不代表最新 Session 代码任务交付。
`deepseek-session-code-20260923-232318` 第一阶段通过，重启后的第二阶段失败；
`deepseek-session-code-20260923-232844` 外部功能断言通过但 Turn 失败，且原报告
`model_ran_verification=true` 存在误判：成功的是 `ls -la`，`python3 verify.py` 实际失败。
现已修正 judge；原报告不改写，离线重判见 [offline-root-audit-20260923.json](offline-root-audit-20260923.json)。

根因与参考源码差异见 [调用链复查](../../docs/design/context_root_findings_2026-09-23.md)。
09-24 已停止生成 v5 隐式回放候选，原 5 项离线回归消除；旧格式读取兼容保留。
预算计数降级和低预算摘要准入已修复，见上述调用链复查第 4 节。
尚未重新通过真实任务验收，暂停新的付费请求；后续实测仅用 DeepSeek Flash。
14 KB 用例保留为压力测试，不以调整预算替代修复，也不单凭它代表正常窗口能力。

## DeepSeek 功能验收（2026-09-21）

官方 [更新日志](https://api-docs.deepseek.com/updates/) 确认最新 Flash 是 V4.1 Flash，
API ID 为 `deepseek-flash`。本次实际调用该 ID，没有切换项目默认主模型。

```bash
uv run python evals/context_management/functional_live.py --thinking disabled
uv run python evals/context_management/functional_live.py --thinking enabled
```

需要 `.env` 或环境变量中的 DeepSeek 凭证，会产生实际模型调用。输入为合成历史和临时工作区，
不会发送项目源码。每个场景经过真实 Gateway、Turn、Rollout、工具编排；报告由独立断言判定，
不是只检查 Turn 返回 completed。SQLite 和临时工作区保留用于追踪失败。

| 场景 | 非思考模式 | 思考模式 | 独立验收依据 |
| --- | --- | --- | --- |
| 三轮压缩，每轮关闭并重开 SQLite | 通过 | 通过 | TTL 17→23→31，原因同步更新，未迁移、未实施、待验证保持正确 |
| 归档中间细节回查 | 通过 | 通过 | 实际调用 read_context，准确找回 HX-7294-KAPPA |
| 并行读取、修改、回读 | 通过 | 通过 | 三个读取执行区间重叠，文件结果为 117，输入不变，写后读取 |

最终报告：[普通模式](deepseek-functional-20260921-150051-disabled/report.json)、
[思考模式](deepseek-functional-20260921-150050-enabled/report.json)。
两种模式均使用产品的 `coding_instructions`。历史报告早于 `145816-enabled` 的均为简化提示词；
现在仍可用 `--prompt minimal` 运行该压力对照。不能把简化入口当作完整产品行为。
最终历史任务每轮还要求最多 8 次摘要、4 次普通模型调用，过量工作即使答对也判失败。
这里以 5000 字节本地上限故意施压，
不等于 DeepSeek 的真实模型窗口大小。

失败记录同样保留，不能只看最后通过的样本：

- `090006-disabled`：摘要略超限制即失败；`090016-enabled`：思考消耗摘要输出额度。
- `090242-disabled`、`090243-enabled`：重复摘要原始归档，预算耗尽后抛异常。
- `090902-enabled`：结果虽正确，第一轮反复摘要 10 次，效率不合格。
- `091300-enabled`：把后续决策覆盖旧决策误认为未解决矛盾，反复检索后失败。
- `091637-enabled`：虽然回答正确，但一轮摘要 20 次，不满足后来补充的收敛要求。
- `091835-enabled`：历史任务预算耗尽；后续场景及 `091957-enabled` 出现连接中断，
  持久化为 outcome_unknown，不能当作功能通过。
- `145816-enabled`：原产品提示词下仍有一轮摘要 13 次。补充不可变历史证据复用规则后，
  `145930-enabled`、`145931-disabled` 以及最终两份报告均完成全部场景；最终报告额外执行收敛断言。

对应修复：候选超限继续缩短、摘要请求关闭显式 thinking、增量摘要、按需检索、
预算耗尽受控停止、保留最近工具交换、摘要和检索暴露历史顺序、明确覆盖范围、
产品提示词约束压缩后复用已建立的历史证据。原始历史不删除。
这些是有限场景的实测，不是长任务语义永不丢失的证明，也不是原生 Codex 压缩能力的复刻。

## Groq 初始样本

`semantic-live-2026-09-20.json` 是 Groq `openai/gpt-oss-120b` 的实测结果。输入全部为合成历史：早期缓存决策及原因、中间被拒绝的迁移及原因、重复检查噪声、末尾待验证事项。总预算 30000 token，单次输出上限 2048 token，以 5000 字节本地上下文限制强制触发压缩。

从仓库根目录运行（需要 `.env` 或环境变量里的 `GROQ_API_KEY`，会产生实际模型调用）：

```bash
uv run python evals/context_management/run_live.py
```

新结果写入 `semantic-live-latest.json`，不覆盖历史样本。脚本借用测试迁移夹具构造 canonical 历史，然后通过真实 Gateway、Turn、Rollout 和预算管理运行；不发送仓库源码。验收需查看摘要和最终回答，不能只看退出码。

本次两次模型调用成功，输入 3342 token、输出 1277 token（其中 reasoning 554），Rollout 校验通过。摘要保留决策、原因、拒绝方案、待验证事项，并记录没有已实施变更。单一样本不是无损语义保证：原始历史和运行时事实仍必须保留。初轮模型曾把决策误写成已修改，收紧提示后重测通过，因此不能把摘要当成执行证据。

结构与恢复自动测试：

```bash
uv run pytest -q tests/agent/harness/test_semantic_compaction.py tests/agent/harness/test_compaction_consistency.py tests/agent/harness/test_context_recall.py tests/agent/harness/test_parallel_tool_orchestrator.py
```
