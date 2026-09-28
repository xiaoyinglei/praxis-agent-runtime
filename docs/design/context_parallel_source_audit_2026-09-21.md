# 上下文与工具并行：参考源码审查

日期：2026-09-21。状态：参考对照和反例审查已完成；实施设计及新验收夹具尚需补齐，现有实验实现未交付。

工作目录 `/Users/leixiaoying/LLM/RAG学习`，分支 `compact`，基线
`13867eac7da84345e73485a2b12e2c099c911d01`。本轮审查没有清理、回滚、提交、合并现有改动。
下述文件行号指本轮审查时的工作树；后续修改后应以符号名和 Git diff 定位。

## 1. 参考来源与可确认范围

**Claude 本地源码 C**：`/Users/leixiaoying/PycharmProjects/brath-claude-code`。
提交 `276b8e6a0160939c74f04074a5e10ec1ba76982b`，提交日期 2026-04-03，
说明为“feat：最新源码，59MB反编译后的”。remote origin 是
`https://gitee.com/Guoqing-Li/brath-claude-code.git`，github remote 是
`https://github.com/xiaoyinglei/claude_code-src.git`。
README 自称源于 2026-03-31 的 source-map 泄露；这是仓库自己的来源声明，未独立认证。
没有根 package.json，未找到可靠的发行版本证明，不能将提交日期或 README 日期当作产品版本。
本轮核对其跟踪文件无 diff；只有 .DS_Store 未跟踪文件。

另有相同提交的副本：
`/Users/leixiaoying/Documents/Codex/2026-09-05/https-github-com-xiaoyinglei-claude-code/work/reference-src/claude_code-src`。
它和用户仓库 remote 名称相符；具体哪次用户消息首次提供它尚未追溯，不能伪称已核实首次来源。

**Codex 本地源码 X**：
`/Users/leixiaoying/Documents/Codex/2026-09-05/https-github-com-xiaoyinglei-claude-code/work/reference-src/codex`。
remote `https://github.com/openai/codex.git`，提交
`ddf04ad26789d040f9ef6a96736f76602e35a6cc`，2026-09-05，跟踪工作树干净。
以下结论针对这份固定快照，不声称是今天最新 Codex 或桌面实际部署版本。

C 中缺少 feature 分支引用的 `cachedMicrocompact.ts`、`reactiveCompact.ts`、
`snipCompact.ts`、`contextCollapse` 实现。可以确认入口、注释、门控，不能确认缺失分支内部算法。
因此不能把“五层”当作已验证、默认串联启用的统一架构。

## 2. 源码 → 机制与限制 → Praxis 差异 → 处置

X 下路径以 `codex-rs/core/src/` 为根；C 下以 `src/` 为根。

| 范围 | 参考位置和实际机制 | Praxis 当前差异 | 处置 |
|---|---|---|---|
| 触发 | C `services/compact/autoCompact.ts:32–93,160–237`：窗口减摘要输出预留（模型输出上限与 20000 的较小者），再减 13000；支持覆盖、递归保护、实验门控。X `session/context_window.rs:54–124`：配置 scope、模型可用窗口、token-budget buffer；`session/turn.rs:1082,1248` 负责调用和 provider 分流 | `turn.py:883–896` 在 adapter 有效输入预算上再乘 0.85；没有说明与已有 safety margin 的关系 | 修改。使用同一份有效请求预算，明确输出预留、协议开销与安全余量；不可复制参考常数或新造百分比 |
| 预算 | C `utils/tokens.ts:45` 将输入、cache、输出 usage 合计并结合估算；不等于精确重新分词。X 区分活跃窗口与压缩 scope | `modeling/gateway.py:220` 已计算 `min(stage_input, model_window-output-safety)`；adapter 对实际序列化 JSON 计数。序列化一致不等于服务端分词精确；Turn 累计花费与窗口容量是不同限制 | 保留单一序列化及 wire hash 校验。修改文档“exact token”表述，保留 usage 结算和 overflow 兜底，避免重复余量 |
| 工具结果清理 | C `services/compact/microCompact.ts:253–291,447–491`：时间触发或受限 cache edit；时间路径仅替换旧 `tool_result.content`，保留消息、调用 ID 和最近结果。不支持的路径可能直接不处理 | `context.py:311–380` 将全部覆盖消息转成 JSON 字符串摘要，清理也改变消息角色；最近结果只剩字符串。1000/50000 字符和最后 3 条是本地策略 | 修改为结果内容替换投影，独立持久化替换内容与原 item ID；不把无模型清理冒充语义摘要。清理触发与保留策略应由统一请求预算和明确的本地策略决定，不宣称来自参考实现 |
| 摘要输入 | C `compact.ts:1289–1321` 取最近 compact boundary 后的消息，移除图像/重复附件，禁用 thinking；不同内部路径有区别。X `compact.rs:250–335` 使用当前 history，超窗口按历史 item 裁旧，失败有界 | `context.py:260–300` 用已提交投影加新消息，方向正确；`turn.py:689–700` 对 JSON 字符串任意二分，会切断事实和工具组 | 保留增量来源和 hash；撤销字符二分。以完整消息/工具交换为输入单元；过大工具结果先归档，不能靠文本切片声称“完整语义” |
| 近期保留 | C session-memory 路径 `sessionMemoryCompact.ts:55–61,317,568` 按 token/text 消息下限扩展并保护工具对；默认 full compact 并不等同该策略。X 本地 `compact.rs` 保留受限用户消息加摘要；mid-turn 注入位置有模型训练约束 | `semantic_retained_tail()` 仅在最后一条是 tool 时保留最后一组；失败四轮后放弃整组。公开候选还有 12/8/4/2/0 消息阶梯 | 修改为明确保护当前请求及最近完整工作单元，按实际预算选边界。不能宣称“一组”或消息数阶梯来自参考 |
| 原文检索 | C `compact/prompt.ts:340–350` 提供完整 transcript 路径。不是向摘要递归塞回完整原文。X 可确认 rollout 持久化，不等于存在同名 read_context | `context_recall.py:40–110` 做可见 item 分页和单 item 文本搜索，history 目录只有元数据；不知道 item ID 时必须逐项探查 | 保留权限边界、分页、完整原文；修改为可见历史范围内的有界内容搜索，返回 item ID、历史顺序、匹配片段、覆盖范围。这是 Praxis 自主 ACI 设计 |
| 连续压缩 | C 当前边界后历史、session memory 游标；后者为实验门控，不能混为一谈。X 将 replacement history 作为后续实际输入 | 当前增量来源已避免重读所有归档，但 `context.py:194–205,489–560` 累积完整旧用户/context 文本、所有读写来源；固定事实包能单独超窗 | 保留增量摘要；修改事实分层：当前运行硬状态保留，旧事实以可检索证据引用承载。预先测量不可压缩底座，超限必须零摘要调用受控失败 |
| 失败与恢复 | C `autoCompact.ts:260–356` 连续失败 3 次熔断；full compact 有 prompt-too-long/stream retry。X `compact.rs:300–348` 区分中断、预算、窗口、重连；`session/mod.rs:3761` 保存实际 replacement history，`rollout_reconstruction.rs:347` 恢复 | 现有 candidate CAS、摘要独立 operation、prepared/committed 恢复、overflow 一次重试可保留；摘要 4 个词数档×有无尾部×递归最多 64 调用，局部上限不代表 Turn 总预算可靠收敛 | 保留事务与恢复；撤销词数阶梯，摘要尝试预算在尝试前冻结并落盘。不能在明显放不下时先花完预算，再 fallback |
| 工具并行 | C `services/tools/toolOrchestration.ts:20–115` 按原顺序形成连续安全段，unsafe 独占；StreamingToolExecutor 也有独占门。X `tools/parallel.rs:116–160` 由 router 声明决定读/写锁准入 | Harness 批量接入、预检、claim、取消 drain 可保留；`tool_orchestrator.py:359` 和 Executor `_can_run_in_parallel` 是整批判定，一个冲突使整批串行 | 保留可靠性边界；修改为连续、保序分段调度，不跨独占工具重排。资源判定仍由现有 Executor 负责，不能新增第二个工具安全系统 |

## 3. 已复现的问题

使用项目现有 Python 环境与临时 SQLite，不调用模型，不修改项目代码。
命令入口：`UV_CACHE_DIR=/private/tmp/praxis-uv-cache uv run --no-sync python`。
初次默认 uv cache 访问被沙箱拒绝；改用已有临时 cache，复现正常完成。

### 清理后工具结构消失

沿用 `test_semantic_compaction.py::test_old_tool_outputs_are_archived_before_spending_a_summary_call`
的五组读取历史，两个旧长结果、三个近期短结果：

```text
before_roles = [user, assistant, tool, assistant, tool, assistant, tool, assistant, tool, assistant, tool]
after_roles = [context, user]
recent_results_structured = 0
candidate_equals_rebuild = True
next_cheap_candidates = 0
```

说明试算/落盘一致性成立仍然不足：一致地执行了错误的投影策略。
现有测试验证最近内容出现在 summary 字符串，未验证其结构仍可供模型使用。

### 固定事实包超限

历史加入一条 `context_message`（`old context ` + 6000 个 X），再加可压缩的长 assistant 历史。
本地总预算 5000 字节，候选摘要仅 `Done.`：

```text
ContextBudgetExceededError: Model context exceeds the configured total byte limit (7181 > 5000).
```

旧 context 全文被复制进 protected facts；调 500/250/125/62 词不能解决这个下界。

## 4. 完整调用链审查

1. **Session** 冻结 binding，创建/恢复 Turn；`session.py:540–633` 新增摘要 prepared/completed 分支。
   保留在既有恢复协议内，需验证重启后继续同一摘要 operation，不重复计费或发布最终回答。
2. **Turn** `capture_step_context` 从 ContextManager 取得消息，`run_turn` prepare；超限进入
   `_prepare_compacted_step`，摘要也走持久化模型 operation。串行模型调用期间的来源 hash 与 commit CAS 要同时保留。
3. **ContextManager** 从 Rollout 投影；候选包含冻结 payload/messages，提交前检查 source revision。
   原始 item 保留。语义替换和工具结果替换目前混用，是首要修改点。
4. **ModelAdapter** `prepare:86` 构造 StableModelContext，摘要 purpose 不暴露工具，
   `LLM_SUMMARIZE` 有独立 stage；`_budgeted_request:592` 只检测，不暗中改写 transcript。
   删除 core/model_request 中的另一套压缩逻辑应保留，不能重新制造双权威。
5. **Provider/Gateway** 使用同一 OpenAI/local wire 序列化；adapter `dispatch:276` 校验返回 wire hash。
   Gateway `effective_stage_budget:220` 已扣输出和安全余量。实测需记录实际输入、输出 usage 和 rejection，不能只比较候选字符数。
6. **落盘/恢复** Rollout 事务记录 compaction、模型 operation 和预算；内部 summary 不发布用户输出。
   测试存在 prepared、completed、commit 后恢复、fork 和 stale source 场景；它们不能证明摘要质量。
7. **工具调用** Turn `execute_batch:1307` → Harness preflight/context/claim → ToolExecutor。
   prepared、ready、running、unknown 是恢复事实；断电后的副作用不能靠重跑或猜测变成成功。

## 5. 建议实施顺序及选择理由

推荐在现有内核上改投影和调度，保留存储/预算/恢复权威。相比继续调提示词，它修复明确反例；
相比复制 Claude/Codex 全栈，它不依赖缺失内部模块或 DeepSeek 不具备的远程 compact 协议。

1. 先把无模型工具内容清理改为结构保持的、可恢复的内容替换；最近完整工具组可用。
2. 将摘要规划改为预算驱动：测量固定上下文、当前请求、保留尾部、工具 schema 与摘要壳，
   摘要输出额度同时受主请求剩余容量、LLM_SUMMARIZE 阶段输入/输出上限、Turn 剩余预算约束；还需为摘要后至少一次主请求预留额度。adapter 现有独立 2048 输出上限也应纳入统一计算。没有空间直接报告不可满足，不进行注定无效的模型调用。
   正常窗口使用当前活跃历史一次摘要；历史超过摘要窗口时按完整工作单元分段，明确记录覆盖范围。
   摘要输出超限/不完整不能提交；重试数量与 token 上限预先冻结并绑定持久化 plan ID、source hash；恢复 prepared、completed-before-commit 或 source-changed 分支时不能重置计数，已完成摘要不重复结算。不能看答案后调门槛。
   单个工作单元仍超窗时，工具结果可改为归档引用，并把覆盖标为“原文未进入摘要”；超长用户要求或硬约束无法完整容纳时受控失败，不把未读归档计作已摘要。
3. 当前硬约束/计划/不确定操作与旧可检索证据分开，去掉旧事实包无界增长；保留 chronology 与来源。
4. 增加有界跨 item 搜索，减少“目录 → 每条 item”的无信息回读。
5. 工具连续安全段并行、冲突和独占调用构成屏障；预检和取消保持现有持久化协议。
6. 产品提示词中“一概不要重读已建立事实”的样例补丁暂不作为验收前提。先修复表示和检索，
   对照原产品提示词与当前提示词；不可通过删掉失败对照或加强禁止回读来掩盖机制问题。

这些是基于参考边界和项目协议作出的自主选择，不是 Claude/Codex 算法的逐行复刻。

## 6. 修改前冻结的验收范围

旧报告和失败样本全部保留；旧的 ≤8 次摘要、≤4 次 agent 调用是后来添加的门槛，不能追认为先验设计依据。
实现前另外固定输入、初始工作树、模型 `deepseek-flash`、thinking 设置、产品提示词版本、预算和 judge，
以相同配置做前后对照，报告每次运行而非只挑成功。

| 固定任务 | 独立判据 |
|---|---|
| 老结果清理后继续编辑 | 原工具 ID/配对/近期结果保留；真实文件修改与验证通过；原 item 可取回 |
| 多轮决策变更＋拒绝方案＋重启 | 必须实际经历至少两次连续压缩；按历史顺序回答当前决策、原因、早期否定证据、未实施状态；关闭重开 SQLite 后一致；检索命中有来源 |
| 长代码任务跨压缩继续 | 临时小仓库内完成用户要求的改动；隐藏功能断言、实际 diff 和回归命令成立；禁止仅判断 completed |
| 固定底座本身超限 | 原历史无损，受控失败，摘要模型调用为零 |
| 混合读取/写入/读取 | 固定序列“读、读、写、读、读”；屏障前后各自两次读取均重叠；写入前读取已结束；写后读取看到结果；副作用次数准确 |
| 审批、取消、故障恢复 | 审批边界成立；取消后没有悬空运行任务；unknown 不自动重复副作用；重启后调用结果不缺失不重复 |

报告必须含成功/失败、错误完成、每种压缩次数、摘要/主模型调用次数、token、耗时、检索次数、
模型 request 与工具 operation ID、前后 diff。并发用区间证据，语义用独立任务判据。
具体新代码任务输入与资源上限仍需在实现前写入可执行验收夹具；此文不是已完成真实验收的声明。

## 7. 本轮结论

保留候选一致性、原文持久化、增量来源、摘要独立 operation、预算结算及恢复基础。
修改工具结果投影、固定事实包、摘要分段/预算和混合批次调度。
撤销未经推导的 85% 与词数阶梯，不把样例提示词或测试数量当交付依据。
缺失的 Claude 内部分支、发行版本、远程 Codex 服务端 compact 细节均标记未知。

## 8. 本轮基线与独立复核

- 定向测试：51 passed，覆盖四个现有 context/parallel 测试文件；两个反例并未被这些测试排除。
- 沙箱联网失败样本：`evals/context_management/deepseek-functional-20260921-201200-disabled/report.json`。三项首次请求均为 Connection error，operation unknown，无可用模型结果；不算语义验收失败的证据，也不删除。
- 获准联网后原实现基线：`evals/context_management/deepseek-functional-20260921-201313-disabled/report.json`。既有三项检查均通过；三轮历史摘要调用分别 1、2、2，主模型各 1；回查 2 次主模型，读改验 4 次。模型为 deepseek-flash，thinking disabled，当前 production 提示词。
- 本次未改 runtime、产品提示词、既有验收输入或门槛；通过仍是旧合成场景基线，不覆盖上面发现的结构与容量缺陷，也不等于完整产品 Session 验收。
- 报告 usage 中 model_ms/tool_ms/wall_ms 为 0，不能用于耗时结论；并行只能引用真实 read_intervals。
- 独立只读 reviewer 认可参考机制区分，指出预算交集、单个超大工作单元、恢复计数与具体验收冻结仍需明确。本修订已补前述设计边界；可执行的新夹具和精确资源上限仍未冻结，因此不应直接开始摘要策略实现。


## 2026-09-28 后续实施更正

第 4 节所述 LLM_SUMMARIZE 是审查时状态。现在上下文压缩用独立 CONTEXT_COMPACTION，
缺省继承 agent_step 输入容量与 llm_summarize 输出额度，而非沿用普通摘要的较小输入限制。
依据、原现场预算复算、真实成功/失败边界见 `context_root_findings_2026-09-23.md` 第 5 节。
这些继承规则属于本项目设计；参考代码提供同一模型历史压缩边界，不提供本项目的具体数值。
