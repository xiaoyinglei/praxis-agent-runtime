# 上下文投影和并行实施计划

依据：`context_parallel_source_audit_2026-09-21.md`。在当前 dirty 工作树原地修改，不提交。
先执行可由源码与反例确定的两项；摘要预算重构必须先完成独立的预算与恢复夹具。

## A. 工具结果内容替换

- [x] 先增加失败测试：候选/提交/重启后三者消息角色、调用 ID 与最近完整交换一致；再次清理仍可处理新增结果；后续语义压缩不会使旧替换结果复活。
- [x] `context.py`：压力下按旧到新逐步替换已完成且非错误的工具结果正文；保护最近完整交换。只尝试实际减少字节的替换，取消 1000/50000 字符阶梯。原文仍在 canonical item。
- [x] 在既有 compaction payload 增加可选 `tool_result_overrides: {item_id: content}`；`rollout.py` 在现有事务中校验目标属于所冻结前缀且为工具结果，再落盘。无新数据库表或第二条消息权威。
- [x] 投影见 overrides 时保持原消息角色，只替换指定正文，不插入 summary；语义摘要覆盖旧 prefix 时清除被覆盖的旧替换。旧 v3 历史仍按原规则恢复。
- [ ] 真实模型沿用已冻结 functional_live 场景、当前配置与门槛再跑，与 `201313-disabled` 基线比较。

## B. 保序分段调度

- [x] 失败测试固定“读、读、写、读、读”，使用屏障证明两段分别并行、写前已 drain、后段看见新内容。
- [x] Harness 保留整批 call 登记与特殊远程/审批恢复约束，分连续可并行本地段；不跨独占调用重排。
- [x] Executor 仍负责动态资源冲突和全段预检；把存在资源冲突的已预检调用分成保序无冲突段；每段不超过原 max_parallel_calls。
- [x] 取消必须 drain 当前已启动段；后段不得开始；重新运行审批/取消/unknown 恢复测试。

## 后续：摘要预算与完整任务验收

摘要内容、触发与产品提示词在 A/B 期间不调整，以隔离机制变化。
接着冻结：固定底座不可容纳时零摘要调用；摘要 stage/input/output/Turn budget 交集；恢复计数不重置；
完整消息边界分段；具体代码任务跨两次压缩、SQLite 重开后完成真实 diff 与外部功能验证。
完成这些前不得宣称整个上下文任务交付。

## 2026-09-23 冻结 Session 真实代码验收

脚本 `evals/context_management/session_code_live.py` 在首次模型请求前写 frozen.json 和脚本 SHA256。
只用 DeepSeek Flash 非思考模式；每 Turn 16 步、100000 token，模型输出 4096，已有 summary stage 输出额度。
本地上下文 14000 字节，六份实际规格/样例文件分两阶段读取；不注入伪造历史，不改产品提示词。
第一阶段实现租户事件 reconcile；关闭 Session/SQLite 后同一 thread 第二阶段实现 totals，保留旧 API。
每阶段要求三个规格读取真实重叠、实际语义压缩、模型执行自己的验证命令；外部隐藏断言检查跨租户、
同序号后输入胜出、删除与恢复、排序、输入不可变、大整数、空集合及零值租户。规格文件哈希必须不变。
验收脚本或 oracle 出错须单独修正并保留原记录，不能调整这些通过条件来追认结果。

2026-09-23 追加证据：Session 首次实测 092430 连续六次重复读规格、五次摘要后预算失败。
消息链路发现语义摘要总在 covered[0]：当前轮执行结果先于触发它的 user，且被 OpenAI serializer 并入 system。
固定参考 Codex compact.rs:683–758 明确保留用户消息后再放摘要。新 semantic-compaction-v4 将摘要放到最后被覆盖的
非保留消息位置；历史摘要留在后续新请求之前，当前轮执行摘要在当前请求之后。旧 v3 记录保持旧投影，重写时升级。
摘要输入增加原本被保留/排除的 current_task，避免摘要在缺少目标时猜下一步。
新增测试检查 candidate、重开 SQLite 投影以及实际 Provider wire 顺序。没有加入禁止重读提示。
Session 验收识别算法版本从仅 v3 扩为 {v3,v4}；任务、字节/模型预算、功能断言、并行和重启条件完全未变。

### DeepSeek 消息角色适配的依据（2026-09-23）

v4 时序修复后的 154712 任务仍重复读取。六次 wire hash 各不相同，snapshot 有摘要，Gateway/SDK 路径未发现漏传。
随后以同一历史做三次单请求对照（不执行返回工具）：`deepseek-shape-probe-20260923-155412/report.json`。
- 原 user-role 摘要：继续读 identity/deletion/ordering 三规格。
- 完全相同摘要改 assistant-role：开始读 ledger.py，并请求 ls。
- 原始 assistant/tool 交换：开始读 ledger.py。

据此仅为 deepseek provider 适配位于 user 后的 continuation compaction memory 为 assistant；前缀和普通 context 不变。
这不是从 Codex 直接搬来的机制，而是本项目的 provider 适配，仍须完整任务验证，三次对照不能证明普适正确。
`continuation_summary_role` 随 Step 快照冻结；旧快照恢复默认 context，避免旧 prepared/unknown 请求变成新的 wire。
没有调整当前任务、产品提示词、模型调用/字节预算、外部功能断言。完整任务再次运行前加 SDK 参数白名单记录，
仅保存模型名、messages/tools 及推理参数，不保存任何凭证或请求头。

### 近期原文和摘要的预算关系

232318 实测第一阶段已通过全部外部断言；重开 SQLite 后第二阶段有进展但摘要+最近两份代码原文
达到 15071/14000，失败记录保留。离线复现证明空摘要底座能容纳，并非固定事实不可压缩。
之前 summary_source 排除保留 tail，所以生成后不可安全去掉超限的 tail；它会丢掉从未摘要的证据。
修改为完整当前可压缩 history 作为摘要输入（含待重放的完整最新工具交换），与参考 full-history summary 对齐。
候选仍优先保留近期原文；实际摘要测量失败时只尝试同一摘要去掉冗余 tail，不重新生成，不改变预算或任务。
若同一摘要仍无法容纳则受控失败。测试证明尾部进入摘要请求，并且只派发一次摘要。


### 2026-09-28 状态补充

A 的投影、原文归档与候选一致性、B 的保序并行与恢复已经实现并验证；
具体真实任务结果与尚未满足的原门槛见 `context_root_findings_2026-09-23.md` 第 5 节。
新增 context_compaction 阶段从 Agent 输入容量继承，避免通用 16K 摘要阶段制造无谓分块。
此处旧日期的角色、尾部保留改动仅是当时步骤，不能单独作为机制成熟的证据。
