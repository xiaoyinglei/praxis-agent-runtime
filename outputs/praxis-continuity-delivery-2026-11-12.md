# Praxis 长任务连续性交付

在 `/Users/leixiaoying/LLM/RAG学习` 直接修改。HEAD 仍为 `313818817bfe1710abd54ce65877d3d1d72eef8b`，保留进入本轮前的未提交工作；没有提交、重置工作树、覆盖旧 outputs 或改写原始 `.praxis` 数据库/WAL。本轮没有付费模型调用，也没有运行全量 Agent/Provider 回归。

## 主方案和实际替换

采用“有界工作上下文 + 完整替换检查点 + 不变原始记录”。CLI 经 `Agent` SDK、`Session`、`services.py` 组装同一个 `RolloutContextManager`、`TurnRunner` 和 SQLite RuntimeStore，没有测试专用 Runtime 或第二套生产压缩管线。

1. **替换摘要输入规划器**：`agent_runtime/harness/context_engine.py::_plan_checkpoint` 不再搜索能装下的最近历史后缀。输入是上一份完整检查点与之后的新工作；超过摘要输入容量时按完整消息/工具交换分块，再按时间顺序合并。每一片都必须参与，不能为装入请求丢掉较早检查点。冻结树、每个节点的输入界限与输出界限后才允许模型 I/O。
2. **独立摘要请求**：默认 `complete-v1` 计划使用无工具 standalone 请求，不继承执行请求的工具 schema。保留旧冻结计划的读取/重放分支，新的默认执行不会再生成 `summary_history_start` 后缀计划。
3. **大工具结果的处理**：摘要输入中的过大工具正文可以变成明确标注的部分预览；保留调用、结果身份和 `read_context(item_id=...)` 引用。原始结果不被改写。若完整交换仍装不下，或不可缩减的非工具输入超限，在派发前拒绝，不截断用户任务或伪造摘要。
4. **摘要和续跑一起准入**：先组装实际续跑底座，计入用户约束、运行时事实、工具 schema 和输出预留；再求摘要的序列化字节容量，测量叶子和合并请求，为所有摘要节点及至少一次续跑预留 Turn token/cost。预算不足可缩小摘要输出容量，但不能省略输入分片。`post_compact_target` 只作调度目标，不再充当续跑硬上限。生成结果仍须满足冻结的字节界限，并通过实际续跑请求测量才能安装。
5. **持久失败与原子安装**：保留并使用 `compaction_jobs.py` 的持久作业、节点结果和预算预留；新的失败身份由完整源与计划修订定义，重开后不会因剩余预算、步骤或时间变化重新付费。响应先入原始记录，再做本地结果校验。若模型违反输出界限，原响应保留、作业失败，下一次相同源请求被拒绝。检查点替换与下一条 prepared 模型操作在同一事务提交。
6. **先定位检查点再读取窗口**：`RuntimeStore.list_model_context_items` 通过 `RolloutStore.latest_context_checkpoint` 先查最新完整替换，读取显式保留的原 Item 和后缀。普通模型请求不再先反序列化整个工具历史。工具发现、项目指令、计划及作业读取改为类型/工具名过滤；附件完整性检查仅查询引用身份。分支截止点与 context reset 在读取正文之前解析。
7. **兼容和取证**：`replacement_history` 仍使用 `semantic-compaction-v7` 存储契约，不要求重写旧数据库。没有完整检查点的旧 Rollout，以及覆盖范围采用旧全档案契约的后续 elision，走旧读取路径。原始工具调用/结果和模型响应留在同一 Rollout；`read_context` 能搜索历史并按原始 Item 读取。模型投影继续校验工具调用与结果配对。

核心改动路径：

- `agent_runtime/harness/context_engine.py`：完整源规划、节点执行、双请求容量/预算准入。
- `agent_runtime/harness/context.py`、`context_ops.py`：检查点投影、原始源和提交验证；新的工具结果裁剪以当前工作窗口为覆盖范围。
- `agent_runtime/harness/runtime_store.py`、`rollout.py`：检查点优先读取、祖先/重置边界、带条件的原始记录读取。
- `agent_runtime/harness/compaction_jobs.py`、`model_retry_ops.py`、`model_runner.py`、`recovery.py`：持久失败、冻结计划、恢复和原子续跑衔接。
- `agent_runtime/harness/request_context.py`、`tool_router.py`、`artifact_ops.py`：正常请求所需状态的定向读取。

这是对已有未提交修复的继续替换，上述文件的整个 `git diff HEAD` 包含进入本轮前的工作，不能全部归为本轮新增。

## 源码调用链与选取依据

Codex 固定提交 `a4c9194247c6a1e79aa3e72cc6cd77238f2ed354`，本地切片位于 `artifacts/agent-reliability-fix-20261020/codex-reference`。缺少的两个文件通过同提交 GitHub 文件接口读取：

- [session/mod.rs](https://github.com/openai/codex/blob/a4c9194247c6a1e79aa3e72cc6cd77238f2ed354/codex-rs/core/src/session/mod.rs)：`record_conversation_items` → `record_prepared_conversation_items`（约 3636–3752 行），模型历史与原始 Rollout 写入分开；`replace_compacted_history`（约 4242–4418 行）处理历史代换、代际检查、保留尾部及恢复元数据。
- [session/rollout_reconstruction.rs](https://github.com/openai/codex/blob/a4c9194247c6a1e79aa3e72cc6cd77238f2ed354/codex-rs/core/src/session/rollout_reconstruction.rs)：重建时选择最新可用检查点及之后的记录；兼容旧记录，并恢复相应运行时元数据。
- `context_manager/history.rs`：`record_annotated_items`/`record_item_with_metadata` 约 507–582 行处理模型侧工具输出大小；`for_prompt` 做调用/结果归一化。解决原始证据与有限请求窗口的职责冲突。
- `session/turn.rs`：`run_pre_sampling_compact` 约 1379 行检查容量，采样后约 610–707 行根据 usage、是否继续及阈值进入 `run_auto_compact`。解决“有新工具结果之后、下一次请求之前”的触发时机。
- `compact.rs` 约 260–450 行：独立 Prompt、容量失败时缩减输入、有限瞬态重试、不完整响应处理、构造替换历史并提交、重新计算用量。Praxis 采用独立摘要与完整代换；没有采用删除最早输入的容量重试，因为本任务必须保留上一检查点中的约束与未完成工作，改用完整源分块归并。

Claude 使用本地历史镜像 `/Users/leixiaoying/PycharmProjects/brath-claude-code/src`，本轮核实提交 `276b8e6`。该镜像来源未独立核验，不能称为当前官方 Claude Code 源码。

- `query.ts` 约 360–395 行：从边界后的消息出发，`applyToolResultBudget` 后经过 microcompact 和自动压缩路径，再进入模型请求；约 510–542 行用 `buildPostCompactMessages` 的结果替换后续请求历史，并携带失败计数。
- `utils/toolResultStorage.ts`：`persistToolResult`、`enforceToolResultBudget`（约 744 行）、`applyToolResultBudget`（约 923 行）、`reconstructContentReplacementState`（约 959 行）。原文落盘、模型侧预览按工具身份稳定替换、重开重建替换状态，避免重开后重新膨胀或失去证据。Praxis 使用自身原始 Item 存储和持久投影，未引入另一套文件数据库。
- `services/compact/autoCompact.ts` 约 22–100 行：上下文容量扣除摘要输出预留、触发缓冲和三次失败熔断；解决调度与重复付费。Praxis 的相同源失败阻断还跨进程持久化，不仅靠循环内计数。
- `services/compact/compact.ts`：`compactConversation`（约 387 行）→ `streamCompactSummary`（约 1136 行），优先缓存父请求 fork、再走 streaming fallback（约 1290 行），有上限的 prompt-too-long 裁头重试、验证摘要、组装边界/摘要/附件/hooks/已发现工具。`buildPostCompactMessages`（约 330 行）建立后续模型历史。Praxis 默认不采用继承工具的 fork，也不采用裁头丢信息；保留自身工具发现和运行事实，摘要只承担语义交接。
- `QueryEngine.ts` 约 681–735 行：持久化消息，边界前先刷保留尾部，用户和边界写入等待完成，assistant 异步写入保持顺序。`utils/sessionStorage.ts::loadTranscriptFile`（约 3472 行起）定位边界和尾部、恢复边界前元数据、链接保留片段后构造会话链；`sessionRestore.ts` 将恢复结果接入 CLI continue/resume。解决运行历史与磁盘恢复历史不一致。Praxis 使用 SQLite 原子检查点/续跑操作提交解决同一问题。

镜像中 `snipCompact`、`reactiveCompact`、`contextCollapse` 的相关实际模块缺失，只能看到开关/引用；没有把这些路径当成已经追通的实现，更没有机械搬入所谓五层。

## 本地验证结果

最终定向集 **292 passed in 53.59s**。覆盖本次改动涉及的容量、摘要作业、预算、旧语义压缩、配对、前缀更新、会话、结果裁剪、恢复、原始证据、附件完整性、工具路由及项目指令。另新增的检查点分支截止/重置回归 **1 passed**；总计 293 个不同用例通过。Ruff 与 `git diff --check` 通过。

测试使用项目 `.venv/bin/python -m pytest`。`uv run` 在启动前遇到 uv 缓存权限错误，因此直接使用同一现有虚拟环境；不是测试失败。最终主集日志在 `/private/tmp/praxis-continuity-final.7bMBpk/pytest.log`。

公开 CLI 验收在 `tests/agent/harness/test_long_task_cli.py`：子进程进入真实 `cli.agent_app()`，只在模型工厂注入确定性本地供应商；默认 Runtime、工具、数据库与恢复链保持真实。进程退出钩子放在已经提交的检查点/失败作业之后。

| 证据 | 实际结果 |
| --- | --- |
| 工作区与数据库 | `/private/tmp/praxis-continuity-final.7bMBpk/cases/test_cli_multiple_semantic_che0/`，独立 workspace 和 `rollout.sqlite` |
| 同一 Turn | `turn_c8975269072449beb7735cbbe771b7f6`，最终 completed |
| 实际工作顺序 | apply_patch Item sequence 9 → 语义边界 51 → apply_patch 60 → 语义边界 96 |
| 退出时状态 | `config.py` 的 STAGE 为 2；第二边界及下一 prepared 模型操作均已提交；进程退出码 73 |
| 重开 | CLI `recover <同一 Turn> --action continue`；先 read_context 搜索，再按原始 Item ID 读取（sequence 105、111） |
| 完成 | apply_patch 117 把 STAGE 改为 3，read_file 123 读取验证；API_NAME 始终为 stable；原工具结果未变；`store.verify().valid` |
| 摘要连续性 | 两次摘要均不携带工具 schema；第二次包含第一次检查点；重开请求包含原始用户约束、旧决策及待完成工作 |
| 相同失败不重派 | `test_cli_does_not_dispatch_fai0/failed.sqlite`，Turn `turn_d7008bd3dbf1487a8a2341a0e8f727b3`；供应商返回超界摘要后保存 completed 模型操作与 failed 作业，退出 74；重开没有新增模型派发/用量，报告 context_summary_previously_failed，零安装 |

额外回归 `tests/agent/harness/test_checkpoint_continuity.py` 验证完整旧检查点参与第二次摘要、大工具交换可有界准入且原文不变、检查点后的裁剪可重放、正常准备模型请求不加载全档案，以及分支未来记录/context reset 不泄漏。已有语义压缩测试从“只保留最新后缀”改为“所有叶子按序覆盖全部输入”；没有降低配对、预算或完整性要求。

最短复跑（无付费）：

```sh
cd '/Users/leixiaoying/LLM/RAG学习'
.venv/bin/python -m pytest -q tests/agent/harness/test_checkpoint_continuity.py tests/agent/harness/test_long_task_cli.py
```

这些结果证明确定性模型下的执行、容量、提交与恢复契约，不等于云模型长期语义质量保证。模型若不遵守摘要大小或遗漏事实，仍会失败或需要原始证据回查；运行时不把格式有效当作语义正确。正常模型投影已按检查点读取，语义压缩源构造和显式档案查询仍可读取旧记录，也没有声称所有预算/操作元数据查询均为常数空间。

## 用户自行执行的付费验收

以下命令尚未执行。使用现有 `deepseek-flash` 凭据；复制配置到临时目录，只降低触发阈值以便观察多次压缩，阈值本身不是验收标准。数据库和模型会话均放在独立临时目录。

```sh
export PRAXIS_ACCEPTANCE_ROOT="$(mktemp -d /private/tmp/praxis-paid-continuity.XXXXXX)"
export PRAXIS_MODEL_REGISTRY_PATH="$PRAXIS_ACCEPTANCE_ROOT/models.yaml"
'/Users/leixiaoying/LLM/RAG学习/.venv/bin/python' - <<'PY'
import os
from pathlib import Path
import yaml
root = Path(os.environ['PRAXIS_ACCEPTANCE_ROOT'])
workspace = root / 'workspace'
workspace.mkdir()
config = yaml.safe_load(Path('/Users/leixiaoying/LLM/RAG学习/configs/models.yaml').read_text())
config['auto_compact_token_limit'] = 24000
(root / 'models.yaml').write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False))
for i in range(32):
    notes = ''.join(f'# Inspection {j}: module {i} keeps its public ABI; evidence details must remain readable.\n' for j in range(160))
    (workspace / f'module_{i:02}.py').write_text(f'API_NAME = "stable"\nVALUE = {i}\n' + notes)
(workspace / 'evidence.txt').write_text('FINAL_INCREMENT=7\nKeep API_NAME stable and preserve inspection comments.\n')
(workspace / 'verify.py').write_text('from pathlib import Path\nfor i in range(32):\n    p = Path(f"module_{i:02}.py")\n    d = {}\n    exec(p.read_text(), d)\n    assert d["API_NAME"] == "stable" and d["VALUE"] == i + 7\n    assert p.read_text().count("# Inspection ") == 160\nprint("32 modules verified")\n')
(root / 'task.txt').write_text('完成这一个任务：先读取 evidence.txt，然后按序逐个用 read_file 读取 module_00.py 到 module_31.py 的完整内容，并用 apply_patch 把每个 VALUE 增加 FINAL_INCREMENT。保留 API_NAME 与所有注释，不修改 evidence.txt 或 verify.py。每完成一个文件就读取核对，持续更新规划。不要批量用一个命令改完。最后一个文件修改前，用 read_context 搜索最初 evidence.txt 工具结果并按原始 Item ID 回查增量；最后真实执行 python3 verify.py，成功才结束。压缩后按已有修改继续，勿重复已完成工作。')
print(root)
PY
cd "$PRAXIS_ACCEPTANCE_ROOT/workspace"
'/Users/leixiaoying/LLM/RAG学习/.venv/bin/agent' run "$(cat "$PRAXIS_ACCEPTANCE_ROOT/task.txt")" \
  --model deepseek-flash --non-interactive --disable-workspace-mcp \
  --allow-write-tools --allow-execute-tools --require-workspace-change \
  --acceptance-command 'python3 verify.py' --max-steps 180 --max-tokens-total 500000 \
  --checkpoint-db "$PRAXIS_ACCEPTANCE_ROOT/rollout.sqlite" \
  --model-session-path "$PRAXIS_ACCEPTANCE_ROOT/model-session.json"
```

从另一个终端只读查看边界（替换目录为上面打印的实际值）：

```sh
sqlite3 -readonly /private/tmp/praxis-paid-continuity.XXXXXX/rollout.sqlite \
  "SELECT turn_id,sequence,json_extract(payload_json,'$.algorithm_revision') FROM items WHERE kind='context_compaction' ORDER BY sequence;"
```

观察到至少两次 `semantic-compaction-v7`，且中间有实际文件修改、仍有待完成模块时，用 Ctrl-C 退出原 CLI。在同一 workspace、同一临时配置环境中，恢复**同一 Turn**：

```sh
'/Users/leixiaoying/LLM/RAG学习/.venv/bin/agent' recover --last \
  --checkpoint-db "$PRAXIS_ACCEPTANCE_ROOT/rollout.sqlite"
# 上一步显示可用动作包含 continue 时：
'/Users/leixiaoying/LLM/RAG学习/.venv/bin/agent' recover --last --action continue \
  --checkpoint-db "$PRAXIS_ACCEPTANCE_ROOT/rollout.sqlite"
python3 verify.py
```

如果退出正好发生在供应商请求已发出但响应未提交的窗口，先按 `recover` 显示的 unknown 状态核对，不把再次调用当成已知无费用的 continue。不要用 `run --previous-turn-id` 或交互 `resume` 代替本次同 Turn 恢复。付费验收应核对边界、实际补丁、回查及最终测试记录；不能只凭最终文字或压缩次数判定通过。
