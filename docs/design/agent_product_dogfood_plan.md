# Praxis Agent 历史 Dogfood 记录

Status: historical evidence snapshot

> 本文保留当时的模型、配置和运行结果，不代表当前默认配置或发布门禁。

> **执行方式：** 当前任务内联执行。真实模型和公开 CLI 是主验收；历史提交中的测试只在 Agent 完成修改后注入，作为隐藏回归门槛。

**目标：** 用真实发生过的仓库任务测出 Agent 的端到端完成率、人工介入和失败原因，先完成 5 例试跑，再决定是否扩展到 20～30 例。

**边界：** 不新增评测 Runtime，不修改历史任务答案，不让 Agent 看到目标提交 diff；每例都在 `/tmp` 的隔离快照中执行，不修改当前工作树。

## 统一执行协议

1. 用 `git archive <parent>` 导出目标提交的父版本。
2. 从快照目录运行当前分支公开命令 `agent run`。
3. 优先使用真实模型 `deepseek_chat`，允许 workspace 写入和受限命令执行。Groq 免费额度只作可用性对照，不作为主验收模型。
4. Agent 结束后，注入目标提交版本的验收测试。
5. 运行指定测试，并记录：退出状态、Turn 状态、工具调用、是否修改生产代码、是否通过隐藏验收、失败分类。
6. 快照在运行前执行 `uv sync --frozen`，确保 `run_command` 验证的是任务自己的真实环境。

## 试跑任务

### 1. 优先使用 wheel 内置模型目录

- 父版本：`4cc51a397f9955cb2af586ecb2441bc85c5725a0`
- 目标提交：`00466a079f385a81193cbac9c1d6782070626aa4`
- 任务：显式环境覆盖优先；wheel 内置资源其次；源码 checkout 配置只能兜底，不能覆盖已安装 wheel 的模型目录。
- 隐藏验收：`tests/agent/test_package_distribution.py::test_built_wheel_loads_bundled_qwen35_model_outside_repo`

### 2. OpenAI wire 只允许一个前置 system message

- 父版本：`cda745f3dec8fa366535840ab986e21fb3e9f9a1`
- 目标提交：`4cc51a397f9955cb2af586ecb2441bc85c5725a0`
- 任务：合并前置 system/context；对话开始后的 context 转成 user event；拒绝后置 system；更新序列化 revision。
- 隐藏验收：`tests/provider/test_openai_wire.py` 和 `tests/provider/test_llm_gateway.py::test_gateway_adapts_one_canonical_request_to_openai_wire`

### 3. 持久化并公开 update_plan

- 父版本：`c64e86d6e22c48ecbf550a399877ccc8c7980c5c`
- 目标提交：`e0faaa135bcea20a7d73ba0ee411f0d00014fede`
- 任务：`update_plan` 必须写入 canonical PlanState、checkpoint、结果和完整流式事件；CLI 展示模型提交的计划，不能只返回临时计数器。
- 隐藏验收：`tests/agent/test_update_plan_surfaces.py`

### 4. CLI 展示 apply_patch 的 canonical diff

- 父版本：`317d21727220e8167d7cf31ca7332356ed2451c9`
- 目标提交：`b93dbfd12ce70c4b99cdd7daa86608334314a282`
- 任务：成功补丁生成有界 unified diff，经 ToolResult metadata 和流式 result details 到 CLI；不能污染模型可见 structured content。
- 隐藏验收：对应提交中的 `test_filesystem_tools_list_read_patch_and_expose_changes_immediately`、`test_cli_displays_patch_diff_from_existing_result_event`。

### 5. 阻断重复相同工具失败循环

- 父版本：`f4d478b0a0d5c5016edf8e7e42ef47d4480cae67`
- 目标提交：`df97c5ffd3c7b90704ff596967bb25b5244eb855`
- 任务：根据 canonical 调用和 checkpointed 结果阻断重复失败；允许改变参数恢复；不可抢占同批调用或绕过 reconciliation；持续重复时快速终止。
- 隐藏验收：对应提交中的重复失败恢复、持续重复终止、非幂等失败三项核心测试。

## 试跑后的决策

- 方法可复现且能区分失败原因：扩展到 20～30 例。
- 多数失败来自同一公共链路：只修最高频根因并复跑本批次。
- 任务本身无法独立还原：替换任务，不降低验收标准。

## 首轮结果（修复前）

| 任务 | Agent 正常结束 | 生产代码 | 隐藏验收 | 主失败 |
| --- | --- | --- | --- | --- |
| wheel 模型目录 | 否 | 是 | 1/1 | 已完成修改后继续探索，耗尽 Turn |
| OpenAI wire | 否 | 是 | 16/16 | 上下文预算与模型输入不一致，随后未主动结束 |
| update_plan 持久化 | 否 | 否 | 未进入验收 | 把搜索行号当字节 offset，30 Turn 都在阅读 |
| apply_patch diff | 否 | 是 | 0/3 | 跨层 metadata/details 契约理解错误 |
| 重复工具失败熔断 | 否 | 是 | 2/7 | retry、checkpoint、reconciliation 语义不完整 |

首轮端到端完成率是 **0/5**。这不是“测试不够”，而是 Agent 在真实任务上没有稳定的交付状态：即使代码已经正确，也不会可靠地验证并结束。

## 失败榜与本轮处理

1. **P0：交付收敛失败（5/5）**。连续浏览达到 8 次时提醒收敛，12 次时明确要求编辑，20 次时关闭广泛探索；模型可用一次 `update_plan` 换取最多 8 次聚焦检查，仍不交付则以 `delivery_stalled` 明确失败。`update_plan` 不能反复刷新额度。完成编辑等具体交付动作后才开始新的验证周期。没有新增第二套 Executor 或事件系统。
2. **P0：编码 ACI 误导模型。** `read_file` 新增一等的 `start_line/max_lines/next_line`；字节模式明确返回 `next_offset`，且两种模式不能混用。`search_text` 默认跳过生成目录、返回 2 行局部上下文，并把生产源码排在 tests/docs 之前；根目录 `list_files` 隐藏 `.venv/.rag/node_modules` 等噪声。省略 `regex` 时识别常见正则语法。
3. **P1：模型窗口与 stage budget 不一致。** provider 现在按模型窗口和 gateway 的有效 stage 输入预算两者较小值投影上下文；coding turn 的默认预算调整为 32k 输入、4k 输出。
4. **P1：真实验证命令无法使用项目解释器。** 受限沙箱只读开放 workspace `.venv/bin/python` 实际指向的那一个 uv-managed CPython 目录；host 环境变量、网络和 workspace 外写权限仍不开放。
5. **P1：模型供应商不是可忽略的产品依赖。** Groq 对本批任务触发 8k TPM 限额；`mimo_cloud` 能读到环境变量但服务端返回 401。默认模型改为已经真实跑通的 `deepseek_chat`，Groq 保留显式选择。供应商可用性单独记账，不能伪装成 Agent 代码失败。
6. **P1：计划不能在失败后继续自证完成。** `update_plan` 运行中仍完整保留模型提交的快照；但 Turn 最终失败时，没有 `tool_call_ids/evidence_refs` 的自报完成步骤会改为 `blocked`，有真实工具证据的完成步骤保留。

## 修复后真实复跑

- 小型公开 CLI 冒烟：Agent 读取两个文件、修改 `calculator.py`、在受限沙箱中从失败的 `python` 自动改用 `python3`，看到 `DELIVERY_SMOKE_OK` 后正常结束。结果 **1/1 端到端成功**。
- 默认路径冒烟：不传 `--model` 时解析为 `deepseek_chat` 并再次完成同一修改；从当前 worktree 不传 `AGENT_ENV_FILE` 运行，仍能自动发现共享 `.env` 并返回 `ENV_OK`。密钥可见性问题已用真实请求验证，不再靠推测。
- OpenAI wire 历史任务干净复跑：Agent 在交付提醒后提交计划、完成实现、运行两个 provider 测试文件并正常结束。注入目标提交隐藏测试后为 **15/16**；唯一失败是异常文本未包含目标测试要求的 `non-leading system`，行为语义本身正确。
- update_plan 跨层任务复跑：行号/字节误用已经消失，源码优先搜索也能直接找到 `PlanState` 和 builder stub；但 `deepseek_chat`、`deepseek_v4_flash`、本地 Qwen3.5-9B 都在 28 次有效检查内只解释架构、不提交编辑，最终明确结束为 `delivery_stalled`。这证明当前可用模型栈不具备稳定的跨层编码能力，不能靠继续增加 Runtime 层或放宽 Turn 假装解决。

结论：Agent 已经能稳定完成局部编码任务，也能对不收敛任务给出有边界、可诊断的失败；但它还不是任意仓库任务的生产级 coding agent。当前产品边界应明确为**局部、可定位、可用聚焦命令验证的修改**。跨层架构改造仍需更强模型或先拆成有明确文件边界的子任务。下一批应扩到 20～30 个历史任务，并把“首次运行正常结束且隐藏验收通过”作为唯一主指标。
