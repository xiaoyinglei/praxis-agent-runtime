# Praxis 默认工具表与开关矩阵（阶段 4 T16 收口）

组合根 `praxis/composition.py::build_session`（`sdk.open_session` 是其唯一公开别名）
装配默认工具表。本文档钉住默认表集合/稳定序与两个尾部开关的口径；机械回归钉在
`tests/praxis/test_default_table3b.py`（集合/数量/wire tools/替换路径），
memory 面另见 `tests/praxis/test_memory_context.py`、`tests/praxis/test_memory_cli.py`。

## 默认表：15 builtin（两开关缺省关，旧 parity 零扰动）

稳定序（`_default_tools`，测试钉序）：

| # | 工具 | 工厂 |
|---|------|------|
| 1 | `read` | `create_read_tool(root)` |
| 2 | `write` | `create_write_tool(root)` |
| 3 | `bash` | `create_bash_tool(root, sessions=…)` |
| 4 | `manage_command` | `create_manage_command_tool(sessions)` |
| 5 | `list_files` | `create_list_files_tool(root)` |
| 6 | `search_text` | `create_search_text_tool(root)` |
| 7 | `inspect_data_file` | `create_inspect_data_file_tool(root)` |
| 8 | `apply_patch` | `create_apply_patch_tool(root)` |
| 9 | `update_plan` | `create_update_plan_tool()` |
| 10 | `run_python` | `create_run_python_tool(root)` |
| 11 | `read_artifact` | `create_read_artifact_tool(artifact_store)` |
| 12 | `web_fetch` | `create_web_fetch_tools(client, …)` |
| 13 | `web_search` | `create_web_search_tools(client, …)` |
| 14 | `invoke_skill` | `create_invoke_skill_tool(skills_catalog)` |
| 15 | `materialize_skill_asset` | `create_materialize_skill_asset_tool(root)` |

- MCP 动态工具**不进入**默认表，只经 `extra_tools` 并入（重名边界拒绝）。
- `tools=` 整体替换时默认装配路径不执行：不构造 web client、不装渲染器、
  不读 web 配置（capability coupling，替换路径保持廉价）。
- `read_artifact` 与 ArtifactStore 注入耦合：替换表无 `read_artifact` ⇒ 不注入
  artifacts，管道退回字节截断语义。
- web 面与 Session 同源：`web_fetch`/`web_search` 共用同一 `WebLimits` 与同一
  authorize 闭包（读 live `session.permissions`，不缓存策略快照）。
- 渲染探针 `_playwright_importable()` 是 monkeypatch 接缝：False ⇒ 纯文本面；
  True ⇒ render-capable（`EXECUTE_PROCESS`）。base 安装（playwright 缺失）
  探针 total ⇒ False，默认会话仍可打开。

## 开关矩阵（尾部默认零扰动，正交可叠加）

| `enable_subagents` | `enable_memory` | 表规模 | 追加 |
|---|---|---|---|
| False（缺省） | False（缺省） | **15** | — |
| True | False | **20** | agent 五件套（表尾） |
| False | True | **16** | `read_memory`（表尾） |
| True | True | **21** | agent 五件套 + `read_memory` |

### `enable_subagents=True`（T6）

- 表尾追加五件套（`create_agent_tools` 稳定序）：`spawn_subagent` /
  `wait_subagent` / `send_to_subagent` / `cancel_subagent` /
  `integrate_child_workspace`。`tools=` 整体替换路径同样追加（重名边界拒绝）。
- 四件编排（spawn/wait/send/cancel）自动附加 **session 域 ToolRule allow**
  （`AGENT_ORCHESTRATION_NAMES`）：预算份额 + 子限额才是闸门；allow 只活在
  内存，**绝不进 always 落盘**；显式 policy deny 仍跨源最严胜出。
- `integrate_child_workspace` 是 `DESTRUCTIVE + WRITE_WORKSPACE`：显式审批
  不可豁免（S5），不在 session 域 allow 集。
- `SubagentManager` 在 Session 之后构造（parent 引用），经 holder 闭包绑定
  工具面，并经 `_SubagentManagerCloser` 挂收尾链（integrations 最前）。
- `subagent_limits` 覆盖 `max_concurrent/max_depth/max_total`
  （缺省 `SUBAGENT_LIMITS_DEFAULT = {4, 2, 16}`；未知键/负值/bool 边界拒绝）。
- write 子隔离由 `GitWorktreeProvisioner(child_env)` 承载（T7，落点
  `host_worktrees_dir()`，非地板私有集）；read_only 子共享父工作区，
  `persist_always=False`（always 只改内存不落盘，防写父 permissions.json 泄漏）。

### `enable_memory=True`（T11/T12/T13）

- 表尾追加 `read_memory`（普通只读工具，`create_read_memory_tool`；state 读取
  缝在组合根绑定，每次调用现读当前 committed state）。
- session 域 allow 与 read/list_files 同姿态（内存域，绝不进 always 落盘）。
- 注入 `MemoryDriver`（Q6 进程内 async 消费）；maintenance turn 经唯一 spawn
  原语（`source="memory"`），因此 manager 同样构造——但**不挂 agent 工具面**
  （工具面仅 `enable_subagents`）。子 Session 不继承本开关。
- 人类面 CLI：`praxis memory status/remember/edit/delete/explain [--json]`
  （只经 sdk 公开助手，`praxis_cli_via_sdk` 契约）。

## 子工具表重构（R-T8-1，与开关无关的装配纪律）

子装配经 `parent_tools` 通道：builtin 按**子会话工作区根**经既有工厂重构
（子的 CommandSessions/ArtifactStore/skills/web authorizer 全绑子自身），
非 builtin（MCP extra/测试玩具）保留父实例，agent 五件排除（depth 结构门）。
`composition._BUILTIN_TOOL_NAMES` 必须与默认表集合逐字一致
（`test_child_rebuild_builtin_name_set_matches_default_table` drift 钉）。

## 预算默认（T15 Q8，与工具表耦合）

工具表含 web 工具族（`WEB_TOOL_NAMES`）时，root `Limits` 的
`web_requests/browser_requests` None 维折入 `WebLimits.turn_requests/
browser_requests`（默认 32/64）；显式维优先，宿主 web.json 可调。无 web 工具
的替换表不读 web 配置不折叠。子 Session 的 grant 由份额公式
（`limits_merge_grant`，`MAX_CHILD_FRACTION=0.5`）从父剩余切出，逐维有界，
折叠对子天然 no-op。

## CI 机械强制（T16）

- 默认表/开关口径：`tests/praxis/test_default_table3b.py`。
- 份额公式/payload 拼装/清算字面/git 工作区命令单一来源：
  `scripts/check_anti_patterns.py` 规则 25–28（计划编号㉒–㉕；㉑=规则 24，
  T10 已建），负形样本在 `tests/praxis/test_anti_patterns.py`。
- 分层契约：`.importlinter` `praxis_tools_adapters_boundary` 覆盖
  `adapters/workspaces` 不得 import core（第 15 契约评估结论：既有契约已覆盖，
  不新增；覆盖关系由 `test_workspaces_core_forbidden_covered_by_existing_contract` 钉）。