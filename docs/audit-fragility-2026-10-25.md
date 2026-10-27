# agent_runtime 脆弱性与规则堆积审计报告

- 分支：`audit/fragility-review`
- 日期：2026-10-25
- 范围：`agent_runtime/` 全部 128 个 Python 文件（约 60,000 行），7 个模块簇逐一深审，交叉核对 git 历史（179 个提交）与测试目录
- 方法：每个文件通读；每条发现要求 file:line + 代码证据 + git 补丁来源；关键断言已由主审抽查复核

---

## 一、总体结论

1. **骨架是认真的，补丁是失控的。** 事件溯源 + reducer + fencing/lease 的持久化骨架、沙箱安全面（DNS pin、TOCTOU 反复 recheck、seccomp、凭据权限校验）都属高水准。但恢复/兼容/呈现层呈现明显的"事故驱动补丁"特征：git 历史里 20+ 个 `fix: harden/recover/handle` 提交与代码内的场景分支一一对应。
2. **"为特殊问题不停加规则"不是个别现象，而是系统性模式**，已归纳出六种形态（见第三节），每种都有 git 铁证。最典型的一条链：pydantic 输入审计黑名单由 **11 个连续补丁提交**堆成（`649dfb4`→`3b0550f`，全部是"遇到一种不支持形态就加一条 reject"）。
3. **自证证据**：仓库自带 `scripts/check_anti_patterns.py`，专门扫描 `rag/` 目录的硬编码模型名/provider 名并报错退出——但它**不扫 `agent_runtime/`**。旧子系统的反模式检查器存在，而同类特判在新子系统重新长出至少 18 处（见 3.6 节清单）。
4. 代码里 **0 个 TODO/FIXME 标记**、64 处 `except Exception`——欠账从不标记、异常经常静默，问题只在线上暴露后才被补。

---

## 二、全局性设计缺陷（跨模块）

### D1. 上帝模块（四个文件占全库 1/4）

| 文件 | 行数 | 实测职责数 |
|---|---|---|
| `harness/rollout.py` | 6163 | `RolloutStore` 单类 5360 行、约 100 个公开方法、18 类职责（schema/事务/投影/Turn 状态机/交互/审批/模型操作/预算/工具操作/计划投影/上下文/压缩/工件 GC/记忆 KV/限额/验证产物/序列化/ID 派生） |
| `cli.py` | 2903 | 20 类职责（命令注册、5 个领域的 CLI 表现层、聊天 REPL、审批 UI、持久化投影、结果渲染） |
| `harness/turn.py` | 2642 | Turn 执行 + 4 套恢复入口 + 双压缩计划器 + 上下文编排 |
| `tools/builtins/shell.py` | 2234 | 双平台沙箱 + seccomp + 工作区发布 + 描述补丁堆 |

`cli.py` 内 `_chat_facade_loop`（cli.py:989 起）单函数约 777 行，内含 6 个闭包、20 个 `if query == "/x"` 手写分派链、12 个 nonlocal 可变状态。

### D2. Turn 状态机没有主人

"账本处于什么状态、下一步合法做什么"的知识分布在至少 5 处：rollout 写侧方法、reducer（`reducer.py:466-488` 纯赋值、**零转换校验**，`turn_paused` 不检查当前是否 running）、`session.resume` 的 9 分支 if/elif（`session.py:1184-1304`，其中两个分支动作完全相同、只是谓词叠加）、turn.py 的 4 个 `recover_*` 入口（turn.py:640-914）、`ownership.recover_owner_loss`。**每出一种新故障形态，就往其中一条链加一个分支**——这是恢复层所有规则堆积的共同根因。实际产物：`resolve_clarification`/`resolve_choice`（rollout.py:1395-1456, 1497-1562）漏掉了 `receive_turn_input`（rollout.py:727-728）已有的取消检查，用户已发 durable 取消后仍可把 Turn 恢复运行。

### D3. 模型可见消息文本被当作内部数据总线

注入方与解析方分属不同文件、无共享常量：

- `"Runtime input limits:\n"` — model_adapter.py:262-264 注入，model_adapter.py:820-821 靠 `startswith` 找回去重
- `"Runtime budget status:\n"` — turn.py:572-574 `removeprefix` 后 `json.loads` 反解析结构化状态
- `"Context compaction:\n"` — protocol.py:37-41 靠内容前缀给历史消息打迁移分类
- `" restore "` — agent.py:360 靠 subject 文案里恰好含这个词识别回滚锚点（GitDelivery 改个措辞，回滚能力静默消失）

改一个字符即静默失效，且 prompt 文案与内部协议混在同一字符串里。

### D4. 政策 revision 字符串是事实上的类型系统，但没有类型

`"working-context-v1/v2"`、`"completion-v1/v2"`、`"append-v1"`、`"pressure-v1"`、`"one-replan-v1"`、`"semantic-compaction-v3/v4/v5/v6"` 等 28+ 处字面量散布 7 个文件（turn.py 14 处、services.py 8 处、model_adapter.py 3 处……）。消费端甚至用**整字典字面量相等**做匹配：

- turn.py:2191 `policy == {"revision": "restart-step-v1", "max_retries": 1}` — 未来加一个 `note` 键，恢复策略静默失效
- turn.py:2272 用整段 JSON 文案（含自然语言 message）做"恢复机会已用掉"的去重键 — 文案微调后旧 Turn 被判定"没用过"，再送一次恢复
- rollout.py:4517 `reset == {"revision": "v1", "reason": "memory_revoked"}` — reason 改一个词，被撤销的记忆重新进入上下文

### D5. v1/v2 双轨长期并存而非迁移收敛

- completion.py:22-199：v1 与 v2 **两套完整判定实现并存**（工作区变更检测一套按 sequence、一套按 record_id 排序）
- events.py：`payload_schema_version == 1` 兼容分支散布 9 处（347/403/406/424/575/617/690/730/869），而 reducer.py 已把同类迁移收敛在 `_upcast_payload` 单点——同一 concern 两种组织方式
- turn.py:1198-1409：`_plan_capacity_compaction` 与 `_plan_legacy_compaction` 两个 100+ 行、语义 80% 重叠的规划器并存，无删除时间表
- streaming/events.py：同一枚举保留两组生命周期（`TURN_COMPLETED` 与 `TURN_END`），渲染层为 legacy 事件维护 4 个 `_render_legacy_*` 分支

### D6. 私有 API 穿透是常态

- `store._transaction()` 被 5 个外部模块直接调用（ownership.py:76、session.py:362/525、memory_management.py:65、tool_orchestrator.py:272）
- harness 层跨层 import 工具层私有符号：`from agent_runtime.tools.builtins.shell import _normalize_command_output`（tool_orchestrator.py:232、command_recovery.py:10）
- runtime/mcp.py:547-564 一次调用 shell 模块 5 个下划线私有函数
- turn.py:1462 通过 `isinstance` 探测后调用 ContextManager 的 `_projected_messages` 等 10+ 个 Protocol 未声明成员——protocol.py 的 Protocol 声明形同虚设
- MCP 六个文件（runtime/mcp、mcp_catalog、mcp_oauth、mcp_admin、connection_manager、mcp_cli）互相 import 私有符号

### D7. 热路径全量扫描 / 全量哈希

- rollout.py：每次模型派发、工具认领、上下文组装都对库内**全部** artifact 逐个整读 + SHA-256（`_assert_artifact_integrity`，9 处调用）；`read_budget_state` 每次派发载入整个 Thread 全部记录；旧哈希 Thread 的下一次写会在 `BEGIN IMMEDIATE` 写锁内触发全库 `verify()` 重放
- completion.py:209-236：每次模型 Stop 全库扫描 Turn/Thread/Record 重建排序
- events.py:169-203：`read_global` 每次轮询全量重投影再丢弃大多数
- shell/executor：一次 `run_command(workspace_write)` 至少 5 次全树遍历 + SHA-256×2
- 命令会话清理解析 `/bin/ps` 文本输出且无超时（web_browser.py:273-286）

长会话/大仓库下延迟线性劣化，是"自己拖垮自己"的结构性风险。

### D8. 异常静默（64 处 `except Exception`，最危险 10 处）

| 位置 | 行为 | 后果 |
|---|---|---|
| core/llm_registry.py:457-466 | 构建 generator `except Exception: pass` | 任何根因丢失，只剩 "Failed to build provider"，排障靠猜 |
| mcp_oauth.py:458-459 | OAuth 回调 `except Exception: pass` | state 不匹配（疑似注入）与畸形请求不可区分 |
| mcp_oauth.py:551-552 | 登录失败 `raise ... from None` | 超时/Keychain 拒绝/网络断/代码 bug 折叠成同一句话并销毁因果链 |
| file_manifest.py:480-481 | Excel 扫描 `except Exception: pass`（连日志都没有） | 全库最沉默的吞异常 |
| git_delivery.py:464-469 | `gh run list` 失败/坏 JSON → `return ()` | CI 修复工作流拿到空日志继续跑，用户无法区分"无日志"与"采集坏了" |
| connection_manager.py:163-167 | 连接任务统一中文文案 | 有意防泄漏但无 debug 逃生口，所有失败不可区分 |
| mcp_catalog.py:196-198 | descriptor 反序列化失败 → 降级占位工具 | 缓存 schema 漂移被伪装成"服务需要登录" |
| modeling/gateway.py:1099-1103 | 流取消 `except Exception: pass` | 取消失败零观测，泄漏无从追踪 |
| runtime/mcp.py:584-585 | MCP 关栈失败仅 warning 后继续 | 5 秒宽限后 stdio 子进程可能仍存活 |
| tools/executor.py:1120 | trace sink `except (CancelledError, Exception): pass` | 后台异常有意消费，分类错误副作用静默丢失 |

---

## 三、"为特殊问题不停加规则"的六大模式（重点）

### 3.1 模式一：字符串当协议

用人类可读文案 / 整段 JSON / ID 后缀做机器判定键，改一个字符即静默失效：

- **Bing 403/429 靠错误文案跨文件匹配**：web.py:322-324 匹配的字符串正是 web_http.py:544 生成的 `"The remote server returned HTTP 403."`——正确做法是 error 携带 status_code 字段
- **apply_patch 用 message 反查错误码**：filesystem.py:235-242 `_PATCH_ERROR_CODES` 把 message 文案当 key，退化兜底 `.get` 后下游分支全部失准
- **gh CLI 错误文案匹配**：git_delivery.py:452-457 `if "no pull request" in detail.lower()`
- **异常 message 分流**：session.py:968-969 `if str(error) != "resource_limit_exceeded": raise`；cli.py:638-645 用 `str(error).startswith("Web proxy must")` 决定 CLI 参数提示
- **整段 JSON 文案做去重键**：turn.py:2264-2276（见 D4）
- **request_id 后缀携带控制流**：turn.py:1815 写入 `":context-retry:1"`，turn.py:555 靠 `endswith` 识别
- **bwrap stderr 前缀判断沙箱失败**：shell.py:582/609/797/823 四处重复，且 582 用 bytes、609 用 str 两套写法

### 3.2 模式二：手工清单无单一事实来源

- **保护目录清单 3 处独立维护**：shell.py:63-67（Python 常量）、shell.py:1749（`_LINUX_VIEW_PROTECTED`）、shell.py:1654-1674（Seatbelt 正则再手工展开一遍大小写类）；filesystem.py:30-47 又有一份内容重叠的。`.codex`（竞品工具）、`.aws`（凭据）明显是逐案追加，`.ssh`/`.gnupg` 还没遇到所以不在名单
- **provider 集合分派复制 4 处**：`provider in {"mlx","ollama"}` 在 gateway.py:111/285、model_adapter.py:235、local_agent_wire.py:29——而 `supports_native_tools` 能力位本应足够
- **provider→kind 集合已与 catalog 漂移**：llm_registry.py:724 的集合缺 catalog 里已有的 `kimi`、`minimax`；`.env.example` 有 `MIMO_API_KEY` 而 catalog 无此条目
- **工具名集合散布 4 文件 12+ 处**：`{"run_command","manage_command"}` 与 `{"web_search","web_fetch"}` 在 terminal_render.py、terminal_ui.py、cli.py、permissions.py 重复出现；permissions.py:236-281 通用权限裁决函数里写死这两个名字（已有 `PUBLIC_WEB_READ` profile 这一通用判据后又叠工具名白名单）
- **停止原因归一化写了 3 遍**：gateway.py:434-441、chat.py:236-243、openai_wire.py:166-173，集合还不一致
- **macOS 网络白名单逐案枚举**：shell.py:1654-1674 硬编码 mDNSResponder、8 个 mach 全局名——"每遇一个服务报错就加一行"
- **工具检索靠手工中英文别名表**：selection.py:57-100，10 个工具逐个写别名；未加别名的 MCP 工具对中文查询基本不可发现
- **workspace 忽略目录 13 项枚举**：workspace.py:18-34，每出现一种新工具缓存就要人工加一条，漏加 = 快照噪声 + 误判文件变更

### 3.3 模式三：按版本号/代际堆分支（兼容逻辑 O(版本数) 增长）

- **压缩算法版本枚举 6 个代际**（v2/v3/v4/v5/v6），散布三处：context.py 8 处、context_recall.py:86-93 第三次按 v5 特判归档集合、rollout.py:1104-1124 存储侧再加 v5/v6 版本门。git 显示 `2660257` 一次引入 v3/v4/v5 三代、`933a748` 加 v6——每修一个压缩 bug 就立一个新版本号并在所有读取侧追加分支
- **`payload_schema_version == 1` 散布 9 处**（events.py，见 D5）
- **双轨记账兼容**：rollout.py:2701-2708 注释自认 "Compatibility for Turns created before durable reservations existed"，预算 correctness 依赖"是否被预留表追踪过"这一历史分界；全文件至少 5 处独立"旧数据兼容"补丁各自为政
- **7 连"缺列则 ALTER"**：rollout.py:5586-5610，每次 schema 变更永久追加一个特判分支，`PRAGMA user_version` 从未递增，无统一迁移框架
- **CLI 旗标别名堆积**：`--last`/`--continue`/`--resume` 三入口互斥还要专门校验（cli.py:2335）；`--max-steps`/`--max-turns`、`--rollout-db`/`--checkpoint-db` 双名并存

### 3.4 模式四：为具体工具/网站/服务商硬编码

- **Bing 结果页 HTML 结构整套特判**：web_content.py:103-135 硬编码 `b_algo`/`b_no`/`b_caption` 类名、验证码页特征、`/ck/a?u=a1...` 跳转的 base64 解码——Bing 改版即整体退化为 `web_search_invalid_response`，无降级策略
- **Tavily/Bing 搜索服务商 if/elif 特判散布整个 search 函数**：web.py:304-376, 646-673，字段名、时间映射（`{"pd":"ez1","pw":"ez2","pm":"ez3"}` 魔法串）、错误文案、`execution_revision` 全部逐案硬编码
- **通用存储方法里为单个工具写 150 行特判**：rollout.py:3912-4001 `record_tool_result` 内嵌 `tool_name == "update_plan"` 的完整解析
- **通用消息层写死工具名**：core/messages.py:210-216 `result.tool_name == "web_fetch"` 的字段裁剪白名单
- **通用结果投影写死工具名**：result.py:278 `if tool_name != "search_knowledge": continue`；result.py:186-188 靠 `"integration-mcp-v1:"` 协议魔数前缀区分 MCP/原生工具
- **编排层按工具名/版本串分派**：tool_orchestrator.py:748 `record.tool_name == "run_command"` 决定事件类型；833-834 `policy.get("revision") == "completion-v2"` 按历史版本号保留旧 gate
- **GitHub 专属规则贯穿通用 MCP 注册层 4 处**：mcp_admin.py:129-142, 203-246（专属默认 URL、专属工具白名单、专属 header、gh-token 流程）；connection_manager.py:62 又为 Linear 单独定 scope；gh 可执行文件用 3 个硬编码 POSIX 路径探测（不用 `shutil.which`）
- **验证器按可执行文件名硬编码 if 链**：verification.py:197-220 只认 pytest/unittest/mypy/ruff/cargo/go/npm/pnpm/yarn，封闭名单；每工具一套选项黑名单（245-266）；手写 uv CLI 解析器 hardcode 21 个选项（270-315），uv 新增开关 → 整条命令判定不可验证
- **解释器黑名单 `.*sh` 误杀漏放**：approval.py:37, 87-89——实测 `push/flush/flash` 被误判为解释器，`Rscript/java/awk/deno` 反而不在名单
- **web 工具特例渗入通用上下文层**：context.py:984-1083 对 web 输出 envelope 做结构级特判 + ~30 个键名的手工白名单两份

### 3.5 模式五：工具描述被当补丁堆

工具描述（ACI）里堆满针对既往失败的具体教诲，每次模型行为纠偏都改描述：

- web_fetch 描述 65+ 行（web.py:680-746）："Resource limits and 403 will not be fixed by repeating an unchanged request"、"changed means text differs, not business requirements changed"……
- run_command 描述成段教 pytest 用法与毫秒/秒单位（shell.py:346-376，git `4897577` 毫秒单位补丁沉淀为描述文案）
- filesystem 一天内 5 个 commit 的加-撤-再加循环（2026-08-06：dc41b19→1e4c168→674643c→38b971a→b9e7ca3），最终沉淀为描述里的 "Never reapply the patch merely to reconfirm it"

与 AGENTS.md"精心设计 ACI、保持简洁"的原则相悖：描述在膨胀而非收敛。

### 3.6 模式六：数据问题被写成代码

最讽刺的一类——**现成的数据通道不用，靠代码识别身份**：

| 位置 | 特判对象 | 本应走的通道 |
|---|---|---|
| llm_registry.py:561 | `provider_name == "deepseek" and model_id == "deepseek-flash"` 补旧档案缺字段 | 档案升级一次性迁移（catalog 新条目已带该字段） |
| tokenization.py:267-274 | `model_name == "deepseek-v4-official"` 挂载捆绑 tokenizer + 内联 sha256 | 模型定义的 `tokenizer_model` 字段 + 资产清单 |
| web.py:284 | 工具层再写死一次 `"deepseek-v4-official"` | 同上（第二处硬编码） |
| server_model_gateway.py:31-34 | 模型→provider→端点映射写死在代码 | 部署配置文件 + 启动校验 |
| ollama/generator.py:20 | 默认模型 `"qwen2.5:7b"` | catalog |
| providers/ollama | "成功率 99.99%" 注释的无依据承诺 | — |
| shell.py:1768-1776 | 只认 uv 安装路径的解释器信任 | "venv 内解释器真实路径"这一通用属性 |

**铁证**：`scripts/check_anti_patterns.py` 的存在说明团队完全知道"硬编码模型名是反模式"，但检查器只覆盖 `rag/`；`agent_runtime/` 里同类特判重新长出 18 处。正面反例也存在：`core/generation_settings.py` 明说 "Wire controls are declared by the model, never inferred from its name"，`budget/pricing.py` 完全数据驱动——正确做法是知道的，只是没贯彻。

### 3.7 git 交叉证据：反复 patch 热点

| 位置 | 补丁链证据 |
|---|---|
| tool.py pydantic 审计（813-1241，430 行黑名单） | **11 个连续 pydantic 补丁提交**（649dfb4→3b0550f，历史路径 rag/agent/tools/tool.py），全部是"加一条 reject" |
| shell.py seccomp socketpair（1259-1267） | 逐位白名单 + `369498f` 为某平台 AttributeError 再补 2 行 |
| shell.py darwin 强杀进程组（663-664） | `6481175` 为一次 hang 打的平台补丁 |
| mcp_oauth.py:523 流式服务器 405 特判 | `e9ab1c6 "fix: handle streaming OAuth servers"` 为具体服务器行为打补丁 |
| rollout.py `record_migrated_tool_operation` | `2bded9b` 引入、`f85b0a2` 再删——先打补丁后拆补丁的完整轨迹 |
| cli.py / turn.py / model_adapter.py / gateway.py / llm_registry.py | 各 19-30+ 提交，多为 harden/fix/coordinate/align 连环 |
| web.py provider 分支 | `b324587` 一次塞入 Tavily 全套特判 |

---

## 四、分模块高危发现（精选，全部经主审复核或交叉印证）

### 4.1 执行内核（rollout / turn / session）

- **rollout.py:4820-4843**：COMMIT 失败（磁盘满/IO 错）走不进 except 分支，连接可能悬挂在打开的事务；未设 `busy_timeout`，多进程争写超 5 秒直接 `database is locked`。**高**
- **rollout.py:2313-2326**：有效且已计费的模型响应在 `observed_at > lease_expires_at`（`<=` 精确比较）时**静默丢弃**，随后被标 unknown 并中断 Turn。**高**
- **turn.py:2103-2132**：generic `except Exception` 把未分类瞬时错误（429/5xx 不在 `model_adapter.py:522-537` 的 (模块名,类名) 字符串元组名单内）判为**确定性失败直接终局**，而非 unknown→可恢复。**高**
- **turn.py:2317-2406**：`_repeated_tool_failure` 内为 `filesystem_denied`/`file_not_found` 各写一条签名改写规则，且在纯逻辑判定里调 `Path.exists()`——同一账本重放结果不确定。**高（规则堆积）**
- **session.py:1184-1304**：resume 9 分支隐式状态机，1268 与 1291 两个分支动作完全相同（都调 `recover_committed_model_response`），后者只是前者加 purpose 过滤的补丁未合并。**高**
- **checkpoints.py:62 vs 353-368**：restore 先删后还原，中途失败只回滚 ref，工作区停留在半恢复态——与 docstring "workspace files were not harmed" 直接矛盾。**中**
- **checkpoints.py:441-455**：prune 在锁**外** rmtree 整个 shadow 仓库，与并发 snapshot/restore 无互斥。**中**
- **ownership.py:91**：生产路径用裸 `assert owned and fencing_token is not None`，`python -O` 下防线消失。**中**

### 4.2 工具子系统

- **shell.py:801 vs 588**：execute_python 的 Linux 发布阶段**同步直调** `_commit_linux_workspace_view`（全树扫描+逐文件复制），卡死事件循环数秒到数分钟且不可取消——run_command 路径已包 `_blocking_phase`，这是双份逻辑漂移的漏改。**高**
- **tool_orchestrator.py:428-429, 631**：progress 先于 claim 到达即 `raise RuntimeError` 炸掉整个工具执行——接口契约靠运行时异常执行。**中**
- **executor.py:1107-1121**：`_finish` 的 drain 循环在 record_sink 挂起时无法取消（`await _emit_record` 无超时）。**中**
- **web_browser.py:273-286**：浏览器清理靠解析 `/bin/ps` 文本，失败静默降级（僵尸 Chromium 进程）；`communicate()` 无超时，ps 挂起会让 finally 中的 shield 永久挂住。**高**
- **filesystem.py:691→731**：apply_patch 读-改-写无并发防护，外部编辑器保存的改动被 last-writer-wins 覆盖。**中**
- **data.py:225-232**：检查 worker 无 rlimit 无本地超时（对照 web_content.py:58 同款 worker 两者都有——标准不一致），zip 炸弹可在窗口内吃满内存。**中**
- **permissions.py / executor.py / workspace.py**：路径 containment 三套实现并存，对 symlink/缺失路径/大小写处理各不相同。**中**

### 4.3 模型子系统

- **整个 provider 层零重试/退避**（gateway.py、chat.py、providers/ 全文 grep 无 retry/backoff）：一次 429/502 = 一次整个 agent step 失败。**高（成熟度最大缺口）**
- **chat.py:121**：`stream_options` 无条件硬塞，旧 vLLM/部分网关直接 400，且无 per-model 开关。**高**
- **server_model_gateway.py:283-291**：`stream=True` 请求被网关**整体缓冲后一次返回**——名义 text/event-stream，实际首 token 延迟等于全量生成时间。**高**
- **tokenization.py:44-60, 267-319**：token 计数三层不准兜底（无 tokenizer → UTF-8 字节冒充 token；流式 delta 逐段独立计数跨界重复/漏计；输入侧拍平成纯文本近似），且退化**静默无告警**。**中**
- **tokenization.py:312-317**：`trust_remote_code=True`——对一个满是 HMAC/0600/ACL 校验的信任体系，这是明显不一致的口子。**中**
- **gateway.py:526, 580, 658**：lease_id 用 `id(prompt)` 生成，CPython 地址复用可顶掉旧账目。**中**
- **models.py:1184-1185**：`resolve_or_fallback` 声称实现 Protocol 却悄悄丢弃 fallback 语义——接口契约欺骗。**中**
- **概念过密**：`ModelSpec` 3 个不同定义、`ModelRuntimeConfig` 2 个、registry 家族 6 个实体——与 AGENTS.md"减少不必要抽象层"直接冲突。

### 4.4 上下文 / 记忆 / 预算 / 技能

- **skills/runtime.py:88**：`_ARGUMENTS_RE.sub(args, instructions)` 把用户参数当**正则替换模板**——实测 `args="C:\new\folder"` 被静默损坏、`args="path\1target"` 直接抛 `re.error`。模型传路径/正则参数必然踩中。**高（已实测的真实 bug）**
- **memory.py:265-266 vs 329-330 vs 413-435**：同一记忆预算期初按 `exposure` 扣、事后按 `used` 扣、失败再按 exposure 判——三种口径混用，系统性漂移。**高**
- **skills/policy.py:33**：`trust_external_skills: bool = True`，与 docstring "External skills require explicit opt-in" **相反**；skills/models.py:24-27 默认禁用 BUNDLED 却给它最高优先级——两处自相矛盾。**高**
- **skills 安全声明字段解析后不执行**：manifest 的 `allowed_tools`/`path_patterns`/`can_autoload` 全局 grep 无生产调用，catalog.py:199-203 注释自认 "path-conditional skills not yet implemented"——**安全面声明是死的**。**高**
- **budget/accounting.py:104**：settle 按"实际用量是事实"入账但不复查上限——预估偏小即静默越限，`subtract_floor_zero` 逐字段取 0 掩盖记错。**中**
- **记忆无淘汰**：`MAX_MEMORY_ENTRIES=32` 只限单批提取，存储层 entries/entry_history/`:version:N` 快照行无限累积。**中**
- **上下文字节护栏两套矛盾默认**：context.py:50（4MB）vs modeling/contracts.py:49-52（32MB），且字节↔token 换算无恒定汇率（budget 用逻辑 token、无 DeepSeek tokenizer 时用 UTF-8 字节、context 用 JSON 字节——三头记账）。**中**
- **planning.py 的 PlanTracker 生产路径死代码**：仅测试引用；harness 实际用 `model_visible_plan` 直读——测试覆盖的是不接线的那套状态机。**中**

### 4.5 MCP / 交付 / 基础设施

- **git_delivery.py:238-255**：`switch -c` 成功后 commit 失败（pre-commit hook 改文件等）**无任何回滚**——用户被留在新建分支、变更已暂存未提交，与 docstring "working tree is byte-identical" 矛盾；27 个测试无一覆盖此路径。**高**
- **git_delivery.py:154-168, 281-290**：所有 git/gh 子进程**无超时**，`git push`/`fetch` 经 `asyncio.to_thread` 后整个会话可无限挂起。**高**
- **git_delivery.py:332**：`"github.com" not in remote_url` 子串包含——GHES 企业域名被拒、`gitlab.com/github.com-mirror` 反而通过。**中**
- **runtime/mcp.py:351, 425**：每次 MCP 调用 / 每个 HTTP 请求**同步读盘 + SHA256 整个配置文件**（async 路径上的阻塞 IO）。**高**
- **mcp_oauth.py:82-96**：Keychain 子进程同步调用（最长 15-30 秒）在 async 路径上阻塞整个运行时。**高**
- **file_manifest.py（799 行）整模块零生产调用**：`build_file_manifest` 唯一调用方是它自己的 21 个测试——799 行 + 21 测试维护着一个未接线（或已弃用）的功能；期间还藏 4 次全量加载 Excel（其中一次 `read_only=False` 全量载入内存）。**高（决策点：接线或删除）**
- **connection_manager.py:250-251**：`yaml.safe_load` 抛 `YAMLError`（非 OSError/ValueError）击穿 except——ThreadingHTTPServer 线程崩溃、浏览器连接重置；"配置写坏"这个最可能的用户错误恰恰没防住。**中**
- **python_verification.py 已退化为 31 行兼容启动器**（`566b961` 删掉了 66 行 AST 断言插桩观察器），但 shell.py:781 仍给它传死参数——历史机制拆除、兼容包袱未清。**低**

### 4.6 CLI / 终端 UI

- **terminal_app.py:1121-1141**：`operation()` 的 finally `gather` 无超时——UI 退出路径可永久卡死；`self._operation` 可被覆盖失去取消句柄。**高**
- **cli.py:1600-1645**：两处取消路径的 Turn ID 来源不一致（事件流最后看到的 vs 本轮结果的），`assert cancelled is not None` 在返回 None 时直接崩。**高**
- **terminal_app.py:590, 944-957**：鼠标审批按钮靠全局按键序号精确 +1 配对——resize/输入法/拖动都会让点击静默失效。**中**
- **terminal_render.py:74-77**：截断的 OSC 8 序列跨 delta 时，URL 片段作为可见文本打印（工具输出可注入对话显示）。**中**
- **terminal_app.py:701-755**：文本选择按字符索引推进，鼠标坐标是显示列——CJK/emoji 系统性错位；git 显示宽度问题已修补 3 轮未收敛。**中**
- **cli.py:1537-1545**：async 事件循环内同步读 SQLite，历史库大时每次发消息前 UI 冻结。**中**
- **terminal_input.py:65-71**：粘贴恰好全是文件路径的文本被静默改判为附件，判定基于 cwd，失败无提示。**中**
- **text.py:13-20**：`looks_command_like` 枚举 `"uv run"`、`"curl -x"`、`"rag_"`、`"ollama "`——打地鼠式关键词启发式。**规则堆积**

---

## 五、测试覆盖盲区

覆盖广度总体不错（harness 40+ 测试文件、git_delivery 27 测、mcp_oauth 17 测），但存在两类结构性盲区：

1. **核心模块零直接测试**：`model_admin.py`（314 行，ModelAdminService 全仓库测试零引用）、`setup_cli.py`、`command_cli.py`、`runtime/builder.py`。
2. **覆盖假象**（更危险）：
   - file_manifest.py 21 个测试守护的是**零生产调用的死代码**；"功能从未接线"这件事本身没有任何测试或契约能发现
   - planning.py PlanTracker 有专门测试文件，但生产路径用的是另一套内联实现——测的是不接线的
   - git_delivery 的 `"no pull request"` 文案匹配由 mock 固定文案维持，真实 gh 升级无防护网；"分支已建但 commit 失败"路径零覆盖
   - 零 TODO/FIXME 标记 + 静默吞异常，意味着很多"没人知道坏了"的路径连测试的动机都不会产生

---

## 六、修复优先级建议

### P0 — 立即修（真实 bug / 数据丢失 / 挂死）

1. skills `$ARGUMENTS` 正则替换损坏（skills/runtime.py:88，改成 `lambda m: args` 一行修复 + 回归测试）
2. COMMIT 失败悬挂事务 + busy_timeout 缺失（rollout.py:4820-4843）
3. 有效模型响应因租约边界静默丢弃（rollout.py:2313-2326）
4. execute_python 漏包 `_blocking_phase`（shell.py:801）+ git/gh 子进程无超时（git_delivery.py 全部）
5. llm_registry.py:457 吞掉 generator 构建根因（`raise ... from exc`）
6. `resolve_clarification`/`resolve_choice` 补取消检查（与 receive_turn_input 对齐）

### P1 — 单点收敛（消灭"改动一处另一处静默失效"）

1. **显式化 Turn 恢复状态机**：一张转换表 + 每个持久状态一个 `next_actions()`，合并 session.resume 9 分支与 turn.py 4 个 recover 入口；同时给 reducer 加转换校验。这一项能同时消灭最大的三类规则堆积
2. **错误码/错误消息分离**：PublicWebError 携带 status_code；apply_patch 改结构化错误对象；gh/git/异常 message 匹配全部改错误码或异常类型
3. **字符串前缀协议换成结构化标记**：`Runtime input limits:`/`Runtime budget status:`/` restore ` 锚点等，注入与解析共享常量或独立元数据通道
4. **政策 revision 与配置字典改 TypedDict/Literal 枚举**，禁止整字典 `==` 比较（可加 lint 规则）
5. **工具名/能力注册表**：`{"web_search","web_fetch"}`、`{"run_command","manage_command"}`、update_plan/search_knowledge 特判全部迁到 Tool 元数据声明（approval_profile/output 策略/投影器注册表已有雏形）
6. **保护目录/忽略目录/provider 集合单一事实来源**：常量模块或 catalog 字段，Seatbelt 正则由常量生成

### P2 — 结构性拆分（配合功能迭代逐步做）

1. rollout.py 按 18 职责物理拆分（schema/records/projections/lifecycle/interactions/approvals/model_ops/budget_ops/tool_ops/plan/context/compaction/artifacts/memory_store/evidence）；拆分是物理切分而非加门面（AGENTS.md 原则）
2. cli.py 拆出聊天 REPL（斜杠命令注册进已有的 `CHAT_COMMANDS`，消灭 20 连 if 链）、三套暂停处理器合一、model admin 移走
3. pydantic 审计黑名单改白名单文法声明，终结 11 连补丁模式
4. provider 层加重试/退避/超时策略（这是宣称"长任务恢复"的 runtime 最名不副实的缺口）
5. file_manifest.py 决策：接线或删除（连同 21 个测试）
6. `check_anti_patterns.py` 扩展到 `agent_runtime/`——把"防规则堆积"本身变成单一通用拦截点，而不是靠人肉记忆

### P3 — 偿还一致性债

- 三套路径 containment、三处停止原因归一化、双压缩计划器、v1/v2 completion、双事件协议、两种 usage 兜底构造——各选一处保留
- 11 处终端魔法尺寸收敛为断点表；剪贴板策略收敛为策略对象
- 静默吞异常补最小诊断事件（不必全改，按第四节 D8 清单逐个补 `logger.warning(..., exc_info=True)` 级别的可观测性）

---

## 附：审计覆盖对照

| 模块簇 | 状态 | 主要产出 |
|---|---|---|
| harness/rollout.py | ✅ 全文通读 | 18 职责分解、R1-R9/F1-F12/D1-D7 |
| harness 会话事件层（turn/session/events/reducer/protocol/completion/command_recovery/model_adapter/services/ownership/verification/checkpoints） | ✅ 12 文件通读 | 跨文件观察 8 条 |
| tools 子系统（shell/web 四件/executor/orchestrator/tool/selection/permissions/filesystem/data/evidence） | ✅ 17 文件通读 | 六大堆积模式原始证据 |
| CLI/终端层（cli/terminal_*/*_cli/streaming/result/text/primitive_ops） | ✅ 通读 | cli.py 20 职责分解 |
| 模型子系统（core/modeling/model_* 21 文件） | ✅ 通读 | 18 条硬编码特判全清单 |
| 上下文/记忆/预算/审批/技能（context/memory/planning/approval/knowledge/budget/skills 等） | ✅ 通读 | 三子系统耦合分析 |
| MCP/交付/基础设施（runtime/mcp*/connection_manager/file_manifest/workspace/git_delivery） | ✅ 通读 | 全局异常统计 + 测试盲区 |

每条发现的完整证据（代码引用、git 提交号、行号）见上文章内引用；本报告所有行号基于 2026-10-25 的工作区代码（feat/worktree-isolation@1993249，即 main@3ff1ec2 + worktree 隔离功能；行号已按该版本逐一复核）。
