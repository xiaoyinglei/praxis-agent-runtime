# 渠道、持久通知与 CI/PR 跟进

渠道只接收经过 provider 验证的内容，并把已授权输入交给已有的后台宿主、Session 和 TurnRunner。编码身份仍是 Thread/Turn。binding、grant、inbox、outbox、CI 订阅和 publication intent 是同一 rollout 中的 canonical facts，可用同一重放机制重建 `channel_entries` 投影；输入收据与 host command/wake 同事务提交。

## 本地配置与精确寻址

先用 `agent run --background` 建立工作流，记录 Thread 与数据库身份。新话题必须由本地操作者明确绑定现有 Thread；未绑定话题拒绝执行，链接、编号、显示名称和模型猜测都不能建立授权。不会向未授权新受众提供其他讨论的候选或结果。

GitHub source JSON 示例（凭证仅保存环境变量引用）：

```json
{"source_id":"github-personal","kind":"github","repository":"owner/repository","webhook_secret_env":"PRAXIS_GITHUB_WEBHOOK","outbound_token_env":"PRAXIS_GITHUB_TOKEN","outbound_principal":"123456"}
```

`outbound_principal` 是发信账号的稳定 numeric ID；GitHub 出站要求显式配置，用于精确回声抑制及丢失 ACK 核对，不能用显示名替代。飞书配置使用 `kind=feishu`、`app_id`、`chats`、`encrypt_key_env`、`verification_token_env` 和出站所需 `app_secret_env`。支持签名及严格 PKCS7/AES 解密，token 认证的 URL challenge 不执行任何工作。环境变量的值应由本机凭证设施注入，配置文件不保存 secret/token。

```sh
uv run agent channel configure THREAD source.json --checkpoint-db /absolute/runtime.sqlite --workspace /absolute/workspace
uv run agent channel bind github-personal github:owner/repository:17 THREAD github:owner/repository:17 --checkpoint-db /absolute/runtime.sqlite --workspace /absolute/workspace
uv run agent channel grant github-personal 123456 github:owner/repository:17 THREAD --role owner --operations input,status,respond,cancel,notify --checkpoint-db /absolute/runtime.sqlite --workspace /absolute/workspace
uv run agent channel status --checkpoint-db /absolute/runtime.sqlite --workspace /absolute/workspace
```

GitHub 讨论精确映射到 repo + issue/PR；review/check 事件保留原生身份及 SHA。飞书映射到 chat + root message，附件只作为 provider key 引用进入输入，不当作本地路径，也不自动下载或广播。显式 `local_identity` 可记录跨渠道身份关联；每个来源、principal、讨论、受众和操作仍需单独 grant。`role` 是明确的本地角色记录；有效操作以当前 grant 的 `operations` 为准。grant 的每次变更都递增版本。

`/status`、`/cancel` 必须是完整且未引用的消息体；`/respond REQUEST ACTION` 精确绑定当前 interaction（可追加换行后的回答），`/input TEXT` 或普通正文为输入。引用代码、日志中的命令不会成为控制操作。bot、review 和 CI 事件只成为证据；不能从正文获得审批或取消权。status ACK 只返回当前生命周期/交付摘要；有独立 notify 权限时也通过 outbox 回复原讨论。

```sh
uv run agent channel serve --checkpoint-db /absolute/runtime.sqlite --workspace /absolute/workspace
```

只开放 `POST /events/{source_id}`，默认监听本机。公网 TLS/转发由部署者配置；HTTP 没有 grant、bind 或管理入口。`receive SOURCE BODY_FILE HEADERS_JSON` 走同一认证和事务，便于诊断。Webhook 接收成功会启动同一个后台宿主。认证失败、撤销、异载荷重用事件 ID 均不会泄露结果。

GitHub 签名覆盖正文，**不覆盖 delivery header，也没有签名时间戳**。持久收据同时按 source/event ID 和签名正文中的 native event identity 去重；更换 delivery header 不会重新执行。相同身份不同规范载荷拒绝；JSON 格式变化不改变规范摘要。飞书还检查签名时间窗口。收据缓存命中仍重验当前 principal、role、workspace、受众和版本。

## 通知、撤销与未知结果

宿主在需要用户行动或 Turn 结束时收集精简的 canonical 状态/交付摘要；默认不发送答案全文、原始日志或文件。相同受众、Turn 和摘要只排队一次，普通状态不变保持安静。审批摘要列出请求身份、可选动作、工具、资源与 effects。完成表示 Turn 结束，E 的 verified/unverified/blocked 独立保留。

outbox 每次领取和紧邻最终发送时复核当前来源、原主体、grant 版本和精确受众；取飞书 token 后仍须再次核对。凭证/config 更新会使旧配置授权失效，需要本地明确重新授权。权限已经撤销时，队列里保存的文字也不能发送。canonical 摘要在领取和紧邻发送时还复核 Thread head、Turn、E 新鲜度、interaction 和 CI 状态；旧事实不继续广播，尚未发送的旧状态提醒会合并。通过最终检查后已在途的请求无法保证撤回。

确定未发出的临时故障使用持久指数退避/抖动和 Retry-After，受总 attempt/deadline 限制。connect/read/write/pool timeout 之外，还有包含 token 获取与响应读取的整体绝对期限。发送中丢失 ACK 或发送进程被杀后进入 uncertain；只有正面的 provider correlation + 发信身份核对才能确认已接收，查询未找到不能证明未发送。飞书目前没有可用的持久 UUID 查询合同，未知发送保留 uncertain，不承诺重投。

```sh
uv run agent channel outbox ID reconcile --checkpoint-db DB --workspace WORKSPACE
uv run agent channel outbox ID retry --checkpoint-db DB --workspace WORKSPACE
uv run agent channel outbox ID reauthorize --grant-id CURRENT_GRANT --checkpoint-db DB --workspace WORKSPACE
uv run agent channel outbox ID abandon --checkpoint-db DB --workspace WORKSPACE
```

`retry/reauthorize` 只对确定未发送的 pending/blocked 记录有效，保留原 audience、attempt 和 deadline，不重置授权额度。uncertain 只能核对或放弃；新 grant 也不会自动重复相同未知通知。`status` 提供来源、凭证引用健康、开关、版本、排队和诊断事实；凭证缺失不无限重试。

## CI 订阅和有限续跑

`agent channel ci-subscribe CONFIG.json` 与 `Channels.subscribe_ci(...)` 同义。配置包含 `subscription_id/thread_id/source_id/principal/conversation_id/repository/pull_request/head_sha/check_ids`，以及有界 `poll_policy`（`max_attempts/deadline/initial_delay/max_delay`）。check 身份为精确 `APP_ID:CHECK_NAME`；同时核对 run ID 与 head SHA，不能只按名称信任其他 app。外部 CI sender 还需独立的 evidence grant。

可选 `repair_allowance` 包含 `max_new_turns/deadline/max_steps/max_tokens_total/max_cost_micros` 总额，`repair_budget` 包含每轮三个预算上限；两者须同时设置。创建订阅要求当前 input 授权，不能由 CI 事件自身授权。额度先原子预留再排队已有宿主 start command，保守预留不退款，重启、重投、换 head 都不重置。原执行配置和权限上限继续生效，排队及紧邻模型/工具执行都重验授权。范围、权限或期限不再允许时，保留证据等待用户处理。

失败、review 和查询结果先写结构化证据。无 checks、pending、skip、查询失败均不能判 pass。完整非空订阅 check 集显式成功才 pass。各 review 决定按原生身份保留；review comment 与其他 reviewer 的评论不能清除尚未解决的 changes_requested。新 head 清空旧检查结果并保留未解决的 review 阻塞；有连续 before→after 身份的签名 synchronize 事件可推进 head。丢失中间事件或轮询观察到不同 head 时显示 `head_mismatch`/待明确选择，不能用可能迟到的旧快照回滚 head。

```sh
uv run agent channel ci-status SUBSCRIPTION --checkpoint-db DB --workspace WORKSPACE
uv run agent channel ci-head SUBSCRIPTION EXACT_SHA --checkpoint-db DB --workspace WORKSPACE
uv run agent channel ci-disable SUBSCRIPTION --checkpoint-db DB --workspace WORKSPACE
```

本地 E 验证、远端检查与完成分开呈现。`delivery_complete` 还要求本地 verified 记录的最终 source HEAD 等于订阅 SHA。改 head 不重置轮询或续跑额度；耗尽后必须由用户选择新的显式授权。

真实渠道验收使用已授权的专用话题、稳定的 HTTPS 回调和独立样例工作区。沿用上面的 `channel configure/bind/grant/serve`，再从 provider 发送测试任务。用 `agent background result` 确认受审回答、交付记录和验证树，再核对远端通知及重投去重。若还要验收 CI，先在样例中选定待交付提交，在该 HEAD 上重跑本地检查和提交审阅，再用 `channel ci-subscribe` 和 `channel ci-status` 核对同一 SHA 的完整检查集。测试撤权前记录原始投递和收据，撤权后重投同一原生事件，确认宿主和 outbox 均未产生新副作用。

## 独立远端副作用

push、open_pr、review_reply 各需独立操作 grant、intent、精确 scope、GitDelivery preview ID、远端关联键和结果。先 `publication-preview`，把显示的预览 ID 及精确资源/正文写入配置，再 `publication-prepare CONFIG.json`；准备只持久化可审查 intent。`publication-execute OPERATION_ID` 执行该单项动作，`--reconcile` 只核对远端。SDK 同名方法同义。

push 只推预览中的 commit 到预览中的 branch；建 PR 不隐式 push；review 回复绑定具体 PR/comment 与当前 commit，旧位置要求重新定位。GitDelivery 的原有 index/选择/提交边界不变。超时会先查远端精确 branch SHA、PR/correlation 或 reply/correlation，不能盲重做；持久 sending 在重启后也不能直接领取第二次。结果未知和被撤销的旧 intent 可在本地诊断，但不能绕过当前授权继续发布。

## Python 与验证边界

`agent.channels` 返回同一数据库/工作区上的 Channels；直接嵌入 `Channels(database=..., workspace=..., autostart=False)` 可由调用方运行 BackgroundHost。configure/bind/grant/receive/inspect/collect/deliver、outbox 操作、CI 与 publication 方法共享上述实现。查询不构造模型。

本批验证使用脚本模型、本机进程、MockTransport 和 loopback 模拟 provider，覆盖撤销、重复输入、未知 ACK、慢流期限、进程被杀及旧 head。没有发送真实 GitHub/飞书消息、push、创建 PR 或回复 review；不能当成真实账号连通性、云模型编码或 G 发布验收。当前支持 GitHub/飞书上述原生事件与操作，不宣称任意 provider 或完整飞书资源下载。持久宿主不是开机系统服务；跨关机自动启动仍需部署管理。
