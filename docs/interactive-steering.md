# 持续对话与执行中补充输入

本轮直接在保存项目的本地目录实现，保留已有未提交修改，不提交、不推送、不运行付费模型。

## 参考与取舍

对照 OpenAI Codex 的 `core/src/session/turn.rs`（2026-10-18 读取 main）：同一 Turn 保留 pending input，在下一请求前排入历史；采样结束还要检查 pending input，才能决定结束。协议的 expectedTurnId 防止输入误投另一 Turn。

- https://github.com/openai/codex/blob/main/codex-rs/core/src/session/turn.rs
- https://github.com/openai/codex/blob/main/codex-rs/app-server-protocol/src/protocol/v2/turn.rs
- https://developers.openai.com/blog/mastering-codex-remote-for-engineering
- https://code.claude.com/docs/en/interactive-mode#queue-messages-while-claude-works

Praxis 复用 Session / TurnExecutor / ToolOrchestrator / Rollout，采用相同的单执行循环、安全边界消费和 Turn 身份校验。本地先落盘再确认接收。普通补充影响当前 Turn；停止是显式取消，替换任务通过停止后提交新 Turn 表达，不猜关键词。独立的后续任务队列不属于本轮接口。

## 契约

- `await session.steer(text, expected_turn_id=...)`：只向指定的活动 Turn 提交补充，不授予权限、不启动第二个执行循环。返回持久化 user_message 凭据；`session.input_status(id)` 从模型请求账本推导 pending/applied/终态未消费。
- 输入到达时立即写入 user_message；每个 Step 捕获输入版本并在 request_ref 保存。请求准备期间到达输入，重新捕获；采样期间到达输入，保留真实计费和响应账本，但不执行过时动作/接受旧最终答复。
- 工具真正启动前再次检查输入版本；已启动工具正常收集结果，未启动工具写明确未执行结果。模型上下文把工具中途到达的补充放在完整工具组之后，原始账本顺序不变。
- 审批暂停中的补充先保存，后续恢复不得执行旧审批动作；未知副作用仍走原 reconciliation，不能靠补充跳过。
- 最终完成事务校验输入版本，避免在事件回调/并发连接中接收消息后按旧目标完成。上下文压缩保留全部当前 Turn 的补充原文。
- CLI 忙碌时 Enter 提交补充；反馈接收/待生效/已进入请求。拒绝时保留输入内容；Ctrl+C 保持停止语义。SDK 同步提供上述契约。

## 实施与验证计划

- [x] 先添加运行中采样补充、快速多输入、连续对话的失败测试，随后实现 Rollout 接收及版本凭据、Session 公共接口。
- [x] 先添加工具间隙、审批、最终提交和重开恢复的失败测试，随后贯穿 Step、工具启动、完成事务和 context 投影。
- [x] 先添加真实 prompt_toolkit 输入测试，随后接 CLI operation、反馈和错误保留。
- [x] 运行针对性测试、Agent/CLI 主回归、Ruff、mypy、import-linter、diff 检查；审查新修改与已有命令恢复修改的交互。

边界：steer 不宣称回滚已发生副作用，不把模型已接收等同于正确理解。不追加预算，不扩大原完成门禁，不自动重试 unknown provider/tool outcome。长工具中输入先待生效，到工具安全结束后生效；需要立即停止由 Ctrl+C / abort 执行。

## Python SDK 示例

```python
import asyncio

from agent_runtime import Agent
from agent_runtime.streaming.events import EventType


async def main():
    agent = Agent(workspace_path=".")
    turn_id = None
    receipt = None
    async for event in agent.stream("解释项目入口", require_workspace_change=False):
        if event.type is EventType.TURN_STARTED:
            turn_id = event.turn_id
            # 实际应用也可在后续输入事件中调用 steer。
            try:
                receipt = await agent.steer("还要说明异常处理路径", expected_turn_id=turn_id)
            except RuntimeError as error:
                # 接收与完成竞争时明确拒绝，调用者仍持有未发送的原文。
                print("补充未接收：", error)
    if receipt is not None:
        print(receipt.item_id, agent.input_status(receipt.item_id))
    if turn_id is not None:
        print((await agent.read_result(turn_id)).answer)


asyncio.run(main())
```

`applied` 按持久化的普通模型 dispatch 记录判定，不按 prepared 请求或摘要请求判定。它表示请求已进入执行，不证明远端模型已理解，或传输/采样最终成功。`unconsumed` 的原文仍保存在 Rollout，调用者可在明确的新 Turn 中重新提交。

每个持久化模型请求（包括摘要请求）必须显式包含正整数 `input_revision`；布尔值、缺失值和其他类型均拒绝。旧格式请求不再根据历史 Item 推算版本，恢复入口在模型准备、owner 接管和执行前明确报错，提示新建 Turn。可安全取消的旧 Turn 仍可 abort，已有数据库不会被删除或静默改写。明确未发送的过时 prepared 请求可以恢复到下一步；unknown 请求/工具仍走原有恢复与 reconciliation 契约。

## 验证记录

2026-10-18 本地验证：

- Agent / CLI 主回归：2892 passed、76 skipped、17 warnings。
- 随后压缩提交竞态与历史回放修改的相关集合：806 passed；最新补充输入核心集合：19 passed，真实终端输入集合：4 passed。
- Ruff、mypy（123 sources）、import-linter（5 contracts）、`git diff --check` 均通过。
- 独立只读审查发现的重复失败误计数、prepared 被替换后的崩溃窗口、压缩提交时输入竞态，均有失败后通过的回归测试。

测试使用脚本模型与本地真实终端事件，验证接收、执行边界和恢复契约；未运行付费模型，不据此宣称真实模型在大型任务上的成功率。冻结评测未改动，全部工作保留为未提交修改。

## 仍需独立处理的命令会话设计债

当前 `command_sessions.py` 保留 32 个内存结果并限制同一 Session 的活跃命令；本轮未改变这些并发/资源契约。私有 job 日志、孤立准备记录和固化代码 bundle 的自动 GC 尚未实现。安全清理必须同时考虑 owner/run 锁、恢复引用和账本保留，不能用直接删除目录替代。全工作区候选复制、权限变化时的保守取消也维持现有契约；本轮不将对话修复宣称为这些问题已解决。
