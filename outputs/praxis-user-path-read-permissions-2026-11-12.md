# 用户输入路径的只读授权修复

修复依据是原始 Turn `turn_05e5e47ddba3485080e0fe64af5aa847`：用户正文提供 Claude src 路径，Shell 返回 EPERM（Item sequence 10），文件工具返回 filesystem_denied（18），最终答复要求复制目录（530）。原数据库仅只读查询，没有改写历史授权或重新运行付费任务。

## 新行为

用户直接发送的正文里，存在的绝对路径、`~/` 路径或本机 `file://` 引用，视为本次会话的只读选择。文件授予该文件读取，目录授予目录内读取；不复制内容，不增加写权限，不要求每次读取审批。含空格的混合正文路径使用引号、反引号或 shell 风格转义；独占一行的路径可以直接含空格。原始文本保留。

只从真实用户输入入口识别：CLI 与 SDK 的 Session.submit，以及 Session/Agent.steer。模型回复、工具输出和模型产生的子任务文本不产生授权。不存在的路径不会预先授权。旧 Rollout 中没有这类授权记录的历史正文不作追溯扩权。

授权随用户消息原子提交，保存输入路径、当时解析目标和文件/目录类型；重开从原始用户记录恢复。新输入生效于下一次冻结的模型/工具请求，不扩大已经运行中的工具权限。独立 SDK 进程发送的补充输入也走同一链路。

明确 deny、管理员和调用方当前上限继续生效。替换符号链接目标、把获准文件替换成目录，不会自动获得新目标或子树授权。重复输入回执不重新解析路径；失败的输入不留下授权。

## 代码路径

- `agent_runtime/local_files.py::user_path_references`：识别用户输入中的本地路径并冻结引用身份，不读取文件内容。
- `agent_runtime/harness/turn_ops.py`、`execution.py`：授权引用与用户输入同事务提交；模型子任务默认不识别路径授权。
- `agent_runtime/harness/rollout.py`、`runtime_store.py::user_read_references`：只读取指定可见历史中 producer=user 的原始引用字段，不展开工具历史；遵守分支截止点。
- `agent_runtime/permission_policy.py::with_read_references`、`agent.py`：恢复只读选择，在施加当前管理员上限之前组合授权；冻结的旧请求限制和当前来源限制仍各自执行。
- `agent_runtime/harness/session.py`、`services.py`：下一 Step 刷新已提交用户引用，使文件工具与 Shell 使用同一有效权限；支持独立输入入口与重开。
- `tests/agent/test_user_path_permissions.py`：解析、真实读取、跨进程 CLI 恢复、无审批、无复制、拒绝覆盖、重定向和回执测试。

## 本地验证

- 权限、附件、SDK、输入回执、steering、CLI 相关定向集：124 passed（排除单独运行的真实 macOS 沙箱用例）。
- 补充的新拒绝覆盖/输入回执用例：3 passed。
- 公开 CLI + 真实 macOS Shell + 两个进程、同一 Turn 恢复：1 passed。初次读取外部目录内文件，退出码 75；recover 后文件工具再次读取并完成，无权限审批记录，原文件不变且工作区没有副本。该用例在外层沙箱之外运行，以允许创建 Praxis 自己的 macOS 沙箱。
- 长任务检查点和 CLI 恢复专项：7 passed，结果记录在 `/private/tmp/praxis-user-path-continuity.log`。上述合计 135 个不同用例通过。
- Ruff 和 `git diff --check` 通过。未调用付费模型，未跑全量 Agent/Provider 回归。

复跑当前新增测试：

```sh
cd '/Users/leixiaoying/LLM/RAG学习'
.venv/bin/python -m pytest -q tests/agent/test_user_path_permissions.py
```

重新启动 Praxis 后，直接输入原来的外部源码路径即可；无需先配置 permissions 文件。修复不会热替换已经运行的旧 Python 进程，也不会为已结束的旧 Turn 补造授权。
