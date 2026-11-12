# Praxis 执行链审计与修复交接

## 当前工作区

- 项目及新聊天起始目录为 `/Users/leixiaoying/LLM/RAG学习`。分支 `main`，HEAD `313818817bfe1710abd54ce65877d3d1d72eef8b`。
- 当前有上一批未提交的修改和新增测试 `tests/agent/harness/test_delivery_completion.py`。它们已处理终端补充输入、主动压缩容量和重试、普通只读收尾及状态展示。保留这些修改和已有 `outputs/`。先核对 Git 状态，在隔离检出中带上相关基线进行修改和验证，交付时检查差异与重叠。
- 原始记录在 `.praxis/checkpoints.sqlite`，仅只读检查。首轮 `turn_eb9f9b2a7fcb46e08d922cf8a6bb0535`，追问 `turn_d001aee2af184aa7b4497e5e4cbd0e3f`。上一批实际模型验收使用独立数据库 `/private/tmp/praxis-tty-probe.DkDhgh/verified-streaming/trace.sqlite` 和 `/private/tmp/praxis-tty-probe.DkDhgh/verified-long/trace.sqlite`。用户原始长任务未重跑。

## 已核实的问题

1. **变更证据误判和重复全树扫描，优先处理。** 在公开 Python SDK 走真实补丁工具和完成门，使用脚本模型控制步骤。两次都把 `value.txt` 从 `before` 改为 `after`，工具均记录一个目标文件变更。普通工作区结果为 `done/completed`。加入一个无关的不可读文件后，前后整树哈希都返回 `None`，工具把它折叠成 `workspace_tree_changed=False`。启用 `require_workspace_change` 后最终耗尽步数。相关链条是工具范围与权限 → `tools/executor.py` 的写前观察、执行、写后观察和后台句柄观察 → 工具结果持久化 → `harness/stop.py` 的验收判定 → 交付证据。原始调用的工具结果序号 457 记录了 46,374 个文件，单次整树哈希约 3.75–9.57 秒。旧报告给出的 20 次写入耗时是外推。
2. **子进程 I/O 边界。** `tools/builtins/shell.py` 的前台 Shell 和托管 Python 启动未显式设置 stdin。后台 worker 使用 `DEVNULL`。`manage_command` 支持 status、wait、cancel、read。核对前后台的 stdin 所有权、取消、权限和会话恢复，隔离非交互进程的输入。根据对应产品源码确定交互式命令能力的最小可用边界，用无敏感内容的独立终端验证。历史轨迹尚未证明 stdin 继承是用户那次无法输入的根因。
3. **实际发给模型的工具定义、历史投影和容量恢复。** 原始首轮冻结 OpenAI 请求的 18 个工具，序列化 JSON 约 39 KB，`run_command` 单项约 5.7 KB。分开核对工具元数据、工具可见性、输出预算、历史记录与模型投影。当前同一容量恢复链的第二次服务端溢出会结束。Praxis 已有根据实际请求测量的工具输出裁剪路径。区分上一批已改的主动压缩失败重试和这里的服务端溢出恢复。依据实际 wire 和回合证据决定哪里需要改，避免按字数、模型名或固定比例拍脑袋。
4. **OpenAI 兼容响应归一化。** `modeling/openai_wire.py` 的完整响应解析对非字符串内容执行 `str()`。直接调用解析器，文本分片数组确实输出 Python repr。追到 Gateway 的完整响应、流式和非流式回退，核对各自的正式支持契约，再修复文本分片和未知形态的处理。现有证据只覆盖解析器，实际供应商覆盖待验。
5. **权限配置边界。** `trusted-local` 确实授予根目录读权限，显式拒绝和管理员上限可进一步收紧。原始两轮冻结配置都为 `workspace`，没有证据表明这次发生了凭证泄露。只用合成路径和独立配置验证产品承诺与实际权限是否一致。

## 对标和交付要求

- Codex 源码切片在项目 `artifacts/agent-reliability-fix-20261020/codex-reference`，版本 `a4c9194247c6a1e79aa3e72cc6cd77238f2ed354`。Claude 本地历史镜像在 `/Users/leixiaoying/PycharmProjects/brath-claude-code/src`，来源未独立核验。资料缺段时找可核验的实现，并明确证据边界。
- 按问题追三套实际调用链，从接收、执行、记录到消费和恢复。说明每一层信任谁提供的事实、在哪一步丢失信息、哪些边界应保留或替换。许可允许且接口合适的成熟实现可直接复用。现有结构可以调整。
- 首先完成变更检测链的修复。区分已变、未变、无法判定，优先考虑可核验的执行与变更事实。对精确文件写入和任意命令的观测责任分别对标，保住验收、审批、恢复和外部并发边界。工具或模型自行声明的路径不能单独作为完成证据。对前后权限检查隔着耗时观察的情况保留必要的再确认。
- 按证据继续修复其余已证实缺陷。工具发现、重复失败保护、后台恢复和抽象层去留，依据正常反例和实际调用决定。原报告中「Codex 没有工具发现」「已有机制可以直接删掉」「沙箱整体更强」等判断缺少支持。
- 实现遵守项目 AGENTS.md，保持设计简洁、规划可见、ACI 可验证。禁止根据任务关键词、站点、模型习惯、路径、benchmark ID 或工具顺序加特例。使用受影响的正常反例和真实公开入口验证。真实终端与真实模型结果、实际工具与脚本模型结果、回归测试分别标明。报告代码位置、保留和替换的机制、运行记录和剩余缺口。
