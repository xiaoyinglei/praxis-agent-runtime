# README Code Agent 刷新设计

## 目标

以 `codex/code-agent-v1-benchmark` 的当前源码为唯一事实来源，刷新根目录
`README.md`，让安装、公开入口、模型选择、工具面、运行时安全和产品边界与代码一致。

## 内容边界

- 不声明默认模型；模型示例统一表述为可通过 `--model`、`agent model` 或 SDK 参数选择。
- 不写易过期的测试数量、单次 benchmark 结果或“已生产就绪”宣传。
- 删除已淘汰的 persistent memory、旧 API、旧工具数量和旧 provider 假设。
- 保留 RAG 作为可选证据子系统，突出 Python-first Code Agent 的公开 CLI/SDK 主路径。
- 用源码可验证的事实说明 goal contract、advisory plan、上下文压缩、工具调用恢复、
  workspace 写入和修改后验证边界。

## 验收

- README 中的命令能够从当前 CLI 帮助和公开 SDK 签名得到验证。
- 工具名称、模型别名和安全语义与当前源码一致。
- 不再出现“默认模型是 DeepSeek”或已删除 persistent memory 的现状描述。
- Markdown 链接、代码块和目录锚点保持有效，`git diff --check` 通过。
