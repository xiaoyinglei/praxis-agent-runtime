# 主流模型元数据快照

核对截止日期：**2026-10-22**。维护源为 [model-catalog.yaml](model-catalog.yaml)。本版覆盖 9 个服务商、18 个代表型号，聚焦文本、推理和工具调用；不是全部在售模型清单，也不是模型质量排行榜。

## 使用与维护

修改维护源，然后运行：

```bash
uv run python scripts/sync_model_catalog.py
uv run python scripts/sync_model_catalog.py --check
```

生成脚本将已核验接口声明的条目同步到公开目录 `configs/agent-models.yaml`，并更新 `configs/models.yaml` 中对应的云模型条目。移除或降为仅元数据的受管型号会从两份运行配置撤下；保留研究目录中的本地、嵌入和其他旧定义。后者不自动成为本次核验范围。检查模式不写文件，CI 检查两份视图是否过期。wheel 同时包含公开配置和带来源的快照。

公开 CLI 和 Python SDK 继续使用现有 ModelRegistry / ModelControlPlane，没有新增运行时解析器。默认模型仍为 `deepseek-flash`。普通用户只需配置所选模型的 Key，再切换模型；内置条目的容量无需重复手填。

```bash
uv run agent model list --source
uv run agent setup --model kimi-k3 --no-mcp
uv run agent chat --model kimi-k3
```

安装包使用模型登记能力，不联网自动更新目录。需要后续更新时重新核验官方资料、修改维护源、再生成与测试。即使元数据快照不变，服务商也可能更新模型别名背后的权重。

## 覆盖范围

`chat_completions` 表示声明与现有协议相容、可生成运行配置，**不表示账号授权、联网探针或业务任务验收已经通过**。本版没有为新增型号调用付费推理；仅只读查询 Kimi Models API，确认其准确上下文整数。所有其他数值依据每项 `sources` 中的官方页面。

| 服务商 | 型号 | 本版状态 | 容量与限制 |
| --- | --- | --- | --- |
| DeepSeek | `deepseek-flash`、`deepseek-v4-pro` | 进入运行目录 | 上下文 1,048,576；单次输出硬上限 393,216 |
| Kimi | `kimi-k3` | 进入运行目录 | 上下文与输出硬上限均为 1,048,576；单次请求还需为输入留空间 |
| 通义 Qwen | `qwen3.8-max`、`qwen3.8-flash` | 进入运行目录 | 官方上下文 1,000,000；运行窗口保守设为 983,616，确保不超过独立输入上限；combined 输出 cap 131,072 |
| MiniMax | `MiniMax-M3.1-Flash` | 进入运行目录 | 上下文 1,000,000；输出硬上限 524,288，包含思考 |
| Groq | `openai/gpt-oss-120b` | 进入运行目录 | 上下文 131,072；输出硬上限 65,536 |
| Groq | `llama-3.3-70b-versatile` | 进入运行目录 | 上下文 131,072；输出硬上限 32,768；修正旧配置混淆的两项数值 |
| OpenAI | `gpt-6.1-sol`、`gpt-6-astra`、`gpt-6-luna` | 仅元数据 | API 上下文 1,050,000、输出 128,000；思考下的工具协议需 Responses 适配 |
| Anthropic | `claude-opus-5-5`、`claude-sonnet-5-5` | 仅元数据 | 官方写上下文 1M、普通输出 128,000；精确上下文整数暂留空；当前采样参数与思考协议不兼容 |
| Google | `gemini-4-argon`、`gemini-3.8-flash` | 仅元数据 | Argon 为共享 1,048,576；Flash 是独立输入 1,048,576 / 输出 65,536；必须保留工具思考签名 |
| 智谱 | `glm-5.3`、`glm-5.3-flash` | 仅元数据 | 官方上下文 1M，精确整数暂留空；输出 131,072；`max_tokens` 与现有 `max_completion_tokens` 的兼容性待验证 |
| MiniMax | `MiniMax-M3` | 仅元数据 | 容量已记录；默认思考内容的拆分需要当前配置未支持的 `reasoning_split` |

## 参数边界

- 模型硬上限、单次输出额度和累计消费预算分开。多数新云条目的产品输出额度为 65,536，Groq 为 8,192；这些是本项目选择，不冒充官方默认值。厂商原生默认与特殊范围另记在快照说明中。
- 推理档位仅声明已确认的取值。K3 始终思考，不继承 K2 的 thinking 开关；Qwen 只列 canonical 档位，不把服务商映射别名当成额外能力。未映射的原生控制不对外承诺。
- 1,000,000、1,048,576 和 1,050,000 保持原值。只写 `1M` 的资料不自动补成某个整数；独立输入、回答和思考预算也不相加冒充统一上下文窗口。
- Qwen 的官方完整窗口记在 `limits`；当前运行协议没有独立模型输入上限字段，因此 `config.context_window_tokens` 明确采用保守值 983,616。调低输出额度仍不能突破其思考模式输入上限；这会少用一小部分上下文，后续支持独立输入约束时再恢复完整窗口。
- 严格 JSON Schema 能力与 JSON Object Mode 区分。MiniMax M3.1 和 Llama 本版没有声明严格 Schema；Qwen 两款的 Schema 能力有官方名单依据。
- 保留现有工作历史压缩策略和消费上限；切换模型按该型号容量约束输入和输出，不会提高已设的累计消费额度。
- 本版没有启用图片传输或增加 Responses / Messages 适配器；厂商支持视觉不等于当前 Agent 已支持图片消息。
- 个人注册表重名仍按现有规则报错，不覆盖用户定义。升级前应保留个人文件并解决与新内置 ID 的冲突。

每条记录附官方来源与限制；不收录未经核实的第三方转售端点。价格可能有地区、缓存、时段和阶梯差异，本版不把未经完整建模的价格写入消费预算计算。
