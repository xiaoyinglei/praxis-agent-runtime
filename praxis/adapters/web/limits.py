"""公共 web 逐请求硬上限与宿主配置加载(Q7 裁决;legacy cli._load_web_limits 语义)。

**有意差异(显式)**:这里**不做 turn/会话级请求计数账本**——旧
``reserve_resource_unit`` SQLite 原子预扣与 ``web_request_budget_exceeded``
归阶段 4 预算树(设计 §5.5 的 steps/cost 粒度恰好覆盖它),跨进程原子预扣归
阶段 5。3B 只保留逐请求硬上限(字节/时间/跳数;候选数上限在 client.py)。
``WebLimits`` 字段集合由测试钉死,不得悄悄长出 turn_requests 之类字段。

配置来源(优先级):``PRAXIS_WEB_LIMITS`` env(**值是 JSON 文件路径**,不是
内联 JSON——旧语义保留,文件 ≤4096 字节)> 显式 path 参数 >
``host_config_dir()/web.json`` 的可选 ``"limits"`` 块。宿主文件缺失或块缺失
⇒ 默认值; malformed ⇒ ValueError 点名文件。**workspace 文件绝不读**。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, fields
from pathlib import Path

from praxis.protocol.floor import host_config_dir

__all__ = ["WebLimits", "load_web_limits", "web_configuration_path"]

# legacy cli._load_web_limits:read(4097) 判 >4096。
_LIMITS_FILE_MAX_BYTES = 4096
# legacy web_config.resolve_web_configuration:web.json 整体 16 KiB 上限。
_WEB_JSON_MAX_BYTES = 16_384


def web_configuration_path() -> Path:
    """宿主 ``web.json`` 的唯一路径构造点(Task 13/T20 规则⑳)。

    ``load_web_limits`` 与 ``search.load_web_search_config`` 共用;任何其他
    模块(含 CLI/搜索)不得自行拼 ``host_config_dir()/web.json``。
    """
    return host_config_dir() / "web.json"


@dataclass(frozen=True, slots=True)
class WebLimits:
    """逐请求硬上限;模型参数不能抬高这些上限(T12 schema 钳制在更小区间)。

    范围:max_bytes 4096..32_000_000;timeout_seconds (0, 300];max_redirects 0..10。
    类型严格(JSON 浮点 max_bytes 之类直接拒),与旧 pydantic strict 同向。
    """

    max_bytes: int = 2_000_000
    timeout_seconds: float = 20.0
    max_redirects: int = 5

    def __post_init__(self) -> None:
        if type(self.max_bytes) is not int or not 4096 <= self.max_bytes <= 32_000_000:
            raise ValueError("web limits max_bytes must be an int in 4096..32000000")
        if type(self.timeout_seconds) not in (int, float) or not 0 < self.timeout_seconds <= 300:
            raise ValueError("web limits timeout_seconds must be a number in (0, 300]")
        if type(self.max_redirects) is not int or not 0 <= self.max_redirects <= 10:
            raise ValueError("web limits max_redirects must be an int in 0..10")


def _read_bounded_json(path: Path, max_bytes: int) -> object:
    """有界读取 + JSON 解析;超限/坏 JSON 抛 ValueError 点名文件。缺失抛 FileNotFoundError。"""
    with path.open("rb") as handle:
        raw = handle.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError(f"web limits file exceeds {max_bytes} bytes: {path}")
    try:
        return json.loads(raw)
    except (ValueError, UnicodeError):
        raise ValueError(f"web limits file is not valid JSON: {path}") from None


def _limits_from_mapping(data: object, source: Path) -> WebLimits:
    if not isinstance(data, dict):
        raise ValueError(f"web limits must be a JSON object: {source}")
    unknown = set(data) - {field.name for field in fields(WebLimits)}
    if unknown:
        raise ValueError(f"web limits has unknown keys {sorted(unknown)}: {source}")
    try:
        return WebLimits(**data)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid web limits in {source}: {error}") from None


def load_web_limits(path: Path | None = None) -> WebLimits:
    """加载逐请求上限:env PRAXIS_WEB_LIMITS(文件路径)> path > host_config_dir()/web.json。

    env 指向的文件缺失/超限/坏形状 ⇒ ValueError 点名文件(显式配置必须可用);
    宿主 web.json 缺失或无 "limits" 块 ⇒ 默认值。
    """
    env = os.environ.get("PRAXIS_WEB_LIMITS")
    if env:
        env_path = Path(env)
        try:
            data = _read_bounded_json(env_path, _LIMITS_FILE_MAX_BYTES)
        except FileNotFoundError:
            raise ValueError(f"web limits file not found: {env_path}") from None
        return _limits_from_mapping(data, env_path)
    source = path if path is not None else web_configuration_path()
    try:
        data = _read_bounded_json(source, _WEB_JSON_MAX_BYTES)
    except FileNotFoundError:
        return WebLimits()
    if not isinstance(data, dict):
        raise ValueError(f"web configuration must be a JSON object: {source}")
    if "limits" not in data:
        return WebLimits()
    return _limits_from_mapping(data["limits"], source)
