"""协议层 JSON 基础:JsonValue 类型与 freeze/thaw 规范化(移植自旧 tool.py)。

freeze_json 把任意输入规范化为不可变 JSON 值(Mapping→mappingproxy、list/tuple→tuple),
非有限 float 与非 JSON 兼容值直接拒绝;thaw_json 还原为可变的 dict/list,
供序列化(canonical digest)与 jsonschema 校验使用。
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from types import MappingProxyType

type JsonValue = (
    str
    | int
    | float
    | bool
    | None
    | tuple[JsonValue, ...]
    | Mapping[str, JsonValue]
)

type _MutableJsonValue = (
    str
    | int
    | float
    | bool
    | None
    | list[_MutableJsonValue]
    | dict[str, _MutableJsonValue]
)


def freeze_json(value: object, *, path: str) -> JsonValue:
    """规范化为不可变 JSON 值;path 用于错误定位(如 ``$``、``arguments.items[0]``)。"""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must contain only finite JSON numbers")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, JsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            frozen[key] = freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise TypeError(f"{path} must contain only JSON-compatible values")


def thaw_json(value: JsonValue) -> _MutableJsonValue:
    """还原为可变 dict/list(反序列化/校验前的中间表示)。"""
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def canonical_json(value: object) -> str:
    """稳定 JSON 文本:键序无关、紧凑分隔符、拒绝非 JSON/非有限值。

    digest 与持久化的唯一规范口径(store.memory 的 sha256 包络消费)。先经
    ``freeze_json`` 规范化并拒绝不合规值(Mapping→mappingproxy、序列→tuple、
    非有限 float→ValueError),再 thaw 回可变结构交给 ``json.dumps``。
    """
    frozen = freeze_json(value, path="$")
    return json.dumps(
        thaw_json(frozen),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
