"""token 估算的唯一入口(B24/D2):只测 build_request 定型的出站文本。

tiktoken 可用时返回 ``bpe``;加载失败/离线/未知模型一律退回
``len(text) // 4 + 1`` 并如实标注 ``heuristic``——估算来源必须可审计,
不得把启发式冒充成 BPE。编码器实例按 tokenizer 模型 LRU 缓存(最多 8 个),
``threading.Lock`` 保护;加载在锁外进行,避免网络 I/O 阻塞其他线程。加载失败
的 key 进同样有界的负缓存:离线时后续测量直接走启发式,不会每次重试网络。

``measure_request`` 不重算消息/工具,只测 ``WireRequest.serialized_json``:
出站请求与测量文本是同一份字节,不存在第二口径。
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Literal

import tiktoken

from praxis.protocol.models import WireRequest

_DEFAULT_ENCODER_NAME = "o200k_base"
_HEURISTIC_CHARS_PER_TOKEN = 4
_MAX_ENCODER_CACHE = 8

# tokenizer_model / 默认编码名 → tiktoken 编码器实例(LRU,末尾最新)。
_ENCODER_CACHE: OrderedDict[str, Any] = OrderedDict()
# 负缓存:已知加载失败的 key。离线时避免每次测量都重试一次网络下载;
# 与正缓存同样有界(共用 _MAX_ENCODER_CACHE 的 LRU 上限)。
_UNAVAILABLE_ENCODERS: OrderedDict[str, None] = OrderedDict()
_ENCODER_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class TokenMeasurement:
    """一次文本测量的结果;``source`` 让调用方区分真实 BPE 与启发式。"""

    count: int
    source: Literal["bpe", "heuristic"]


def _load_encoder(tokenizer_model: str | None) -> Any:
    """加载编码器(不缓存);任何失败由 ``count_text_tokens`` 统一兜底。"""
    if tokenizer_model is None:
        return tiktoken.get_encoding(_DEFAULT_ENCODER_NAME)
    try:
        return tiktoken.encoding_for_model(tokenizer_model)
    except KeyError:
        # 有些配置直接给 encoding 名(如 o200k_base),不是模型名。
        return tiktoken.get_encoding(tokenizer_model)


def _get_encoder(tokenizer_model: str | None) -> Any:
    """取缓存编码器;未命中时加载,失败进有界负缓存(离线不反复重试)。"""
    key = tokenizer_model or _DEFAULT_ENCODER_NAME
    with _ENCODER_CACHE_LOCK:
        if key in _UNAVAILABLE_ENCODERS:
            _UNAVAILABLE_ENCODERS.move_to_end(key)
            raise RuntimeError(f"tokenizer encoder unavailable: {key}")
        cached = _ENCODER_CACHE.get(key)
        if cached is not None:
            _ENCODER_CACHE.move_to_end(key)
            return cached
    try:
        encoder = _load_encoder(tokenizer_model)
    except Exception:
        with _ENCODER_CACHE_LOCK:
            _UNAVAILABLE_ENCODERS[key] = None
            _UNAVAILABLE_ENCODERS.move_to_end(key)
            while len(_UNAVAILABLE_ENCODERS) > _MAX_ENCODER_CACHE:
                _UNAVAILABLE_ENCODERS.popitem(last=False)
        raise
    with _ENCODER_CACHE_LOCK:
        _ENCODER_CACHE[key] = encoder
        _ENCODER_CACHE.move_to_end(key)
        while len(_ENCODER_CACHE) > _MAX_ENCODER_CACHE:
            _ENCODER_CACHE.popitem(last=False)
    return encoder


def count_text_tokens(text: str, *, tokenizer_model: str | None = None) -> TokenMeasurement:
    """唯一文本估算入口:tiktoken 失败绝不抛穿,退回有界启发式。"""
    try:
        encoder = _get_encoder(tokenizer_model)
        count = len(encoder.encode(text, disallowed_special=()))
        return TokenMeasurement(count=count, source="bpe")
    except Exception:  # noqa: BLE001 — 离线/缺资源/未知模型都必须可继续运行
        return TokenMeasurement(
            count=len(text) // _HEURISTIC_CHARS_PER_TOKEN + 1, source="heuristic"
        )


def measure_request(req: WireRequest) -> TokenMeasurement:
    """测量一次请求的唯一出站文本(serialized_json),不重建任何消息。"""
    return count_text_tokens(req.serialized_json, tokenizer_model=req.tokenizer_model)


__all__ = ["TokenMeasurement", "count_text_tokens", "measure_request"]
