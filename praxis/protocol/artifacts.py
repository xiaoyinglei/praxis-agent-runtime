"""artifact 读契约(设计 §5.7;D10)。

内容寻址 artifact 的**唯一落盘实现**在 ``praxis.store.artifacts``;本模块只声明
读侧结构契约与错误类型。这样 ``praxis.tools`` 的 ``read_artifact`` 可以消费
store 的读取能力而不 import ``praxis.store``(import 契约:
``praxis_tools_adapters_boundary`` 禁止 tools 依赖 store),组合根负责把
``ArtifactStore`` 按结构传入。

``ArtifactIntegrityError`` 定义在协议层而非 store:它是一次读的**契约性失败**,
工具必须能按类型把它映射为 ``artifact_integrity`` 结果;store 模块 re-export
同名符号,外部仍可从 ``praxis.store.artifacts`` 导入。
"""

from __future__ import annotations

from typing import Protocol


class ArtifactIntegrityError(RuntimeError):
    """artifact 内容与其 digest 不符(读后重算 sha256 失败)。"""


class ArtifactReader(Protocol):
    """read_artifact 工具所需的最小读取面(由 ArtifactStore 结构满足)。"""

    def get(self, digest: str) -> bytes | None:
        """返回 artifact 全文;文件不存在返回 None;完整性不符抛 ArtifactIntegrityError。"""
        ...