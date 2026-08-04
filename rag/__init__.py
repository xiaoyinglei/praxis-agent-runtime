"""Core RAG library public exports."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

__all__ = [
    "AssemblyConfig",
    "AssemblyDiagnostics",
    "AssemblyOverrides",
    "AssemblyRequest",
    "CapabilityAssemblyService",
    "CapabilityRequirements",
    "RAGRuntime",
    "StorageComponentConfig",
    "StorageConfig",
]

if TYPE_CHECKING:
    from rag.assembly import (  # noqa: F401, E501
        AssemblyConfig,
        AssemblyDiagnostics,
        AssemblyOverrides,
        AssemblyRequest,
        CapabilityAssemblyService,
        CapabilityRequirements,
    )
    from rag.runtime import RAGRuntime  # noqa: F401
    from rag.storage import StorageComponentConfig, StorageConfig  # noqa: F401

_EXPORTS = {
    "AssemblyConfig": ("rag.assembly", "AssemblyConfig"),
    "AssemblyDiagnostics": ("rag.assembly", "AssemblyDiagnostics"),
    "AssemblyOverrides": ("rag.assembly", "AssemblyOverrides"),
    "AssemblyRequest": ("rag.assembly", "AssemblyRequest"),
    "CapabilityAssemblyService": ("rag.assembly", "CapabilityAssemblyService"),
    "CapabilityRequirements": ("rag.assembly", "CapabilityRequirements"),
    "RAGRuntime": ("rag.runtime", "RAGRuntime"),
    "StorageComponentConfig": ("rag.storage", "StorageComponentConfig"),
    "StorageConfig": ("rag.storage", "StorageConfig"),
}


def __getattr__(name: str) -> object:
    export = _EXPORTS.get(name)
    if export is None:
        raise AttributeError(f"module 'rag' has no attribute {name!r}")
    module_name, attr_name = export
    module = import_module(module_name)
    return getattr(module, attr_name)
