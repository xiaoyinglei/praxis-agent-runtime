from __future__ import annotations

import os
import re
from collections.abc import Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

MAX_RUNTIME_DIAGNOSTICS = 20
MAX_RUNTIME_DIAGNOSTIC_MESSAGE_LENGTH = 500
_SENSITIVE_ENV_NAME_PARTS = (
    "KEY",
    "PASSWORD",
    "SECRET",
    "TOKEN",
)
_CREDENTIAL_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:gsk_|sk-|ak[-_])[A-Za-z0-9_-]{8,}",
    flags=re.IGNORECASE,
)
_BEARER_TOKEN_RE = re.compile(
    r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}",
    flags=re.IGNORECASE,
)


def redact_sensitive_text(value: object) -> str:
    """Remove process secrets and common provider credential identifiers."""

    redacted = str(value)
    secrets = sorted(
        {
            secret
            for name, secret in os.environ.items()
            if (
                secret
                and len(secret) >= 8
                and any(
                    marker in name.upper()
                    for marker in _SENSITIVE_ENV_NAME_PARTS
                )
            )
        },
        key=len,
        reverse=True,
    )
    for secret in secrets:
        redacted = redacted.replace(secret, "[REDACTED]")
    redacted = _CREDENTIAL_TOKEN_RE.sub("[REDACTED]", redacted)
    return _BEARER_TOKEN_RE.sub("Bearer [REDACTED]", redacted)


class RuntimeDiagnostic(BaseModel):
    """Bounded, checkpoint-safe description of degraded runtime behavior."""

    model_config = ConfigDict(frozen=True)

    code: str = Field(min_length=1, max_length=120)
    component: str = Field(min_length=1, max_length=120)
    message: str = Field(min_length=1, max_length=MAX_RUNTIME_DIAGNOSTIC_MESSAGE_LENGTH)
    severity: Literal["warning", "error"] = "warning"
    degraded: bool = True
    error_type: str | None = Field(default=None, max_length=120)

    @property
    def identity(self) -> tuple[str, str]:
        return self.code, self.component

    @classmethod
    def from_exception(
        cls,
        *,
        code: str,
        component: str,
        error: Exception,
        severity: Literal["warning", "error"] = "warning",
        degraded: bool = True,
    ) -> RuntimeDiagnostic:
        message = redact_sensitive_text(error).strip() or type(error).__name__
        return cls(
            code=code,
            component=component,
            message=message[:MAX_RUNTIME_DIAGNOSTIC_MESSAGE_LENGTH],
            severity=severity,
            degraded=degraded,
            error_type=type(error).__name__[:120],
        )


class AgentLatencyProfile(BaseModel):
    """Per-run latency profile for product Agent execution."""

    model_config = ConfigDict(frozen=True)

    startup_ms: float = 0.0
    build_service_ms: float = 0.0
    model_ready_ms: float = 0.0
    prepare_latency_ms: float = 0.0
    model_latency_ms: float = 0.0
    tool_latency_ms: float = 0.0
    finalize_latency_ms: float = 0.0
    total_ms: float = 0.0
    prompt_bytes: int = 0
    tool_schema_bytes: int = 0


def merge_runtime_diagnostics(
    left: Iterable[RuntimeDiagnostic],
    right: Iterable[RuntimeDiagnostic],
) -> list[RuntimeDiagnostic]:
    merged: dict[tuple[str, str], RuntimeDiagnostic] = {}
    for diagnostic in [*left, *right]:
        key = diagnostic.identity
        merged.pop(key, None)
        merged[key] = diagnostic
    return list(merged.values())[-MAX_RUNTIME_DIAGNOSTICS:]


__all__ = [
    "MAX_RUNTIME_DIAGNOSTICS",
    "AgentLatencyProfile",
    "RuntimeDiagnostic",
    "merge_runtime_diagnostics",
    "redact_sensitive_text",
]
