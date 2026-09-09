from __future__ import annotations

from typing import Protocol


class ProviderQuotaPreflightError(RuntimeError):
    """External quota rejected work before provider I/O."""


class ProviderQuotaGate(Protocol):
    async def acquire(
        self,
        *,
        provider: str,
        model: str,
        requested_tokens: int,
    ) -> None: ...


async def acquire_provider_quota(
    gate: ProviderQuotaGate | None,
    *,
    provider: str,
    model: str,
    requested_tokens: int,
) -> None:
    if gate is None:
        return
    if not provider or not model:
        raise ValueError("provider and model must be non-empty")
    if (
        isinstance(requested_tokens, bool)
        or not isinstance(requested_tokens, int)
        or requested_tokens < 0
    ):
        raise ValueError("requested_tokens must be a non-negative integer")
    try:
        await gate.acquire(
            provider=provider,
            model=model,
            requested_tokens=requested_tokens,
        )
    except ProviderQuotaPreflightError:
        raise
    except Exception as exc:
        raise ProviderQuotaPreflightError(
            f"provider quota gate failed closed: {type(exc).__name__}"
        ) from exc


__all__ = [
    "ProviderQuotaGate",
    "ProviderQuotaPreflightError",
    "acquire_provider_quota",
]
