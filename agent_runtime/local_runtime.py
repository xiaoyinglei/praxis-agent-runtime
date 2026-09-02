from __future__ import annotations

from collections.abc import (
    Awaitable,
    Callable,
)
from typing import TYPE_CHECKING

import httpx

from agent_runtime.core.llm_registry import ModelNotAvailableError

if TYPE_CHECKING:
    from agent_runtime.models import ModelSpec


class LocalRuntimeError(ModelNotAvailableError):
    """Configured local model provider is unavailable."""


class EndpointConflictError(LocalRuntimeError):
    """Health endpoint is alive, but serves a different model."""


class LocalProviderProbe:
    """
    Probe an independently managed local model provider.

    This object does not start, stop, own, or supervise provider processes.
    Ollama, LM Studio, MLX server, or another local provider must already be
    running before the model is resolved.
    """

    def __init__(
        self,
        *,
        request_json: Callable[
            [str, float],
            Awaitable[object],
        ]
        | None = None,
    ) -> None:
        self._request_json = (
            request_json
            or _request_json
        )

    async def ensure_ready(self,spec: ModelSpec,) -> None:
        if getattr(spec, "location", None) != "local":
            return

        runtime = getattr(spec, "runtime", None)
        health_url = (
            getattr(runtime, "health_url", None)
            if runtime is not None
            else None
        )

        if not health_url:
            raise LocalRuntimeError(
                f"Local model {spec.id!r} has no runtime.health_url"
            )

        expected = (
            getattr(runtime, "expected_model_contains", None)
            if runtime is not None
            else None
        ) or getattr(spec, "provider_model", "")

        try:
            payload = await self._request_json(str(health_url),5.0,)
    
        except Exception as exc:
            raise LocalRuntimeError(
                _provider_not_running_message(
                    spec=spec,
                    health_url=str(health_url),
                )
            ) from exc

        _raise_if_unexpected_model(
            payload,
            expected=str(expected),
            model_id=str(getattr(spec, "id", "unknown")),
            health_url=str(health_url),
        )


def _provider_not_running_message(
    *,
    spec: ModelSpec,
    health_url: str,
) -> str:
    runtime = getattr(spec, "runtime", None)
    launch_command = (
        getattr(runtime, "launch_command", ())
        if runtime is not None
        else ()
    )

    message = (
        f"Local provider for model {spec.id!r} is not running "
        f"or is unreachable at {health_url}. "
        "Start the local model provider before running Praxis."
    )

    if launch_command:
        command = " ".join(str(part) for part in launch_command)
        message += f" Configured start command: {command}"

    return message


async def _request_json(
    url: str,
    timeout: float,
) -> object:
    async with httpx.AsyncClient(
        trust_env=False,
        timeout=httpx.Timeout(timeout),
    ) as client:
        response = await client.get(url)

        response.raise_for_status()

        return response.json()


def _raise_if_unexpected_model(
    payload: object,
    *,
    expected: str,
    model_id: str,
    health_url: str,
) -> None:
    if not expected:
        return

    model_names = _model_names(payload)

    if any(
        expected in name
        for name in model_names
    ):
        return

    raise EndpointConflictError(
        f"endpoint conflict for {model_id!r}: "
        f"{health_url} is serving "
        f"{model_names or ['<no models>']}, "
        f"expected model containing {expected!r}"
    )


def _model_names(
    payload: object,
) -> list[str]:
    if isinstance(payload, dict):
        data = payload.get("data")

        if isinstance(data, list):
            names: list[str] = []

            for item in data:
                if isinstance(item, dict):
                    value = (
                        item.get("id")
                        or item.get("model")
                    )
                    if value is not None:
                        names.append(str(value))

                elif item is not None:
                    names.append(str(item))

            return names

        if "id" in payload:
            return [str(payload["id"])]

    if isinstance(payload, list):
        return [
            str(item)
            for item in payload
        ]

    return []

async def ensure_local_provider_ready(
    spec: ModelSpec,
) -> None:
    if spec.location != "local":
        return

    probe = LocalProviderProbe()

    await probe.ensure_ready(spec)


__all__ = [
    "EndpointConflictError",
    "LocalProviderProbe",
    "LocalRuntimeError",
    "ensure_local_provider_ready",
]