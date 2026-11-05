"""Bounded native callback ingress. Administration remains local CLI/SDK."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from agent_runtime.channels.providers import ProviderError
from agent_runtime.channels.service import Channels
from agent_runtime.harness import RuntimeStore


def application(channels: Channels) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.post("/events/{source_id}")
    async def receive(source_id: str, request: Request) -> Any:
        # Fixed upper bound includes config lookup, streaming, signature parsing,
        # authority and SQLite commit. Timeout cannot erase a committed receipt.
        started = time.monotonic()
        try:
            async with asyncio.timeout(60):
                async with await RuntimeStore.open(channels.database, read_only=True) as store:
                    source = await store.run_async(channels._source, store, source_id)
                remaining = source.timeout_seconds - (time.monotonic() - started)
                if remaining <= 0:
                    raise TimeoutError
                async with asyncio.timeout(remaining):
                    chunks, size = [], 0
                    async for chunk in request.stream():
                        size += len(chunk)
                        if size > source.max_body_bytes:
                            return JSONResponse({"status": "too_large"}, status_code=413)
                        chunks.append(chunk)
                    result = await channels.receive(source_id, request.headers, b"".join(chunks))
                    if time.monotonic() - started >= source.timeout_seconds:
                        raise TimeoutError
                    return JSONResponse(result)
        except PermissionError:
            return JSONResponse({"status": "forbidden"}, status_code=403)
        except ProviderError as error:
            return JSONResponse({"status": error.reason}, status_code=401 if error.reason == "unauthenticated" else 400)
        except TimeoutError:
            return JSONResponse({"status": "deadline_exceeded"}, status_code=408)
        except (KeyError, ValueError):
            return JSONResponse({"status": "invalid_or_unbound"}, status_code=400)

    return app
