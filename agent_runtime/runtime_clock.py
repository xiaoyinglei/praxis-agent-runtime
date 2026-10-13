"""Capture runtime-owned local clock facts once per new model step."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def capture_runtime_clock() -> dict[str, str]:
    instant = datetime.now(UTC)
    local = instant.astimezone()
    name = os.environ.get("TZ", "").removeprefix(":")
    if not name:
        try:
            target = str(Path("/etc/localtime").resolve(strict=True))
            name = target.partition("/zoneinfo/")[2]
        except OSError:
            name = ""
    if name:
        try:
            local = instant.astimezone(ZoneInfo(name))
        except (ZoneInfoNotFoundError, ValueError):
            name = ""
    offset = local.strftime("%z")
    return {
        "current_time": local.isoformat(timespec="seconds"),
        "current_date": local.date().isoformat(),
        "timezone": name or local.tzname() or "local",
        "utc_offset": offset[:3] + ":" + offset[3:],
    }
