"""Compare verified saved web text without interpreting task or business meaning."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from agent_runtime.tools.web_http import PublicWebError, validate_public_url


def compare_saved_sources(
    current: Mapping[str, Any], baseline: Mapping[str, Any] | None, *,
    current_source_id: str, baseline_source_id: str | None = None,
) -> dict[str, Any]:
    """Compare full saved-text hashes from validated, hash-verified source mappings.

    The caller owns source loading, permission checks and text/hash verification.
    Missing baselines, incomplete extractions or renderer loads, different final
    URLs and unknown or unordered retrieval times never establish change or equality. A changed hash
    proves only that saved text differs; it cannot establish business changes.
    Output has fixed fields, bounded provenance and no copied page text or diff.
    """
    result: dict[str, Any] = {
        "status": "baseline_missing", "reason": "No saved baseline is available.",
        "scope": "saved_text",
        "limitation": "Saved text comparison does not establish business changes or their publication time.",
        "baseline": _provenance(baseline, baseline_source_id),
        "current": _provenance(current, current_source_id),
    }
    if baseline is None:
        return result
    if any(
        source.get("truncated") or source.get("extraction_error_code") or source.get("extraction_error_message")
        or source.get("extraction_method") in {"failed", "not_attempted"}
        or not source.get("content_hash") or _incomplete_render(source)
        for source in (baseline, current)
    ):
        result.update(status="incomplete", reason="At least one saved text snapshot is incomplete or unavailable.")
        return result
    baseline_url, current_url = _source_url(baseline), _source_url(current)
    if baseline_url is None or current_url is None or baseline_url != current_url:
        result.update(status="incomparable", reason="Snapshots do not identify the same normalized source URL.")
        return result
    baseline_time, current_time = _retrieval_time(baseline), _retrieval_time(current)
    if baseline_time is None or current_time is None:
        result.update(status="incomparable", reason="Retrieval times must be valid timestamps with explicit timezones.")
        return result
    if baseline_time >= current_time:
        result.update(status="incomparable",
                      reason="The baseline retrieval must be earlier than the current retrieval.")
        return result
    changed = baseline["content_hash"] != current["content_hash"]
    result.update(status="changed" if changed else "unchanged",
                  reason="Saved text hashes differ." if changed else "Saved text hashes match.")
    return result


def _incomplete_render(source: Mapping[str, Any]) -> bool:
    diagnostics = source.get("render_diagnostics")
    if not isinstance(diagnostics, Mapping):
        return False
    return bool(diagnostics.get("blocked_actions") or diagnostics.get("resource_failures")
                or diagnostics.get("load_timeouts") or diagnostics.get("snapshot_available") is False)


def _provenance(source: Mapping[str, Any] | None, source_id: str | None) -> dict[str, str | None]:
    value = source if source is not None else {}
    fields = {"source_id": source_id, "url": value.get("url"), "fetched_at": value.get("fetched_at"),
              "content_hash": value.get("content_hash")}
    limits = {"source_id": 128, "url": 4096, "fetched_at": 128, "content_hash": 64}
    return {key: item[:limits[key]] if isinstance(item, str) else None for key, item in fields.items()}


def _source_url(source: Mapping[str, Any]) -> str | None:
    value = source.get("url")
    if not isinstance(value, str) or len(value) > 4096:
        return None
    try:
        return str(validate_public_url(value))
    except PublicWebError:
        return None


def _retrieval_time(source: Mapping[str, Any]) -> datetime | None:
    value = source.get("fetched_at")
    if not isinstance(value, str) or len(value) > 128:
        return None
    try:
        timestamp = datetime.fromisoformat(value)
        if timestamp.tzinfo is None:
            return None
        return timestamp.astimezone(UTC)
    except (ValueError, OverflowError):
        return None
