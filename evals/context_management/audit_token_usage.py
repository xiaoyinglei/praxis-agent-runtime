"""Offline, read-only comparison of reserved input estimates and provider usage."""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


def audit(directory: Path) -> dict:
    report = json.loads((directory / "report.json").read_text())
    rows = []
    with sqlite3.connect(f"file:{directory.resolve() / 'session.db'}?mode=ro", uri=True) as connection:
        for operation, encoded, response_id in connection.execute(
            "SELECT operation_id, request_ref_json, response_item_id FROM model_operations "
            "ORDER BY applied_thread_sequence"
        ):
            request = json.loads(encoded)
            response = connection.execute("SELECT payload_json FROM items WHERE item_id=?", (response_id,)).fetchone()
            usage = json.loads(response[0]).get("usage", {}) if response else {}
            projection = request.get("context_projection", {})
            estimated, actual = projection.get("input_tokens"), usage.get("input_tokens")
            if not isinstance(estimated, int) or not isinstance(actual, int) or actual <= 0:
                continue
            rows.append({
                "operation_id": operation,
                "purpose": request.get("purpose"),
                "count_source": projection.get("count_source", "not_recorded"),
                "estimated_input": estimated,
                "provider_input": actual,
                "estimate_over_actual": estimated / actual,
                "underestimated_by": max(0, actual - estimated),
                "effective_input_limit": projection.get("max_input_tokens"),
            })
    return {
        "report": str(directory / "report.json"),
        "original_pass": report.get("pass"),
        "phases": [{"phase": p["phase"], "checks": p["checks"], "usage": p["usage"],
                    "summary_calls": p["summary_calls"]} for p in report.get("phases", [])],
        "max_underestimated_by": max((r["underestimated_by"] for r in rows), default=0),
        "ratio_min": min((r["estimate_over_actual"] for r in rows), default=None),
        "ratio_max": max((r["estimate_over_actual"] for r in rows), default=None),
        "requests": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"network_calls": 0, "source_reports_modified": False,
              "runs": [audit(path) for path in args.directories]}
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
