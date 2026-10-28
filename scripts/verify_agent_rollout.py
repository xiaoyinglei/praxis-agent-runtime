#!/usr/bin/env python3
"""Verify or rebuild Harness projections from canonical RolloutRecords."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

from agent_runtime.harness import RolloutStore


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="verify-agent-rollout")
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--backup", type=Path, help="Publish a verified DB and artifact snapshot in a new directory")
    args = parser.parse_args(argv)
    with RolloutStore.open_sync(args.database, read_only=not args.rebuild) as store:
        if args.rebuild:
            store.rebuild_projections()

        def audit():
            with store.read_snapshot():
                return _audit_snapshot(store, backup=args.backup)

        output = store.run_sync(audit)
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0 if output["valid"] else 1


def _audit_snapshot(store: RolloutStore, *, backup: Path | None) -> dict[str, object]:
    report = store.verify()
    if not report.valid:
        # Corrupt canonical payloads must still yield a machine-readable audit;
        # decoding them again for a fingerprint would hide the report in a traceback.
        return {
            "errors": list(report.errors),
            "projection_hashes": store.projection_hashes(),
            "record_chain_sha256": None,
            "record_count": store.count('rollout_records'),
            "valid": False,
        }
    chain = hashlib.sha256()
    count, last = 0, 0
    while records := store.list_global_records(after_record_id=last, limit=256):
        for record in records:
            count += 1
            last = record.record_id
            envelope = {
                "payload_hash": record.payload_hash,
                "payload_schema_version": record.payload_schema_version,
                "producer": record.producer,
                "record_id": record.record_id,
                "record_type": record.record_type,
                "record_uuid": record.record_uuid,
                "thread_id": record.thread_id,
                "thread_sequence": record.thread_sequence,
                "turn_id": record.turn_id,
            }
            chain.update(
                json.dumps(
                    envelope,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode()
            )
            chain.update(b"\n")
    output: dict[str, object] = {
        "errors": list(report.errors),
        "projection_hashes": store.projection_hashes(),
        "record_chain_sha256": chain.hexdigest(),
        "record_count": count,
        "valid": report.valid,
    }
    if backup is not None and report.valid:
        output["backup"] = str(store.backup(backup))
    return output


if __name__ == "__main__":
    raise SystemExit(main())
