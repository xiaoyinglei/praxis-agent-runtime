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
    args = parser.parse_args(argv)
    with RolloutStore(args.database) as store:
        if args.rebuild:
            store.rebuild_projections()
        report = store.verify()
        records = store.list_global_records()
        chain = hashlib.sha256()
        for record in records:
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
        output = {
            "errors": list(report.errors),
            "projection_hashes": store.projection_hashes(),
            "record_chain_sha256": chain.hexdigest(),
            "record_count": len(records),
            "valid": report.valid,
        }
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
