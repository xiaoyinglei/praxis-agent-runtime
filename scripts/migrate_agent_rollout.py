#!/usr/bin/env python3
"""Explicit offline migration from legacy Agent state to Rollout Harness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent_runtime.harness import migrate_legacy_turns, restore_legacy_backup


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("database", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backup", type=Path)
    parser.add_argument("--restore", type=Path)
    parser.add_argument(
        "--maintenance-confirmed",
        action="store_true",
        help="Confirm all legacy writers/workers are stopped before importing expired running Turns.",
    )
    args = parser.parse_args()
    if args.restore is not None:
        if args.dry_run or args.backup is not None or args.maintenance_confirmed:
            parser.error(
                "--restore cannot be combined with --dry-run, --backup, or "
                "--maintenance-confirmed"
            )
        restore_legacy_backup(database=args.database, backup=args.restore)
        print(json.dumps({"restored": str(args.database)}, sort_keys=True))
        return 0
    report = migrate_legacy_turns(
        args.database,
        dry_run=args.dry_run,
        backup_path=args.backup,
        maintenance_confirmed=args.maintenance_confirmed,
    )
    print(
        json.dumps(
            {
                "dry_run": args.dry_run,
                "migrated_turn_ids": report.migrated_turn_ids,
                "skipped_turn_ids": report.skipped_turn_ids,
                "blocked": dict(report.blocked),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 2 if report.blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
