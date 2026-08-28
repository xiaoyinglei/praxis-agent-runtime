#!/usr/bin/env python3
"""Explicit offline recovery for a Turn orphaned before provider/tool dispatch."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from agent_runtime.harness import RolloutStore


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="recover-agent-turn")
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--turn-id", required=True)
    parser.add_argument("--reason", required=True)
    parser.add_argument(
        "--maintenance-confirmed",
        action="store_true",
        help="Confirm the prior process/worker has stopped.",
    )
    args = parser.parse_args(argv)
    with RolloutStore(args.database) as store:
        turn = store.interrupt_orphaned_turn(
            turn_id=args.turn_id,
            reason=args.reason,
            maintenance_confirmed=args.maintenance_confirmed,
        )
        verification = store.verify()
        if not verification.valid:
            raise RuntimeError(f"Rollout verification failed after recovery: {verification.errors}")
    print(
        json.dumps(
            {
                "status": turn.status,
                "thread_id": turn.thread_id,
                "turn_id": turn.turn_id,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
