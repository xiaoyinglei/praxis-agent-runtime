#!/usr/bin/env python3
"""Measure growing Rollout histories against an explicit source snapshot.

Wall-clock results are diagnostics, not machine-independent release thresholds.
SQLite VM work and returned row counts distinguish unavoidable history reads
from avoidable history scans on each event. No model or provider calls occur.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def measure(operation: Callable[[], Any], samples: int) -> dict[str, float]:
    times = []
    for _ in range(samples):
        start = time.perf_counter()
        operation()
        times.append((time.perf_counter() - start) * 1000)
    return {"median_ms": round(statistics.median(times), 4), "max_ms": round(max(times), 4)}


def benchmark(runtime: Path, sizes: list[int], payload_bytes: int, samples: int) -> dict[str, Any]:
    sys.path.insert(0, str(runtime))
    import agent_runtime
    from agent_runtime.harness import RolloutStore

    if not Path(agent_runtime.__file__).resolve().is_relative_to(runtime):
        raise RuntimeError("benchmark imported a different runtime source")
    hashes = {
        p.relative_to(runtime).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (runtime / "agent_runtime").rglob("*.py") if "__pycache__" not in p.parts
    }
    rows = []
    with tempfile.TemporaryDirectory(prefix="praxis-history-benchmark-") as directory:
        workspace = Path(directory)
        database = workspace / "rollout.sqlite3"
        with RolloutStore(database) as store:
            thread = store.create_thread(workspace=workspace)
            count = 0
            for size in sizes:
                while count < size:
                    turn = store.start_turn(
                        thread_id=thread.thread_id, user_message="u" * payload_bytes, binding_manifest={},
                    )
                    store.complete_turn(turn_id=turn.turn_id, answer="a" * payload_bytes)
                    count += 1
                def settings() -> None:
                    store.update_session_settings(thread.thread_id, {"model_alias": "benchmark"})

                result: dict[str, Any] = {"turns": size, "nominal_history_text_bytes": size * payload_bytes * 2}
                result["settings"] = measure(settings, samples)
                work = 0

                def progress() -> int:
                    nonlocal work
                    work += 1
                    return 0

                store._connection.set_progress_handler(progress, 1)
                statements: list[str] = []
                store._connection.set_trace_callback(statements.append)
                settings()
                store._connection.set_trace_callback(None)
                store._connection.set_progress_handler(None, 0)
                result["settings_vm_steps"] = work
                result["settings_inserts"] = sum(s.lstrip().startswith("INSERT INTO") for s in statements)
                active = store.start_turn(
                    thread_id=thread.thread_id, user_message="measure append", binding_manifest={},
                )
                result["item_append"] = measure(
                    lambda turn=active: store.record_final_proposal(turn_id=turn.turn_id, answer="a" * payload_bytes),
                    samples,
                )
                result["context_read"] = measure(lambda turn=active: store.list_context_items(turn.turn_id), samples)
                result["context_item_count"] = len(store.list_context_items(active.turn_id))
                store.complete_turn(turn_id=active.turn_id, answer="benchmark complete")
                count += 1

                def verify() -> None:
                    report = store.verify()
                    if not report.valid:
                        raise RuntimeError(str(report.errors))

                result["full_verify"] = measure(verify, samples)

                def reopen() -> None:
                    with RolloutStore(database) as reopened:
                        report = reopened.verify()
                        if not report.valid:
                            raise RuntimeError(str(report.errors))

                result["verified_reopen"] = measure(reopen, samples)
                result["database_bytes"] = sum(
                    p.stat().st_size for p in workspace.glob("rollout.sqlite3*") if p.is_file()
                )
                rows.append(result)
                print(json.dumps(result, sort_keys=True), flush=True)
    return {
        "schema_version": "praxis-history-performance-v1",
        "runtime_source_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
        "platform": platform.platform(), "python": platform.python_version(),
        "samples": samples, "payload_bytes_per_message": payload_bytes, "measurements": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, default=ROOT)
    parser.add_argument("--sizes", nargs="+", type=int, default=[10, 100, 1000])
    parser.add_argument("--payload-bytes", type=int, default=2048)
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.samples < 1 or args.payload_bytes < 1 or not args.sizes or any(n < 1 for n in args.sizes):
        parser.error("sizes, payload bytes and samples must be positive")
    if args.sizes != sorted(set(args.sizes)):
        parser.error("sizes must be strictly increasing")
    result = benchmark(args.runtime_root.resolve(), args.sizes, args.payload_bytes, args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
