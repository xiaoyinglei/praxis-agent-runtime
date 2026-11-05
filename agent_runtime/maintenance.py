"""Model-free maintenance for the canonical Rollout database and its artifacts."""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from pathlib import Path

from agent_runtime.harness.ownership import acquire_background_owner
from agent_runtime.harness.records import VerificationReport
from agent_runtime.harness.rollout import DATABASE_LOCATION_KEY, RolloutStore, database_origins
from agent_runtime.harness.rollout_backup import backup_store


def verify_rollout(database: Path) -> VerificationReport:
    """Audit SQLite, canonical facts, projections and referenced artifact bytes."""
    with RolloutStore.open_sync(Path(database).expanduser(), read_only=True) as store:
        def verify() -> VerificationReport:
            with store.read_snapshot():
                errors = tuple(
                    f"SQLite integrity: {row[0]}"
                    for row in store._connection.execute("PRAGMA quick_check")
                    if row[0] != "ok"
                )
                if store._connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
                    errors += ("SQLite foreign key check failed",)
                errors += store.verify().errors
                return VerificationReport(valid=not errors, errors=errors)

        return store.run_sync(verify)


def _verify(database: Path) -> None:
    report = verify_rollout(database)
    if not report.valid:
        raise RuntimeError("Rollout verification failed: " + "; ".join(report.errors))


def backup_rollout(database: Path, destination: Path) -> Path:
    """Publish a verified point-in-time backup in a new directory.

    The database and immutable artifact blobs retain their existing schema and
    identities. Returns the database path inside the published directory.
    """
    database = Path(database).expanduser().resolve(strict=True)
    with RolloutStore.open_sync(database, read_only=True) as store:
        directory = backup_store(
            store, Path(destination).expanduser(), validate=_verify, source_database=database,
        )
    return directory / "rollout.sqlite3"


def restore_rollout(backup: Path, destination: Path) -> Path:
    """Validate a backup and publish an independent copy at a new destination."""
    backup = Path(backup).expanduser()
    manifest = json.loads((backup / "manifest.json").read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("format") != "praxis-rollout-backup"
        or manifest.get("format_version") != 1
        or manifest.get("database") != "rollout.sqlite3"
        or manifest.get("artifacts_directory") != "rollout.sqlite3.artifacts"
    ):
        raise ValueError("unsupported Rollout backup manifest")
    source = manifest.get("source_database")
    if source is not None and (not isinstance(source, str) or not Path(source).is_absolute()):
        raise ValueError("Rollout backup database location must be absolute")
    destination = Path(destination).expanduser().absolute()
    published = destination.parent.resolve() / destination.name / "rollout.sqlite3"

    def validate(staged: Path) -> None:
        if source is not None:
            _record_move(staged, Path(source), published)
        _verify(staged)

    with RolloutStore.open_sync(backup / "rollout.sqlite3", read_only=True) as store:
        directory = backup_store(
            store, destination, validate=validate, expected_manifest=manifest, source_database=published,
        )
    return directory / "rollout.sqlite3"


def upgrade_rollout(database: Path, destination: Path) -> Path:
    """Migrate a stopped Rollout to a verified new directory.

    Stop foreground writers and input producers before the upgrade. This call
    takes the host's exclusive lock, makes a consistent copy, migrates and audits
    it before publication, and returns the new database path.
    """
    database = Path(database).expanduser().resolve(strict=True)
    destination = Path(destination).expanduser().absolute()
    published = destination.parent.resolve() / destination.name / "rollout.sqlite3"

    def migrate(staged: Path) -> None:
        with RolloutStore.open_sync(staged):
            pass
        with closing(sqlite3.connect(staged, autocommit=True)) as connection:
            connection.execute("PRAGMA journal_mode = DELETE")
        _record_move(staged, database, published)
        _verify(staged)

    descriptor = acquire_background_owner(database)
    try:
        with RolloutStore.open_sync(database, read_only=True) as store:
            directory = backup_store(store, destination, validate=migrate, source_database=published)
    finally:
        os.close(descriptor)
    return directory / "rollout.sqlite3"


def _record_move(staged: Path, source: Path, destination: Path) -> None:
    with closing(sqlite3.connect(staged, autocommit=True)) as connection:
        prior = connection.execute("SELECT value FROM store_metadata WHERE key=?", (DATABASE_LOCATION_KEY,)).fetchone()
        origins = () if prior is None else database_origins(prior[0], source)
        location = {"database": str(destination), "origins": list(dict.fromkeys(map(str, (source, *origins))))}
        connection.execute(
            "INSERT OR REPLACE INTO store_metadata (key, value) VALUES (?, ?)",
            (DATABASE_LOCATION_KEY, json.dumps(location)),
        )
