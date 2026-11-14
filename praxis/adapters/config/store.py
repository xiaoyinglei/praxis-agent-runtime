"""全局模型配置库 ModelConfigStore(设计 §5.15;D4/D5;B26/B28)。

这是"用哪个模型、连哪个端点、单价多少"的唯一持久化来源,默认位于
``host_config_dir()/config.sqlite``(宿主私有目录,已在地板 deny 集合内——
代理绝不能自己改它)。凭据**绝不进这个库**:connections 只存
``api_key_env``(环境变量名),任何 token/key/secret 值都经 Task 6 的
credential 层保存,表结构由测试钉死。

并发与版本:
- 单进程内全部读写经 ``self._lock``(RLock)串行;sqlite 连接以
  ``check_same_thread=False`` 打开,由同一把锁保证线程安全。
- 跨进程写经 ``BEGIN IMMEDIATE`` + ``busy_timeout=5000`` 串行。
- 全局 revision 存在 ``model_config`` 的 ``revision`` 键,每次成功写 +1。
  CAS 不信任调用方/实例的内存视角:进入 IMMEDIATE 事务后 re-read 存储真值;
  首次不符时在同一事务内 re-read 一次(刷新观察值),仍不符才抛
  ``ConfigVersionConflict(observed_revision=...)``——至多一次,不 livelock
  (B26:外部写不再造成永久冲突)。
- 写事务只做最小 SQL;序列化/校验一律在 BEGIN 之前完成(会话库"写事务只
  INSERT"纪律的等价物:事务里不做哈希/重活)。
- 读经 ``OrderedDict`` LRU 缓存(``_CACHE_MAX = 64``)+ 同一把锁;任何写成功
  后清空缓存,保证同实例不脏读(B28:缓存有锁、有界)。

``definition`` 存的是 ModelSpec 去掉 ``cost`` 后的 JSON;``cost`` 单独存
CostMicros 的 JSON。``base_url``/``api_key_env``/``provider`` 只以
connections 行为准(读时填充),所以改连接会同步影响其下所有模型。
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from praxis.protocol.floor import host_config_dir
from praxis.protocol.json import freeze_json, thaw_json
from praxis.protocol.models import CostMicros, ModelProvider, ModelSpec

MODEL_CONFIG_SCHEMA_VERSION = "1"

_CACHE_MAX = 64

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS model_config(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS connections(name TEXT PRIMARY KEY, provider TEXT NOT NULL,
  base_url TEXT NOT NULL, api_key_env TEXT, location TEXT, updated_at REAL NOT NULL);
CREATE TABLE IF NOT EXISTS models(model_id TEXT PRIMARY KEY, connection TEXT NOT NULL,
  definition TEXT NOT NULL, cost TEXT NOT NULL, revision INTEGER NOT NULL, updated_at REAL NOT NULL);
"""

_SPEC_FIELD_NAMES = tuple(field.name for field in dataclasses.fields(ModelSpec) if field.name != "cost")
_COST_FIELD_NAMES = tuple(field.name for field in dataclasses.fields(CostMicros))

_MODEL_SELECT = """
SELECT m.definition, m.cost, m.revision,
       c.name AS connection_name, c.provider, c.base_url, c.api_key_env, c.location
FROM models AS m JOIN connections AS c ON c.name = m.connection
WHERE m.model_id = ?
"""


class ConfigVersionConflict(RuntimeError):  # noqa: N818 — 公开契约名由设计 §5.15 固定
    """CAS 失败:调用方的 expected_revision 与存储真值不符。

    ``observed_revision`` 是冲突发生时数据库里的最新 revision,供调用方
    刷新视角后干净重试(不要盲目重放旧值)。
    """

    def __init__(self, observed_revision: int) -> None:
        self.observed_revision = observed_revision
        super().__init__(
            f"model config revision conflict: observed {observed_revision}, "
            "reload and retry with the observed revision"
        )


@dataclass(frozen=True, slots=True)
class ConnectionEntry:
    """一个 provider 连接:provider/base_url/api_key_env 的唯一来源。

    ``api_key_env`` 是环境变量名(不是凭据值);``location`` 是可选的位置
    标签(如 cloud/local),仅用于展示与选择。
    """

    name: str
    provider: ModelProvider
    base_url: str
    api_key_env: str | None = None
    location: str | None = None


@dataclass(frozen=True, slots=True)
class ModelEntry:
    """库里的一行模型:spec + 所属连接 + 该行自身的 revision。"""

    spec: ModelSpec
    connection: ConnectionEntry
    revision: int


# ---------------------------------------------------------------------------
# 序列化:ModelSpec 自己的字段名即 JSON 形状(不另造模型定义)
# ---------------------------------------------------------------------------


def _spec_definition(spec: ModelSpec) -> str:
    """ModelSpec 去 cost 后的 JSON;provider_options 经 freeze/thaw 规范化为 JSON。"""
    data: dict[str, Any] = {name: getattr(spec, name) for name in _SPEC_FIELD_NAMES}
    data["provider_options"] = thaw_json(
        freeze_json(dict(spec.provider_options), path="$.provider_options")
    )
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def _cost_json(cost: CostMicros) -> str:
    return json.dumps(
        {name: getattr(cost, name) for name in _COST_FIELD_NAMES},
        sort_keys=True,
        separators=(",", ":"),
    )


def _spec_from_row(definition: str, cost_json: str, connection: ConnectionEntry) -> ModelSpec:
    """从 JSON 还原 ModelSpec;连接字段以 connections 行为唯一来源。"""
    raw: dict[str, Any] = json.loads(definition)
    raw["provider_options"] = freeze_json(
        raw.get("provider_options") or {}, path="$.provider_options"
    )
    cost_raw = json.loads(cost_json)
    raw["cost"] = CostMicros(**{name: int(cost_raw[name]) for name in _COST_FIELD_NAMES})
    raw["provider"] = connection.provider
    raw["base_url"] = connection.base_url
    raw["api_key_env"] = connection.api_key_env
    return ModelSpec(**raw)


def _read_revision(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT value FROM model_config WHERE key = 'revision'").fetchone()
    return int(row[0]) if row is not None else 0


class ModelConfigStore:
    """全局模型配置库的门面:连接/模型 CRUD + CAS + 有界缓存。"""

    def __init__(self, path: Path | None = None) -> None:
        self._path = Path(path) if path is not None else host_config_dir() / "config.sqlite"
        self._path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._lock = threading.RLock()
        self._cache: OrderedDict[str, ModelEntry] = OrderedDict()
        self._closed = False
        # check_same_thread=False 由 self._lock 串行化所有访问。
        self._conn = sqlite3.connect(self._path, isolation_level=None, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        try:
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA_SQL)
            self._initialize_metadata()
        except BaseException:
            self._conn.close()
            raise

    def _initialize_metadata(self) -> None:
        row = self._conn.execute(
            "SELECT value FROM model_config WHERE key = 'schema_version'"
        ).fetchone()
        if row is not None and str(row[0]) != MODEL_CONFIG_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported model config schema version {row[0]!r}; "
                f"expected {MODEL_CONFIG_SCHEMA_VERSION!r}"
            )
        self._conn.execute(
            "INSERT OR IGNORE INTO model_config(key, value) VALUES ('schema_version', ?)",
            (MODEL_CONFIG_SCHEMA_VERSION,),
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO model_config(key, value) VALUES ('revision', '0')"
        )

    # -- 生命周期 ----------------------------------------------------------------

    @property
    def revision(self) -> int:
        """当前全局 revision(每次读库真值,不使用缓存)。"""
        with self._lock:
            self._ensure_open()
            return _read_revision(self._conn)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._cache.clear()
            self._conn.close()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"model config store closed: {self._path}")

    # -- 写事务与 CAS --------------------------------------------------------------

    def _write(
        self, action: Callable[[sqlite3.Connection], None], *, expected_revision: int | None
    ) -> int:
        """BEGIN IMMEDIATE → CAS re-read → action(conn) → revision+1 → COMMIT。

        action 只做最小 SQL;序列化/校验在进入本函数前完成。
        """
        with self._lock:
            self._ensure_open()
            conn = self._conn
            conn.execute("BEGIN IMMEDIATE")
            try:
                stored = _read_revision(conn)
                if expected_revision is not None and stored != expected_revision:
                    # 同一 IMMEDIATE 事务内 re-read(刷新观察值)并重放校验一次;
                    # 第二次仍不符才报冲突——至多一次,避免 livelock(B26)。
                    stored = _read_revision(conn)
                    if stored != expected_revision:
                        raise ConfigVersionConflict(observed_revision=stored)
                action(conn)
                new_revision = _read_revision(conn) + 1
                conn.execute(
                    "UPDATE model_config SET value = ? WHERE key = 'revision'",
                    (str(new_revision),),
                )
                conn.execute("COMMIT")
            except BaseException:
                try:
                    conn.execute("ROLLBACK")
                except BaseException:
                    pass
                raise
            self._cache.clear()
            return new_revision

    # -- connections -------------------------------------------------------------

    def upsert_connection(
        self, entry: ConnectionEntry, *, expected_revision: int | None = None
    ) -> int:
        """新增/更新连接;返回新的全局 revision。"""
        now = time.time()
        params = (
            entry.name,
            entry.provider,
            entry.base_url,
            entry.api_key_env,
            entry.location,
            now,
        )

        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO connections(name, provider, base_url, api_key_env, location, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET provider = excluded.provider, "
                "base_url = excluded.base_url, api_key_env = excluded.api_key_env, "
                "location = excluded.location, updated_at = excluded.updated_at",
                params,
            )

        return self._write(_do, expected_revision=expected_revision)

    def get_connection(self, name: str) -> ConnectionEntry | None:
        with self._lock:
            self._ensure_open()
            row = self._conn.execute(
                "SELECT name, provider, base_url, api_key_env, location "
                "FROM connections WHERE name = ?",
                (name,),
            ).fetchone()
            if row is None:
                return None
            return _connection_from_row(row, name_key="name")

    def list_connections(self) -> tuple[ConnectionEntry, ...]:
        with self._lock:
            self._ensure_open()
            rows = self._conn.execute(
                "SELECT name, provider, base_url, api_key_env, location "
                "FROM connections ORDER BY name"
            )
            return tuple(_connection_from_row(row, name_key="name") for row in rows)

    # -- models -----------------------------------------------------------------

    def upsert_model(self, entry: ModelEntry, *, expected_revision: int | None = None) -> int:
        """新增/更新模型;返回新的全局 revision。

        连接必须已存在;``base_url``/``api_key_env``/``provider`` 不写进模型
        definition(读时由 connections 行填充,连接是唯一来源)。
        """
        connection = self.get_connection(entry.connection.name)
        if connection is None:
            raise KeyError(
                f"unknown connection {entry.connection.name!r}; "
                "run: praxis models connection add …"
            )
        stored_spec = dataclasses.replace(
            entry.spec,
            provider=connection.provider,
            base_url=None,
            api_key_env=None,
        )
        definition = _spec_definition(stored_spec)
        cost = _cost_json(entry.spec.cost)
        model_id = entry.spec.model_id
        now = time.time()

        def _do(conn: sqlite3.Connection) -> None:
            row = conn.execute(
                "SELECT revision FROM models WHERE model_id = ?", (model_id,)
            ).fetchone()
            row_revision = (int(row[0]) if row is not None else 0) + 1
            conn.execute(
                "INSERT INTO models(model_id, connection, definition, cost, revision, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(model_id) DO UPDATE SET connection = excluded.connection, "
                "definition = excluded.definition, cost = excluded.cost, "
                "revision = excluded.revision, updated_at = excluded.updated_at",
                (model_id, connection.name, definition, cost, row_revision, now),
            )

        return self._write(_do, expected_revision=expected_revision)

    def remove_model(self, model_id: str, *, expected_revision: int) -> None:
        """删除模型;若它是默认模型则同时清空默认。"""

        def _do(conn: sqlite3.Connection) -> None:
            cursor = conn.execute("DELETE FROM models WHERE model_id = ?", (model_id,))
            if cursor.rowcount == 0:
                raise KeyError(
                    f"model {model_id!r} is not registered; run: praxis models list"
                )
            conn.execute(
                "DELETE FROM model_config WHERE key = 'default_model' AND value = ?",
                (model_id,),
            )

        self._write(_do, expected_revision=expected_revision)

    def get(self, model_id: str) -> ModelEntry | None:
        """按 id 取模型;命中缓存则 LRU 提升,未命中读库并放入有界缓存。"""
        with self._lock:
            self._ensure_open()
            cached = self._cache.get(model_id)
            if cached is not None:
                self._cache.move_to_end(model_id)
                return cached
            row = self._conn.execute(_MODEL_SELECT, (model_id,)).fetchone()
            if row is None:
                return None
            entry = _row_to_entry(row)
            self._cache[model_id] = entry
            while len(self._cache) > _CACHE_MAX:
                self._cache.popitem(last=False)
            return entry

    def list_models(self) -> tuple[ModelEntry, ...]:
        with self._lock:
            self._ensure_open()
            ids = [
                str(row[0])
                for row in self._conn.execute("SELECT model_id FROM models ORDER BY model_id")
            ]
            entries = [entry for model_id in ids if (entry := self.get(model_id)) is not None]
            return tuple(entries)

    # -- 默认模型与解析 -------------------------------------------------------------

    def set_default_model(self, model_id: str, *, expected_revision: int) -> int:
        # 校验在 BEGIN 之前完成:写事务只做最小 SQL(不校验/不哈希/不重活)。
        if self.get(model_id) is None:
            raise KeyError(f"model {model_id!r} is not registered; run: praxis models list")

        def _do(conn: sqlite3.Connection) -> None:
            conn.execute(
                "INSERT INTO model_config(key, value) VALUES ('default_model', ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (model_id,),
            )

        return self._write(_do, expected_revision=expected_revision)

    def default_model(self) -> str | None:
        with self._lock:
            self._ensure_open()
            row = self._conn.execute(
                "SELECT value FROM model_config WHERE key = 'default_model'"
            ).fetchone()
            return str(row[0]) if row is not None else None

    def resolve_spec(self, model_id: str | None) -> ModelSpec:
        """解析运行期使用的 ModelSpec;缺失时报可行动的 KeyError。

        ``model_id is None`` → 默认模型。连接字段已由 connections 行填充。
        """
        if model_id is None:
            resolved = self.default_model()
            if resolved is None:
                if self.list_models():
                    raise KeyError(
                        "no default model; run: praxis models set-default <model-id>"
                    )
                raise KeyError("no models registered; run: praxis models add …")
            model_id = resolved
        entry = self.get(model_id)
        if entry is None:
            raise KeyError(
                f"model {model_id!r} is not registered; run: praxis models list"
            )
        return entry.spec


def _connection_from_row(row: sqlite3.Row, *, name_key: str) -> ConnectionEntry:
    return ConnectionEntry(
        name=str(row[name_key]),
        provider=cast("ModelProvider", row["provider"]),
        base_url=str(row["base_url"]),
        api_key_env=str(row["api_key_env"]) if row["api_key_env"] is not None else None,
        location=str(row["location"]) if row["location"] is not None else None,
    )


def _row_to_entry(row: sqlite3.Row) -> ModelEntry:
    connection = _connection_from_row(row, name_key="connection_name")
    spec = _spec_from_row(str(row["definition"]), str(row["cost"]), connection)
    return ModelEntry(spec=spec, connection=connection, revision=int(row["revision"]))