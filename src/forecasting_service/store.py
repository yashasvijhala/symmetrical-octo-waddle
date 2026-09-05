import hashlib
import json
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class NotFoundError(KeyError):
    pass


class IdempotencyConflictError(ValueError):
    pass


class Store:
    """Transactional local metadata DB plus filesystem artifact storage.

    SQLite runs in WAL mode and opens a short-lived connection per operation, allowing API and
    worker threads/processes to share state safely. The public methods intentionally match a
    future PostgreSQL implementation.
    """

    collections = (
        "datasets",
        "experiments",
        "jobs",
        "models",
        "forecasts",
        "actuals",
        "idempotency",
        "webhooks",
    )

    def __init__(self, root: Path, max_upload_bytes: int = 2_147_483_648) -> None:
        self.root = root.resolve()
        self.max_upload_bytes = max_upload_bytes
        self._write_lock = threading.RLock()
        for name in ("uploads", "artifacts", "predictions"):
            (self.root / name).mkdir(parents=True, exist_ok=True)
        self.database = self.root / "metadata.sqlite3"
        self._initialize()

    @staticmethod
    def new_id(prefix: str) -> str:
        return f"{prefix}_{uuid4().hex[:20]}"

    def create(self, collection: str, record: dict[str, Any]) -> dict[str, Any]:
        self._validate_collection(collection)
        now = utc_now()
        value = {**record, "created_at": now, "updated_at": now}
        with self._write_lock, self._connect() as connection:
            connection.execute(
                "INSERT INTO records(collection,id,tenant_id,created_at,updated_at,payload) "
                "VALUES(?,?,?,?,?,?)",
                (
                    collection,
                    value["id"],
                    value.get("tenant_id"),
                    now,
                    now,
                    json.dumps(value, default=str, allow_nan=False),
                ),
            )
        return value

    def create_idempotent(
        self,
        records: list[tuple[str, dict[str, Any]]],
        tenant_id: str,
        scope: str,
        key: str | None,
        fingerprint: str,
    ) -> tuple[dict[str, Any], bool]:
        """Atomically create related records, returning the original result on a safe retry."""

        if not records:
            raise ValueError("at least one record is required")
        for collection, _ in records:
            self._validate_collection(collection)
        idempotency_id = (
            "idem_" + hashlib.sha256(f"{tenant_id}:{scope}:{key}".encode()).hexdigest()[:40]
            if key is not None
            else None
        )
        now = utc_now()
        values = [
            (collection, {**record, "created_at": now, "updated_at": now})
            for collection, record in records
        ]
        with self._write_lock, self._connect() as connection:
            if idempotency_id is not None:
                row = connection.execute(
                    "SELECT payload FROM records WHERE collection='idempotency' AND id=?",
                    (idempotency_id,),
                ).fetchone()
                if row is not None:
                    link = json.loads(row[0])
                    if link.get("fingerprint") != fingerprint:
                        raise IdempotencyConflictError(
                            "Idempotency-Key was already used with a different request payload"
                        )
                    primary_collection = values[0][0]
                    existing = connection.execute(
                        "SELECT payload FROM records WHERE collection=? AND id=?",
                        (primary_collection, link["resource_id"]),
                    ).fetchone()
                    if existing is None:
                        raise RuntimeError("idempotency record points to a missing resource")
                    return json.loads(existing[0]), False
            for collection, value in values:
                self._insert(connection, collection, value)
            if idempotency_id is not None:
                link = {
                    "id": idempotency_id,
                    "tenant_id": tenant_id,
                    "scope": scope,
                    "fingerprint": fingerprint,
                    "resource_id": values[0][1]["id"],
                    "created_at": now,
                    "updated_at": now,
                }
                self._insert(connection, "idempotency", link)
        return values[0][1], True

    def put(self, collection: str, record_id: str, record: dict[str, Any]) -> dict[str, Any]:
        self._validate_collection(collection)
        now = utc_now()
        value = {**record, "updated_at": now}
        with self._write_lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE records SET tenant_id=?,updated_at=?,payload=? WHERE collection=? AND id=?",
                (
                    value.get("tenant_id"),
                    now,
                    json.dumps(value, default=str, allow_nan=False),
                    collection,
                    record_id,
                ),
            )
            if cursor.rowcount == 0:
                raise NotFoundError(f"{collection.rstrip('s')} {record_id!r} was not found")
        return value

    def update(self, collection: str, record_id: str, **changes: Any) -> dict[str, Any]:
        with self._write_lock:
            record = self.get(collection, record_id)
            return self.put(collection, record_id, {**record, **changes})

    def get(self, collection: str, record_id: str) -> dict[str, Any]:
        self._validate_collection(collection)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT payload FROM records WHERE collection=? AND id=?",
                (collection, record_id),
            ).fetchone()
        if row is None:
            raise NotFoundError(f"{collection.rstrip('s')} {record_id!r} was not found")
        return json.loads(row[0])

    def list(self, collection: str, tenant_id: str | None = None) -> list[dict[str, Any]]:
        self._validate_collection(collection)
        query = "SELECT payload FROM records WHERE collection=?"
        params: tuple[Any, ...] = (collection,)
        if tenant_id is not None:
            query += " AND tenant_id=?"
            params = (collection, tenant_id)
        query += " ORDER BY created_at DESC"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [json.loads(row[0]) for row in rows]

    def owned(self, collection: str, record_id: str, tenant_id: str) -> dict[str, Any]:
        record = self.get(collection, record_id)
        if record.get("tenant_id") != tenant_id:
            raise NotFoundError(f"{collection.rstrip('s')} {record_id!r} was not found")
        return record

    def save_upload(self, dataset_id: str, filename: str, source: Any) -> tuple[Path, str]:
        suffix = Path(filename).suffix.lower()
        if suffix not in {".csv", ".parquet"}:
            raise ValueError("only CSV and Parquet uploads are supported")
        target = self.root / "uploads" / f"{dataset_id}{suffix}"
        size = 0
        digest = hashlib.sha256()
        try:
            with target.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    size += len(chunk)
                    if size > self.max_upload_bytes:
                        raise ValueError(f"upload exceeds {self.max_upload_bytes} byte limit")
                    output.write(chunk)
                    digest.update(chunk)
        except Exception:
            target.unlink(missing_ok=True)
            raise
        return target, digest.hexdigest()

    def path(self, kind: str, name: str) -> Path:
        if kind not in {"uploads", "artifacts", "predictions"}:
            raise ValueError(f"invalid artifact kind: {kind}")
        path = self.root / kind / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode=WAL;
                PRAGMA synchronous=NORMAL;
                PRAGMA foreign_keys=ON;
                CREATE TABLE IF NOT EXISTS records (
                    collection TEXT NOT NULL,
                    id TEXT NOT NULL,
                    tenant_id TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY(collection, id)
                );
                CREATE INDEX IF NOT EXISTS records_tenant_collection_created
                    ON records(tenant_id, collection, created_at DESC);
                """
            )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database, timeout=30)
        connection.execute("PRAGMA busy_timeout=30000")
        return connection

    @staticmethod
    def _insert(connection: sqlite3.Connection, collection: str, value: dict[str, Any]) -> None:
        connection.execute(
            "INSERT INTO records(collection,id,tenant_id,created_at,updated_at,payload) "
            "VALUES(?,?,?,?,?,?)",
            (
                collection,
                value["id"],
                value.get("tenant_id"),
                value["created_at"],
                value["updated_at"],
                json.dumps(value, default=str, allow_nan=False),
            ),
        )

    def _validate_collection(self, collection: str) -> None:
        if collection not in self.collections:
            raise ValueError(f"unknown collection: {collection}")
