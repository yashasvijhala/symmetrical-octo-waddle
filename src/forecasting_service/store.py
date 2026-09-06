import hashlib
import json
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from psycopg import Connection, sql
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

from forecasting_service.db_schema import COLLECTIONS, INDEXED_FIELDS


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class NotFoundError(KeyError):
    pass


class IdempotencyConflictError(ValueError):
    pass


class Store:
    """Pooled PostgreSQL metadata store."""

    collections = COLLECTIONS

    def __init__(
        self,
        database_url: str,
        pool_min_size: int = 1,
        pool_max_size: int = 10,
    ) -> None:
        if pool_min_size > pool_max_size:
            raise ValueError("database pool minimum cannot exceed maximum")
        self.pool = ConnectionPool(
            conninfo=database_url,
            min_size=pool_min_size,
            max_size=pool_max_size,
            kwargs={"connect_timeout": 5},
            open=True,
        )
        self.pool.wait(timeout=10)

    @staticmethod
    def new_id(prefix: str) -> str:
        return f"{prefix}_{uuid4().hex[:20]}"

    def create(self, collection: str, record: dict[str, Any]) -> dict[str, Any]:
        self._validate_collection(collection)
        now = utc_now()
        value = {**record, "created_at": now, "updated_at": now}
        with self._connect() as connection:
            self._insert(connection, collection, value)
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
        with self._connect() as connection:
            if idempotency_id is not None:
                connection.execute(
                    "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                    (idempotency_id,),
                )
                row = connection.execute(
                    "SELECT payload FROM idempotency WHERE id=%s", (idempotency_id,)
                ).fetchone()
                if row is not None:
                    link = self._decode(row[0])
                    if link.get("fingerprint") != fingerprint:
                        raise IdempotencyConflictError(
                            "Idempotency-Key was already used with a different request payload"
                        )
                    primary_collection = values[0][0]
                    existing = connection.execute(
                        sql.SQL("SELECT payload FROM {} WHERE id=%s").format(
                            sql.Identifier(primary_collection)
                        ),
                        (link["resource_id"],),
                    ).fetchone()
                    if existing is None:
                        raise RuntimeError("idempotency record points to a missing resource")
                    return self._decode(existing[0]), False
            for collection, value in values:
                self._insert(connection, collection, value)
            if idempotency_id is not None:
                self._insert(
                    connection,
                    "idempotency",
                    {
                        "id": idempotency_id,
                        "tenant_id": tenant_id,
                        "scope": scope,
                        "fingerprint": fingerprint,
                        "resource_id": values[0][1]["id"],
                        "created_at": now,
                        "updated_at": now,
                    },
                )
        return values[0][1], True

    def update(self, collection: str, record_id: str, **changes: Any) -> dict[str, Any]:
        self._validate_collection(collection)
        with self._connect() as connection:
            row = connection.execute(
                sql.SQL("SELECT payload FROM {} WHERE id=%s FOR UPDATE").format(
                    sql.Identifier(collection)
                ),
                (record_id,),
            ).fetchone()
            if row is None:
                raise NotFoundError(f"{collection.rstrip('s')} {record_id!r} was not found")
            value = {**self._decode(row[0]), **changes, "updated_at": utc_now()}
            self._put(connection, collection, record_id, value)
        return value

    def get(self, collection: str, record_id: str) -> dict[str, Any]:
        self._validate_collection(collection)
        with self._connect() as connection:
            row = connection.execute(
                sql.SQL("SELECT payload FROM {} WHERE id=%s").format(sql.Identifier(collection)),
                (record_id,),
            ).fetchone()
        if row is None:
            raise NotFoundError(f"{collection.rstrip('s')} {record_id!r} was not found")
        return self._decode(row[0])

    def list(
        self, collection: str, tenant_id: str | None = None, **filters: str
    ) -> list[dict[str, Any]]:
        self._validate_collection(collection)
        invalid_filters = set(filters) - set(INDEXED_FIELDS[collection])
        if invalid_filters:
            raise ValueError(f"unsupported {collection} filters: {sorted(invalid_filters)}")
        query = sql.SQL("SELECT payload FROM {}").format(sql.Identifier(collection))
        predicates = []
        params: list[Any] = []
        if tenant_id is not None:
            predicates.append(sql.SQL("tenant_id=%s"))
            params.append(tenant_id)
        for field, value in filters.items():
            predicates.append(sql.SQL("{}=%s").format(sql.Identifier(field)))
            params.append(value)
        if predicates:
            query += sql.SQL(" WHERE ") + sql.SQL(" AND ").join(predicates)
        query += sql.SQL(" ORDER BY created_at DESC")
        with self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return [self._decode(row[0]) for row in rows]

    def owned(self, collection: str, record_id: str, tenant_id: str) -> dict[str, Any]:
        self._validate_collection(collection)
        with self._connect() as connection:
            row = connection.execute(
                sql.SQL("SELECT payload FROM {} WHERE id=%s AND tenant_id=%s").format(
                    sql.Identifier(collection)
                ),
                (record_id, tenant_id),
            ).fetchone()
        if row is None:
            raise NotFoundError(f"{collection.rstrip('s')} {record_id!r} was not found")
        return self._decode(row[0])

    def ping(self) -> None:
        with self._connect() as connection:
            connection.execute("SELECT 1").fetchone()

    def close(self) -> None:
        self.pool.close()

    def _connect(self) -> AbstractContextManager[Connection[Any]]:
        return self.pool.connection()

    @staticmethod
    def _insert(connection: Connection[Any], collection: str, value: dict[str, Any]) -> None:
        field_names = ["id", "tenant_id", *INDEXED_FIELDS[collection], "created_at", "updated_at"]
        values = [value.get(field) for field in field_names]
        field_names.append("payload")
        values.append(Store._payload(value))
        connection.execute(
            sql.SQL("INSERT INTO {} ({}) VALUES ({})").format(
                sql.Identifier(collection),
                sql.SQL(", ").join(map(sql.Identifier, field_names)),
                sql.SQL(", ").join(sql.Placeholder() for _ in field_names),
            ),
            tuple(values),
        )

    @staticmethod
    def _put(
        connection: Connection[Any], collection: str, record_id: str, value: dict[str, Any]
    ) -> None:
        field_names = ["tenant_id", *INDEXED_FIELDS[collection], "updated_at"]
        values = [value.get(field) for field in field_names]
        field_names.append("payload")
        values.extend((Store._payload(value), record_id))
        assignments = sql.SQL(", ").join(
            sql.SQL("{}=%s").format(sql.Identifier(field)) for field in field_names
        )
        cursor = connection.execute(
            sql.SQL("UPDATE {} SET {} WHERE id=%s").format(sql.Identifier(collection), assignments),
            tuple(values),
        )
        if cursor.rowcount == 0:
            raise NotFoundError(f"{collection.rstrip('s')} {record_id!r} was not found")

    @staticmethod
    def _payload(value: dict[str, Any]) -> Jsonb:
        serialized = json.dumps(value, default=str, allow_nan=False)
        return Jsonb(json.loads(serialized))

    @staticmethod
    def _decode(value: Any) -> dict[str, Any]:
        return value if isinstance(value, dict) else json.loads(value)

    def _validate_collection(self, collection: str) -> None:
        if collection not in self.collections:
            raise ValueError(f"unknown collection: {collection}")
