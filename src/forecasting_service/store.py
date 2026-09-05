import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class NotFoundError(KeyError):
    pass


class Store:
    """Small durable metadata and artifact store.

    JSON records make the local distribution zero-configuration. The service layer only relies on
    this interface, so PostgreSQL/S3 implementations can replace it without touching model code.
    """

    collections = ("datasets", "experiments", "jobs", "models", "forecasts", "actuals")

    def __init__(self, root: Path, max_upload_bytes: int = 2_147_483_648) -> None:
        self.root = root.resolve()
        self.max_upload_bytes = max_upload_bytes
        self._lock = threading.RLock()
        for name in (*self.collections, "uploads", "artifacts", "predictions"):
            (self.root / name).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def new_id(prefix: str) -> str:
        return f"{prefix}_{uuid4().hex[:20]}"

    def create(self, collection: str, record: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        value = {**record, "created_at": now, "updated_at": now}
        self.put(collection, value["id"], value)
        return value

    def put(self, collection: str, record_id: str, record: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            value = {**record, "updated_at": utc_now()}
            path = self._record_path(collection, record_id)
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
            temp.replace(path)
            return value

    def update(self, collection: str, record_id: str, **changes: Any) -> dict[str, Any]:
        with self._lock:
            record = self.get(collection, record_id)
            return self.put(collection, record_id, {**record, **changes})

    def get(self, collection: str, record_id: str) -> dict[str, Any]:
        path = self._record_path(collection, record_id)
        if not path.exists():
            raise NotFoundError(f"{collection.rstrip('s')} {record_id!r} was not found")
        return json.loads(path.read_text(encoding="utf-8"))

    def list(self, collection: str, tenant_id: str | None = None) -> list[dict[str, Any]]:
        records = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in self._dir(collection).glob("*.json")
        ]
        if tenant_id is not None:
            records = [record for record in records if record.get("tenant_id") == tenant_id]
        return sorted(records, key=lambda record: record.get("created_at", ""), reverse=True)

    def owned(self, collection: str, record_id: str, tenant_id: str) -> dict[str, Any]:
        record = self.get(collection, record_id)
        if record.get("tenant_id") != tenant_id:
            raise NotFoundError(f"{collection.rstrip('s')} {record_id!r} was not found")
        return record

    def save_upload(self, dataset_id: str, filename: str, source: Any) -> Path:
        suffix = Path(filename).suffix.lower()
        if suffix not in {".csv", ".parquet"}:
            raise ValueError("only CSV and Parquet uploads are supported")
        target = self.root / "uploads" / f"{dataset_id}{suffix}"
        size = 0
        try:
            with target.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    size += len(chunk)
                    if size > self.max_upload_bytes:
                        raise ValueError(f"upload exceeds {self.max_upload_bytes} byte limit")
                    output.write(chunk)
        except Exception:
            target.unlink(missing_ok=True)
            raise
        return target

    def path(self, kind: str, name: str) -> Path:
        path = self.root / kind / name
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _dir(self, collection: str) -> Path:
        if collection not in self.collections:
            raise ValueError(f"unknown collection: {collection}")
        return self.root / collection

    def _record_path(self, collection: str, record_id: str) -> Path:
        if not record_id.replace("_", "").isalnum():
            raise ValueError("invalid resource ID")
        return self._dir(collection) / f"{record_id}.json"
