from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
import pytest
from pydantic import ValidationError

from forecasting_service.config import R2_KEY_PREFIX, Settings
from forecasting_service.object_store import (
    ObjectRef,
    ObjectStore,
    create_object_store,
)


@dataclass
class FakeObject:
    body: bytes
    metadata: dict[str, str]
    etag: str


class FakeR2:
    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], FakeObject] = {}

    def upload_file(
        self,
        filename: str,
        bucket: str,
        key: str,
        ExtraArgs: dict[str, Any],
        Config: object,
    ) -> None:
        del Config
        self.objects[(bucket, key)] = FakeObject(
            body=Path(filename).read_bytes(),
            metadata=ExtraArgs.get("Metadata", {}),
            etag='"etag-1"',
        )

    def head_object(self, Bucket: str, Key: str) -> dict[str, Any]:
        obj = self.objects[(Bucket, Key)]
        return {
            "ContentLength": len(obj.body),
            "Metadata": obj.metadata,
            "ETag": obj.etag,
        }

    def download_file(self, bucket: str, key: str, filename: str, Config: object) -> None:
        del Config
        Path(filename).write_bytes(self.objects[(bucket, key)].body)

    def generate_presigned_url(self, operation: str, Params: dict[str, str], ExpiresIn: int) -> str:
        del operation, ExpiresIn
        return f"https://example.r2.cloudflarestorage.com/{Params['Key']}?signed=1"

    def head_bucket(self, Bucket: str) -> dict[str, str]:
        del Bucket
        return {}


def r2_settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        environment="test",
        state_dir=tmp_path,
        object_store_backend="r2",
        r2_endpoint="https://example.r2.cloudflarestorage.com",
        r2_bucket="forecasting-artifacts",
        r2_access_key_id="id",
        r2_secret_access_key="secret",
    )


def test_settings_reject_incomplete_r2_credentials() -> None:
    with pytest.raises(ValidationError, match="R2 object storage requires"):
        Settings(
            _env_file=None,
            object_store_backend="r2",
            r2_endpoint="https://example.r2.cloudflarestorage.com",
            r2_bucket="",
            r2_access_key_id="",
            r2_secret_access_key="",
        )


def test_settings_reject_insecure_r2_endpoint() -> None:
    with pytest.raises(ValidationError, match="HTTPS"):
        Settings(
            _env_file=None,
            object_store_backend="r2",
            r2_endpoint="http://example.r2.cloudflarestorage.com",
            r2_bucket="forecasting-artifacts",
            r2_access_key_id="id",
            r2_secret_access_key="secret",
        )


def test_local_object_store_roundtrip_and_integrity(tmp_path: Path) -> None:
    store = create_object_store(
        Settings(
            _env_file=None,
            environment="test",
            state_dir=tmp_path,
            object_store_backend="local",
        )
    )
    source = tmp_path / "upload.csv"
    source.write_text("sku,date,sales\na,2026-01-01,1\n", encoding="utf-8")
    key = store.key("tenants", "acme", "datasets", "ds_1", "source.csv")
    ref = store.put_file(key, source, "text/csv")

    assert ref.uri.startswith("local://")
    assert ref.size_bytes == source.stat().st_size
    materialized = store.materialize(ref)
    assert materialized.read_bytes() == source.read_bytes()
    assert store.download_url(ref, "source.csv") is None

    tampered = ObjectRef(
        uri=ref.uri,
        sha256="0" * 64,
        size_bytes=ref.size_bytes,
        content_type=ref.content_type,
    )
    with pytest.raises(OSError, match="integrity"):
        store.materialize(tampered)

    with pytest.raises(ValueError, match="escapes storage root"):
        store.materialize(
            ObjectRef(
                uri="local://../../etc/passwd",
                sha256="0" * 64,
                size_bytes=1,
                content_type="text/plain",
            )
        )


def test_object_key_rejects_parent_segments() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        ObjectStore.key("tenants", "..", "models")


def test_r2_object_store_put_cache_and_presign(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeR2()
    monkeypatch.setattr(boto3, "client", lambda *args, **kwargs: fake)
    store = create_object_store(r2_settings(tmp_path))
    source = tmp_path / "model.joblib"
    source.write_bytes(b"artifact")
    key = store.key("tenants", "acme", "models", "mdl_1", "model.joblib")
    ref = store.put_file(key, source, "application/octet-stream")

    assert ref.uri == (
        f"r2://forecasting-artifacts/{R2_KEY_PREFIX}/tenants/acme/models/mdl_1/model.joblib"
    )
    cached = store.materialize(ref)
    assert cached.read_bytes() == b"artifact"
    assert store.materialize(ref) == cached
    url = store.download_url(ref, 'report"\n.json')
    assert url is not None
    assert url.startswith("https://example.r2.cloudflarestorage.com/")
    store.ping()
