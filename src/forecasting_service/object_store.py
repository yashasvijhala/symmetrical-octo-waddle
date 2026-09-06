from __future__ import annotations

import hashlib
import os
import re
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import quote, urlparse
from uuid import uuid4

from forecasting_service.config import (
    OBJECT_CACHE_MAX_BYTES,
    OBJECT_PRESIGN_SECONDS,
    R2_KEY_PREFIX,
    R2_MAX_CONNECTIONS,
    R2_MULTIPART_CHUNK_BYTES,
    R2_MULTIPART_THRESHOLD_BYTES,
    R2_TRANSFER_CONCURRENCY,
    Settings,
)

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_IO_CHUNK = 1024 * 1024


@dataclass(frozen=True)
class ObjectRef:
    uri: str
    sha256: str
    size_bytes: int
    content_type: str
    etag: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ObjectRef:
        return cls(**value)


class ObjectStore(ABC):
    def __init__(self, cache_dir: Path, max_cache_bytes: int) -> None:
        self.cache_dir = cache_dir.resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_cache_bytes = max_cache_bytes

    @abstractmethod
    def put_file(self, key: str, source: Path, content_type: str) -> ObjectRef: ...

    @abstractmethod
    def materialize(self, ref: ObjectRef) -> Path: ...

    @abstractmethod
    def download_url(self, ref: ObjectRef, filename: str) -> str | None: ...

    @abstractmethod
    def ping(self) -> None: ...

    def put_stream(
        self,
        key: str,
        source: BinaryIO,
        content_type: str,
        max_bytes: int,
    ) -> ObjectRef:
        with self.temporary_path(Path(key).suffix) as staged:
            size = 0
            with staged.open("wb") as output:
                while chunk := source.read(_IO_CHUNK):
                    size += len(chunk)
                    if size > max_bytes:
                        raise ValueError(f"upload exceeds {max_bytes} byte limit")
                    output.write(chunk)
            return self.put_file(key, staged, content_type)

    def read_bytes(self, ref: ObjectRef) -> bytes:
        return self.materialize(ref).read_bytes()

    @contextmanager
    def temporary_path(self, suffix: str = "") -> Iterator[Path]:
        descriptor, name = tempfile.mkstemp(prefix="forecast-", suffix=suffix, dir=self.cache_dir)
        os.close(descriptor)
        path = Path(name)
        try:
            yield path
        finally:
            path.unlink(missing_ok=True)

    @staticmethod
    def key(*segments: str) -> str:
        cleaned = [quote(segment.strip(), safe="-_.") for segment in segments]
        if any(not segment or segment in {".", ".."} for segment in cleaned):
            raise ValueError("object key segments cannot be empty")
        return "/".join(cleaned)

    @staticmethod
    def _digest(path: Path) -> tuple[str, int]:
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as source:
            while chunk := source.read(_IO_CHUNK):
                size += len(chunk)
                digest.update(chunk)
        return digest.hexdigest(), size

    def _cache_path(self, sha256: str, suffix: str = "") -> Path:
        if not _SHA256.fullmatch(sha256):
            raise ValueError("invalid object SHA-256")
        return self.cache_dir / "objects" / sha256[:2] / f"{sha256}{suffix}"

    def _prune_cache(self, protected: Path) -> None:
        files: list[tuple[Path, int, int]] = []
        total = 0
        for path in (self.cache_dir / "objects").glob("*/*"):
            if not path.is_file():
                continue
            stat = path.stat()
            files.append((path, stat.st_size, stat.st_mtime_ns))
            total += stat.st_size
        if total <= self.max_cache_bytes:
            return
        for path, size, _ in sorted(files, key=lambda item: item[2]):
            if path == protected:
                continue
            path.unlink(missing_ok=True)
            total -= size
            if total <= self.max_cache_bytes:
                break


class LocalObjectStore(ObjectStore):
    def __init__(self, root: Path, cache_dir: Path, max_cache_bytes: int) -> None:
        super().__init__(cache_dir, max_cache_bytes)
        self.root = (root / "objects").resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def put_file(self, key: str, source: Path, content_type: str) -> ObjectRef:
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        try:
            sha256, size = _copy_digest(source, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return ObjectRef(
            uri=f"local://{key}",
            sha256=sha256,
            size_bytes=size,
            content_type=content_type,
            etag=sha256,
        )

    def materialize(self, ref: ObjectRef) -> Path:
        parsed = urlparse(ref.uri)
        if parsed.scheme != "local":
            raise ValueError(f"unsupported local object URI: {ref.uri}")
        path = self._path(f"{parsed.netloc}{parsed.path}")
        if not path.is_file():
            raise FileNotFoundError(ref.uri)
        if path.stat().st_size != ref.size_bytes:
            raise OSError(f"object integrity check failed: {ref.uri}")
        sha256, _ = self._digest(path)
        if sha256 != ref.sha256:
            raise OSError(f"object integrity check failed: {ref.uri}")
        return path

    def download_url(self, ref: ObjectRef, filename: str) -> str | None:
        del ref, filename
        return None

    def ping(self) -> None:
        if not os.access(self.root, os.R_OK | os.W_OK):
            raise OSError(f"object store is not readable and writable: {self.root}")

    def _path(self, key: str) -> Path:
        target = (self.root / key.lstrip("/")).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError("object key escapes storage root")
        return target


class R2ObjectStore(ObjectStore):
    def __init__(self, settings: Settings) -> None:
        super().__init__(settings.cache_dir(), OBJECT_CACHE_MAX_BYTES)
        import boto3
        from boto3.s3.transfer import TransferConfig
        from botocore.config import Config

        self.bucket = settings.r2_bucket
        self.prefix = R2_KEY_PREFIX.strip("/")
        self.presign_seconds = OBJECT_PRESIGN_SECONDS
        self.client = boto3.client(
            "s3",
            endpoint_url=settings.r2_endpoint,
            aws_access_key_id=settings.r2_access_key_id,
            aws_secret_access_key=settings.r2_secret_access_key,
            region_name="auto",
            config=Config(
                signature_version="s3v4",
                retries={"max_attempts": 5, "mode": "adaptive"},
                connect_timeout=5,
                read_timeout=120,
                max_pool_connections=R2_MAX_CONNECTIONS,
            ),
        )
        self.transfer = TransferConfig(
            multipart_threshold=R2_MULTIPART_THRESHOLD_BYTES,
            multipart_chunksize=R2_MULTIPART_CHUNK_BYTES,
            max_concurrency=R2_TRANSFER_CONCURRENCY,
            use_threads=True,
        )

    def put_file(self, key: str, source: Path, content_type: str) -> ObjectRef:
        sha256, size = self._digest(source)
        object_key = self._key(key)
        self.client.upload_file(
            str(source),
            self.bucket,
            object_key,
            ExtraArgs={"ContentType": content_type, "Metadata": {"sha256": sha256}},
            Config=self.transfer,
        )
        return ObjectRef(
            uri=f"r2://{self.bucket}/{object_key}",
            sha256=sha256,
            size_bytes=size,
            content_type=content_type,
            etag=sha256,
        )

    def materialize(self, ref: ObjectRef) -> Path:
        key = self._parse(ref.uri)
        suffix = Path(key).suffix
        target = self._cache_path(ref.sha256, suffix)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and target.stat().st_size == ref.size_bytes:
            target.touch()
            return target
        temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
        try:
            self.client.download_file(self.bucket, key, str(temporary), Config=self.transfer)
            sha256, size = self._digest(temporary)
            if sha256 != ref.sha256 or size != ref.size_bytes:
                raise OSError(f"R2 download integrity check failed for {key}")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        self._prune_cache(target)
        return target

    def download_url(self, ref: ObjectRef, filename: str) -> str | None:
        key = self._parse(ref.uri)
        safe_name = Path(filename).name.replace('"', "").replace("\r", "").replace("\n", "")
        return self.client.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": self.bucket,
                "Key": key,
                "ResponseContentDisposition": f'attachment; filename="{safe_name}"',
            },
            ExpiresIn=self.presign_seconds,
        )

    def ping(self) -> None:
        self.client.head_bucket(Bucket=self.bucket)

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}" if self.prefix else key

    def _parse(self, uri: str) -> str:
        parsed = urlparse(uri)
        if parsed.scheme != "r2" or parsed.netloc != self.bucket:
            raise ValueError(f"object URI is outside configured R2 bucket: {uri}")
        key = parsed.path.lstrip("/")
        if self.prefix and not key.startswith(f"{self.prefix}/"):
            raise ValueError(f"object URI is outside configured R2 prefix: {uri}")
        return key


def _copy_digest(source: Path, dest: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as src, dest.open("wb") as output:
        while chunk := src.read(_IO_CHUNK):
            size += len(chunk)
            digest.update(chunk)
            output.write(chunk)
    return digest.hexdigest(), size


def create_object_store(settings: Settings) -> ObjectStore:
    if settings.object_store_backend == "r2":
        return R2ObjectStore(settings)
    return LocalObjectStore(
        settings.state_dir,
        settings.cache_dir(),
        OBJECT_CACHE_MAX_BYTES,
    )
