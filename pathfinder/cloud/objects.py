"""
Object storage with dataset versioning.

Two responsibilities, deliberately kept in one module because the second is
meaningless without the first:

1. A narrow ``put/get/list`` object store over the filesystem or S3.
2. **Dataset versioning** on top of it — the thing that makes a training run
   reproducible.

Why versioning is not optional
------------------------------
A training job that reads ``s3://bucket/kitti/`` is not reproducible. Someone
adds a thousand frames next month, the job is re-run to reproduce a number, and
it silently trains on different data. The model card says one thing and the
weights encode another, and there is no error anywhere.

So a dataset here is an immutable, content-addressed **manifest**: a list of
(key, size, digest) plus a version id derived from the digests. Writing the same
files twice produces the same version id; changing one byte produces a different
one. A training job records the version id it consumed, which makes the question
"what data produced these weights?" answerable rather than a matter of memory.

This is the same idea as a lockfile, applied to data.
"""
from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = [
    "DatasetManifest",
    "DatasetRegistry",
    "LocalObjectStore",
    "ObjectStore",
    "S3ObjectStore",
    "StoredObject",
    "build_store",
]


@dataclass(frozen=True)
class StoredObject:
    uri: str
    key: str
    size_bytes: int


class ObjectStore(ABC):
    """Minimal object surface. Narrow on purpose: every method has an obvious
    filesystem meaning, so the local backend cannot quietly drift from S3."""

    @abstractmethod
    def put(self, key: str, data: bytes) -> StoredObject: ...

    @abstractmethod
    def get(self, key: str) -> bytes: ...

    @abstractmethod
    def list(self, prefix: str) -> list[str]: ...

    @abstractmethod
    def uri_for(self, key: str) -> str: ...

    def put_file(self, key: str, path: Path | str) -> StoredObject:
        return self.put(key, Path(path).read_bytes())

    def exists(self, key: str) -> bool:
        try:
            self.get(key)
            return True
        except FileNotFoundError:
            return False


class LocalObjectStore(ObjectStore):
    """Filesystem store preserving S3 key layout."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        candidate = (self.root / key).resolve()
        if not candidate.is_relative_to(self.root):
            raise ValueError(f"key {key!r} escapes the store root")
        return candidate

    def put(self, key: str, data: bytes) -> StoredObject:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename: a reader must never observe a partial object.
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(path)
        return StoredObject(self.uri_for(key), key, len(data))

    def get(self, key: str) -> bytes:
        path = self._path(key)
        if not path.exists():
            raise FileNotFoundError(f"no object at {self.uri_for(key)}")
        return path.read_bytes()

    def list(self, prefix: str) -> list[str]:
        base = self._path(prefix)
        search_root = base if base.is_dir() else base.parent
        if not search_root.exists():
            return []
        # as_posix(), not str(): object-store keys are '/'-separated on every
        # platform, and str() yields backslashes on Windows. That made every
        # startswith(prefix) comparison fail against a '/'-separated prefix, so
        # list() returned nothing locally while the S3 backend returned the
        # objects — the two implementations of one interface disagreeing only on
        # Windows.
        return sorted(
            path.relative_to(self.root).as_posix()
            for path in search_root.rglob("*")
            if path.is_file() and path.relative_to(self.root).as_posix().startswith(prefix)
        )

    def uri_for(self, key: str) -> str:
        return f"file://{self._path(key)}"


class S3ObjectStore(ObjectStore):
    """S3 backend. ``endpoint_url`` points it at LocalStack or MinIO."""

    def __init__(
        self,
        bucket: str,
        *,
        endpoint_url: str | None = None,
        region: str = "us-east-1",
        create_bucket: bool = False,
    ) -> None:
        import boto3

        self.bucket = bucket
        self._client = boto3.client("s3", endpoint_url=endpoint_url, region_name=region)
        if create_bucket:
            self._ensure_bucket(region)

    def _ensure_bucket(self, region: str) -> None:
        from botocore.exceptions import ClientError

        try:
            self._client.head_bucket(Bucket=self.bucket)
        except ClientError:
            kwargs: dict = {"Bucket": self.bucket}
            # us-east-1 rejects an explicit LocationConstraint; others require one.
            if region != "us-east-1":
                kwargs["CreateBucketConfiguration"] = {"LocationConstraint": region}
            self._client.create_bucket(**kwargs)

    def put(self, key: str, data: bytes) -> StoredObject:
        self._client.put_object(Bucket=self.bucket, Key=key, Body=data)
        return StoredObject(self.uri_for(key), key, len(data))

    def get(self, key: str) -> bytes:
        from botocore.exceptions import ClientError

        try:
            return self._client.get_object(Bucket=self.bucket, Key=key)["Body"].read()
        except ClientError as error:
            if error.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                raise FileNotFoundError(f"no object at {self.uri_for(key)}") from error
            raise

    def list(self, prefix: str) -> list[str]:
        keys: list[str] = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            keys.extend(item["Key"] for item in page.get("Contents", []))
        return sorted(keys)

    def uri_for(self, key: str) -> str:
        return f"s3://{self.bucket}/{key}"


def build_store(
    backend: str,
    *,
    local_root: Path | str = "./s3",
    bucket: str = "pathfinder",
    endpoint_url: str | None = None,
    region: str = "us-east-1",
) -> ObjectStore:
    """Construct the configured object store.

    Raises:
        ValueError: On an unknown backend name.
    """
    normalized = backend.strip().lower()
    if normalized == "local":
        return LocalObjectStore(local_root)
    if normalized == "s3":
        return S3ObjectStore(
            bucket,
            endpoint_url=endpoint_url,
            region=region,
            create_bucket=endpoint_url is not None,
        )
    raise ValueError(f"unknown object store backend {backend!r}; expected local or s3")


# ─────────────────────────────────────────────────────────────────────────────
# Dataset versioning
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DatasetManifest:
    """An immutable, content-addressed description of one dataset version."""

    name: str
    version: str
    created_at: str
    #: (key, size_bytes, sha256) per file, sorted by key.
    entries: list[tuple[str, int, str]] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def total_bytes(self) -> int:
        return sum(size for _, size, _ in self.entries)

    @property
    def file_count(self) -> int:
        return len(self.entries)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str) -> DatasetManifest:
        payload = json.loads(raw)
        return cls(
            name=payload["name"],
            version=payload["version"],
            created_at=payload["created_at"],
            entries=[tuple(entry) for entry in payload["entries"]],
            metadata=payload.get("metadata", {}),
        )


class DatasetRegistry:
    """Publishes and resolves immutable dataset versions in an object store.

    Layout::

        datasets/<name>/versions/<version>/manifest.json
        datasets/<name>/versions/<version>/data/<original key>
        datasets/<name>/latest.json          -> {"version": "..."}

    ``latest.json`` is a mutable pointer, deliberately separate from the
    immutable versions. Training jobs should pin a version; only interactive
    exploration should follow ``latest``.
    """

    def __init__(self, store: ObjectStore, *, prefix: str = "datasets") -> None:
        self.store = store
        self.prefix = prefix.strip("/")

    def _version_prefix(self, name: str, version: str) -> str:
        return f"{self.prefix}/{name}/versions/{version}"

    def publish(
        self,
        name: str,
        files: dict[str, bytes],
        *,
        metadata: dict | None = None,
    ) -> DatasetManifest:
        """Publish files as a new immutable version.

        The version id is derived from the content, so publishing identical
        files twice yields the same id and does not create a duplicate. That
        property is what makes "did the data change?" a comparison rather than
        an investigation.

        Raises:
            ValueError: If ``files`` is empty.
        """
        if not files:
            raise ValueError("cannot publish an empty dataset")

        entries: list[tuple[str, int, str]] = []
        for key in sorted(files):
            data = files[key]
            entries.append((key, len(data), hashlib.sha256(data).hexdigest()))

        # Version id = hash of the (key, digest) list. Content-addressed, so it
        # is stable across machines and independent of upload order or clock.
        fingerprint = hashlib.sha256(
            "\n".join(f"{key}:{digest}" for key, _, digest in entries).encode("utf-8")
        ).hexdigest()[:16]

        manifest = DatasetManifest(
            name=name,
            version=fingerprint,
            created_at=datetime.now(UTC).isoformat(),
            entries=entries,
            metadata=metadata or {},
        )

        version_prefix = self._version_prefix(name, manifest.version)
        manifest_key = f"{version_prefix}/manifest.json"
        if self.store.exists(manifest_key):
            logger.info(
                "dataset %s version %s already published (identical content)",
                name, manifest.version,
            )
            return manifest

        for key in sorted(files):
            self.store.put(f"{version_prefix}/data/{key}", files[key])
        self.store.put(manifest_key, manifest.to_json().encode("utf-8"))
        self.store.put(
            f"{self.prefix}/{name}/latest.json",
            json.dumps({"version": manifest.version}).encode("utf-8"),
        )
        logger.info(
            "published dataset %s version %s (%d files, %d bytes)",
            name, manifest.version, manifest.file_count, manifest.total_bytes,
        )
        return manifest

    def resolve(self, name: str, version: str = "latest") -> DatasetManifest:
        """Load a manifest by version, or follow the ``latest`` pointer.

        Raises:
            FileNotFoundError: If the dataset or version does not exist.
        """
        if version == "latest":
            pointer = json.loads(
                self.store.get(f"{self.prefix}/{name}/latest.json").decode("utf-8")
            )
            version = pointer["version"]
        return DatasetManifest.from_json(
            self.store.get(f"{self._version_prefix(name, version)}/manifest.json").decode("utf-8")
        )

    def download(self, manifest: DatasetManifest, destination: Path | str) -> Path:
        """Materialize a version locally, verifying every digest.

        Verification is not paranoia: a truncated download produces a model that
        trains without error on corrupt data, and the only symptom is a metric
        that is quietly worse than it should be.

        Raises:
            ValueError: If any file's digest does not match the manifest.
        """
        root = Path(destination).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        version_prefix = self._version_prefix(manifest.name, manifest.version)

        for key, expected_size, expected_digest in manifest.entries:
            data = self.store.get(f"{version_prefix}/data/{key}")
            actual_digest = hashlib.sha256(data).hexdigest()
            if actual_digest != expected_digest or len(data) != expected_size:
                raise ValueError(
                    f"dataset {manifest.name}@{manifest.version} file {key!r} failed "
                    f"verification: expected sha256 {expected_digest} ({expected_size} bytes), "
                    f"got {actual_digest} ({len(data)} bytes)"
                )
            target = root / key
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        return root

    def list_versions(self, name: str) -> list[str]:
        keys = self.store.list(f"{self.prefix}/{name}/versions/")
        return sorted({key.split("/versions/")[1].split("/")[0] for key in keys})
