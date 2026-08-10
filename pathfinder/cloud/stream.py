"""
Telemetry stream: sensor and planner state out of the driving loop.

Kinesis-shaped, because telemetry is a *stream* problem rather than a queue
problem, and the difference matters:

* A queue delivers each message to exactly one consumer and then deletes it.
  That is right for work distribution (``cloud/queue.py``) and wrong for
  telemetry, where the same records must feed several independent consumers —
  the S3 archiver, a live dashboard, and an anomaly monitor — at their own paces.
* A stream retains records for a window and lets each consumer track its own
  position. Adding a consumer later does not require re-plumbing the producer.

Partition keys
--------------
Records are partitioned by ``episode_id``. Kinesis guarantees ordering *within*
a shard, and all records with the same partition key land on the same shard, so
one episode's telemetry stays strictly ordered. Ordering across episodes is
neither guaranteed nor wanted — episodes are independent, and forcing a global
order would serialize the whole pipeline onto one shard.

Batching
--------
``put_records`` batches up to 500 records per call because that is Kinesis's
limit and because a per-record round trip at 20 Hz per worker would spend more
time in HTTP than in simulation. The local backend batches identically so buffer
and flush behaviour is exercised before it ever reaches AWS.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


def _json_default(value):
    """Serialize types the telemetry rows legitimately contain.

    Telemetry carries timestamps, and ``json.dumps`` cannot encode ``datetime``.
    Without this every row fails to serialize — and because the producer treats
    telemetry failures as non-fatal, the result is a silently empty warehouse
    rather than a crash. Encoding as ISO-8601 keeps it round-trippable.
    """
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    raise TypeError(f"telemetry record contains non-serializable {type(value).__name__}")

logger = logging.getLogger(__name__)

__all__ = [
    "KinesisTelemetryStream",
    "LocalTelemetryStream",
    "StreamRecord",
    "TelemetryStream",
    "build_stream",
]

#: Kinesis rejects PutRecords calls above 500 records.
MAX_BATCH_RECORDS = 500

#: Kinesis rejects a single record above 1 MiB.
MAX_RECORD_BYTES = 1024 * 1024


@dataclass(frozen=True)
class StreamRecord:
    """One telemetry record with its ordering key."""

    partition_key: str
    data: dict
    sequence_number: str = ""


class TelemetryStream(ABC):
    """Stream interface shared by the local and Kinesis backends."""

    @abstractmethod
    def put(self, partition_key: str, data: dict) -> None:
        """Buffer one record. Flushes automatically when the batch fills."""

    @abstractmethod
    def flush(self) -> int:
        """Write buffered records. Returns how many were written."""

    @abstractmethod
    def read(self, *, partition_key: str | None = None, limit: int = 1000) -> list[StreamRecord]:
        """Read records back, optionally for one partition key."""

    def close(self) -> int:
        """Flush and release resources."""
        return self.flush()

    def __enter__(self) -> TelemetryStream:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class LocalTelemetryStream(TelemetryStream):
    """Append-only JSONL shards on disk, one file per partition key.

    The file-per-partition layout is what makes the ordering guarantee visible:
    a shard file *is* the ordered record sequence for that episode, so an
    ordering bug shows up as a mis-ordered file rather than as a subtle
    downstream discrepancy.
    """

    def __init__(self, root: Path | str, *, batch_size: int = MAX_BATCH_RECORDS) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self._buffer: list[StreamRecord] = []
        self._sequence = 0
        self._lock = threading.Lock()
        self.records_written = 0

    def put(self, partition_key: str, data: dict) -> None:
        encoded = json.dumps(data, default=_json_default).encode("utf-8")
        if len(encoded) > MAX_RECORD_BYTES:
            # Enforce the Kinesis limit locally so an oversized record fails
            # here, in a fast local run, rather than in the cloud.
            raise ValueError(
                f"record is {len(encoded)} bytes, over the {MAX_RECORD_BYTES}-byte limit"
            )
        with self._lock:
            self._sequence += 1
            self._buffer.append(
                StreamRecord(partition_key, data, sequence_number=str(self._sequence))
            )
            full = len(self._buffer) >= self.batch_size
        if full:
            self.flush()

    def flush(self) -> int:
        with self._lock:
            batch, self._buffer = self._buffer, []
        if not batch:
            return 0

        by_partition: dict[str, list[StreamRecord]] = defaultdict(list)
        for record in batch:
            by_partition[record.partition_key].append(record)

        for partition_key, records in by_partition.items():
            path = self.root / f"{_safe_name(partition_key)}.jsonl"
            with path.open("a", encoding="utf-8") as handle:
                for record in records:
                    handle.write(
                        json.dumps(
                            {
                                "partition_key": record.partition_key,
                                "sequence_number": record.sequence_number,
                                "data": record.data,
                            },
                            default=_json_default,
                        )
                        + "\n"
                    )
        self.records_written += len(batch)
        return len(batch)

    def read(self, *, partition_key: str | None = None, limit: int = 1000) -> list[StreamRecord]:
        self.flush()  # readers should see everything produced so far
        paths = (
            [self.root / f"{_safe_name(partition_key)}.jsonl"]
            if partition_key
            else sorted(self.root.glob("*.jsonl"))
        )
        out: list[StreamRecord] = []
        for path in paths:
            if not path.exists():
                continue
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if len(out) >= limit:
                        return out
                    payload = json.loads(line)
                    out.append(
                        StreamRecord(
                            payload["partition_key"],
                            payload["data"],
                            payload["sequence_number"],
                        )
                    )
        return out

    def partitions(self) -> list[str]:
        """Partition keys that have records — the local analogue of shards."""
        self.flush()
        return sorted(path.stem for path in self.root.glob("*.jsonl"))


def _safe_name(partition_key: str) -> str:
    """Make a partition key safe as a filename without losing distinctness."""
    return "".join(character if character.isalnum() or character in "-_" else "_"
                   for character in partition_key) or "default"


class KinesisTelemetryStream(TelemetryStream):
    """Amazon Kinesis Data Streams backend.

    Point ``endpoint_url`` at LocalStack to exercise this exact client against a
    real Kinesis API.
    """

    def __init__(
        self,
        stream_name: str,
        *,
        endpoint_url: str | None = None,
        region: str = "us-east-1",
        batch_size: int = MAX_BATCH_RECORDS,
    ) -> None:
        import boto3

        if not 1 <= batch_size <= MAX_BATCH_RECORDS:
            raise ValueError(f"batch_size must be 1..{MAX_BATCH_RECORDS}, got {batch_size}")
        self.stream_name = stream_name
        self.batch_size = batch_size
        self._client = boto3.client("kinesis", endpoint_url=endpoint_url, region_name=region)
        self._buffer: list[StreamRecord] = []
        self._lock = threading.Lock()
        self.records_written = 0

    def put(self, partition_key: str, data: dict) -> None:
        with self._lock:
            self._buffer.append(StreamRecord(partition_key, data))
            full = len(self._buffer) >= self.batch_size
        if full:
            self.flush()

    def flush(self) -> int:
        with self._lock:
            batch, self._buffer = self._buffer, []
        if not batch:
            return 0

        response = self._client.put_records(
            StreamName=self.stream_name,
            Records=[
                {
                    "PartitionKey": record.partition_key,
                    "Data": json.dumps(record.data, default=_json_default).encode("utf-8"),
                }
                for record in batch
            ],
        )
        # PutRecords is partially-failable: the call succeeds while individual
        # records fail (usually ProvisionedThroughputExceeded). Ignoring
        # FailedRecordCount silently drops telemetry, so retry the failures.
        failed = response.get("FailedRecordCount", 0)
        if failed:
            retry = [
                batch[index]
                for index, result in enumerate(response["Records"])
                if "ErrorCode" in result
            ]
            logger.warning("kinesis rejected %d record(s); retrying once", failed)
            time.sleep(0.2)  # brief backoff; throughput errors are transient
            self._client.put_records(
                StreamName=self.stream_name,
                Records=[
                    {
                        "PartitionKey": record.partition_key,
                        "Data": json.dumps(record.data, default=_json_default).encode("utf-8"),
                    }
                    for record in retry
                ],
            )
        self.records_written += len(batch)
        return len(batch)

    def read(self, *, partition_key: str | None = None, limit: int = 1000) -> list[StreamRecord]:
        self.flush()
        out: list[StreamRecord] = []
        shards = self._client.list_shards(StreamName=self.stream_name)["Shards"]
        for shard in shards:
            iterator = self._client.get_shard_iterator(
                StreamName=self.stream_name,
                ShardId=shard["ShardId"],
                ShardIteratorType="TRIM_HORIZON",
            )["ShardIterator"]
            response = self._client.get_records(ShardIterator=iterator, Limit=limit)
            for record in response.get("Records", []):
                if partition_key and record["PartitionKey"] != partition_key:
                    continue
                out.append(
                    StreamRecord(
                        record["PartitionKey"],
                        json.loads(record["Data"]),
                        record.get("SequenceNumber", ""),
                    )
                )
                if len(out) >= limit:
                    return out
        return out


def build_stream(
    backend: str,
    *,
    local_root: Path | str = "./telemetry",
    stream_name: str = "pathfinder-telemetry",
    endpoint_url: str | None = None,
    region: str = "us-east-1",
    batch_size: int = MAX_BATCH_RECORDS,
) -> TelemetryStream:
    """Construct the configured telemetry stream backend.

    Raises:
        ValueError: On an unknown backend name.
    """
    normalized = backend.strip().lower()
    if normalized == "local":
        return LocalTelemetryStream(local_root, batch_size=batch_size)
    if normalized == "kinesis":
        return KinesisTelemetryStream(
            stream_name, endpoint_url=endpoint_url, region=region, batch_size=batch_size
        )
    raise ValueError(f"unknown telemetry backend {backend!r}; expected local or kinesis")
