"""
Tests for the AWS-SDK-backed cloud classes (SqsMessageQueue, S3ObjectStore,
KinesisTelemetryStream), against moto's mocked AWS rather than real AWS.

The audit that kicked off this build-out found these classes had *never* been
exercised — not against real AWS, not against LocalStack, not even mocked.
The local backends' semantics were well tested, but the actual boto3 code
paths (request/response shapes, error translation, pagination) were
unverified. moto intercepts botocore at the HTTP layer, so this runs the real
boto3 client code against a real (simulated) SQS/S3/Kinesis API — closer to
the truth than a hand-rolled mock, and free/instant/credential-free, so it's
safe to run in CI on every commit rather than only in the LocalStack/real-AWS
paths that need a running container or a network.
"""
from __future__ import annotations

import boto3
import pytest
from moto import mock_aws

from pathfinder.cloud.objects import DatasetRegistry, S3ObjectStore
from pathfinder.cloud.queue import SqsMessageQueue, ensure_queue
from pathfinder.cloud.stream import KinesisTelemetryStream, StreamRecord

REGION = "us-east-1"


@pytest.fixture
def aws():
    with mock_aws():
        yield


# ─────────────────────────────────────────────────────────────────────────────
# SqsMessageQueue
# ─────────────────────────────────────────────────────────────────────────────


def _create_sqs_queue(name: str) -> str:
    client = boto3.client("sqs", region_name=REGION)
    return client.create_queue(QueueName=name)["QueueUrl"]


def test_sqs_send_receive_delete_round_trip(aws):
    queue_url = _create_sqs_queue("pathfinder-test")
    queue = SqsMessageQueue(queue_url, region=REGION)

    queue.send({"episode_id": "ep-0001", "seed": 7})
    [message] = queue.receive(max_messages=1, wait_seconds=0)

    assert message.body == {"episode_id": "ep-0001", "seed": 7}
    assert message.receive_count == 1

    queue.delete(message.receipt_handle)
    assert queue.receive(max_messages=1, wait_seconds=0) == []


def test_sqs_visibility_timeout_hides_in_flight_messages(aws):
    queue_url = _create_sqs_queue("pathfinder-visibility")
    # A long visibility timeout so the message stays hidden for the duration
    # of this test regardless of how long moto takes to respond.
    queue = SqsMessageQueue(queue_url, region=REGION, visibility_timeout=60)

    queue.send({"episode_id": "ep-0001"})
    [message] = queue.receive(max_messages=1, wait_seconds=0)
    assert message is not None

    # Received but not deleted: a second receive must not see it again. This
    # is the real SQS semantic the orchestration layer's crash-recovery
    # depends on — a naive queue would hand the same message to two workers.
    assert queue.receive(max_messages=1, wait_seconds=0) == []


def test_sqs_approximate_depth_excludes_in_flight_messages(aws):
    queue_url = _create_sqs_queue("pathfinder-depth")
    queue = SqsMessageQueue(queue_url, region=REGION, visibility_timeout=60)

    queue.send({"episode_id": "ep-0001"})
    queue.send({"episode_id": "ep-0002"})
    assert queue.approximate_depth() == 2

    queue.receive(max_messages=1, wait_seconds=0)
    # SQS's ApproximateNumberOfMessages counts only visible messages; the
    # in-flight one should not be double-counted as still waiting.
    assert queue.approximate_depth() == 1


def test_sqs_dead_letters_reads_the_configured_dlq(aws):
    dlq_url = _create_sqs_queue("pathfinder-dlq")
    dlq_client = boto3.client("sqs", region_name=REGION)
    dlq_client.send_message(QueueUrl=dlq_url, MessageBody='{"episode_id": "poison"}')

    queue_url = _create_sqs_queue("pathfinder-main")
    queue = SqsMessageQueue(queue_url, region=REGION, dead_letter_queue_url=dlq_url)

    assert queue.dead_letters() == [{"episode_id": "poison"}]


def test_sqs_dead_letters_empty_when_no_dlq_configured(aws):
    queue_url = _create_sqs_queue("pathfinder-no-dlq")
    queue = SqsMessageQueue(queue_url, region=REGION)
    assert queue.dead_letters() == []


def test_ensure_queue_is_idempotent(aws):
    first_url = ensure_queue("pathfinder-ensure-test", region=REGION)
    second_url = ensure_queue("pathfinder-ensure-test", region=REGION)
    assert first_url == second_url

    queue = SqsMessageQueue(first_url, region=REGION)
    queue.send({"episode_id": "ep-0001"})
    assert queue.approximate_depth() == 1


# ─────────────────────────────────────────────────────────────────────────────
# S3ObjectStore
# ─────────────────────────────────────────────────────────────────────────────


def test_s3_put_get_round_trip(aws):
    store = S3ObjectStore("pathfinder-test-bucket", region=REGION, create_bucket=True)

    stored = store.put("runs/ep-0001/telemetry.json", b'{"frames": 100}')
    assert stored.uri == "s3://pathfinder-test-bucket/runs/ep-0001/telemetry.json"
    assert store.get("runs/ep-0001/telemetry.json") == b'{"frames": 100}'


def test_s3_get_missing_key_raises_file_not_found(aws):
    store = S3ObjectStore("pathfinder-test-bucket", region=REGION, create_bucket=True)
    with pytest.raises(FileNotFoundError):
        store.get("does/not/exist.json")


def test_s3_list_returns_sorted_keys_under_prefix(aws):
    store = S3ObjectStore("pathfinder-test-bucket", region=REGION, create_bucket=True)
    store.put("kitti/b.txt", b"b")
    store.put("kitti/a.txt", b"a")
    store.put("other/c.txt", b"c")

    assert store.list("kitti/") == ["kitti/a.txt", "kitti/b.txt"]


def test_s3_exists_uses_get_and_swallows_missing(aws):
    store = S3ObjectStore("pathfinder-test-bucket", region=REGION, create_bucket=True)
    assert store.exists("nope.txt") is False
    store.put("nope.txt", b"now it does")
    assert store.exists("nope.txt") is True


def test_dataset_registry_publish_resolve_download_against_real_s3_api(aws, tmp_path):
    store = S3ObjectStore("pathfinder-datasets", region=REGION, create_bucket=True)
    registry = DatasetRegistry(store)

    manifest = registry.publish(
        "kitti-mini", {"images/000001.png": b"fake-image-bytes", "labels/000001.txt": b"car 0 0"}
    )
    resolved = registry.resolve("kitti-mini")  # follows the latest.json pointer
    assert resolved.version == manifest.version

    destination = registry.download(resolved, tmp_path / "out")
    assert (destination / "images/000001.png").read_bytes() == b"fake-image-bytes"
    assert (destination / "labels/000001.txt").read_bytes() == b"car 0 0"


# ─────────────────────────────────────────────────────────────────────────────
# KinesisTelemetryStream
# ─────────────────────────────────────────────────────────────────────────────


def _create_kinesis_stream(name: str) -> None:
    client = boto3.client("kinesis", region_name=REGION)
    client.create_stream(StreamName=name, ShardCount=1)
    client.get_waiter("stream_exists").wait(StreamName=name)


def test_kinesis_put_flush_read_round_trip(aws):
    _create_kinesis_stream("pathfinder-telemetry-test")
    stream = KinesisTelemetryStream("pathfinder-telemetry-test", region=REGION, batch_size=10)

    stream.put("ep-0001", {"frame_index": 0, "speed_mps": 5.0})
    stream.put("ep-0001", {"frame_index": 1, "speed_mps": 5.2})
    stream.put("ep-0002", {"frame_index": 0, "speed_mps": 3.1})
    written = stream.flush()

    assert written == 3

    ep1_records = stream.read(partition_key="ep-0001")
    assert sorted(r.data["frame_index"] for r in ep1_records) == [0, 1]
    assert all(isinstance(r, StreamRecord) for r in ep1_records)

    all_records = stream.read()
    assert len(all_records) == 3


def test_kinesis_put_autoflushes_at_batch_size(aws):
    _create_kinesis_stream("pathfinder-telemetry-autoflush")
    stream = KinesisTelemetryStream("pathfinder-telemetry-autoflush", region=REGION, batch_size=2)

    stream.put("ep-0001", {"frame_index": 0})
    assert stream.records_written == 0  # buffered, not yet flushed
    stream.put("ep-0001", {"frame_index": 1})
    assert stream.records_written == 2  # batch_size reached -> auto-flushed

    assert len(stream.read()) == 2
