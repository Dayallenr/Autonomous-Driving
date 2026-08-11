"""
Work queue for distributing CARLA episodes across workers.

The interface is SQS-shaped because SQS is the deployment target, and the local
backend implements the parts of SQS semantics that the orchestration layer
actually depends on:

* **Visibility timeout.** A received message is hidden, not deleted. If the
  worker dies mid-episode the message reappears and another worker picks it up.
  This is the entire reason for using a queue instead of pre-sharding the work
  list: CARLA workers crash, and a static shard assignment silently loses that
  shard's episodes.
* **At-least-once delivery.** A message can be delivered twice — after a
  visibility timeout expires on a worker that was merely slow, not dead.
  Consumers must be idempotent. The benchmark runner keys results by episode id
  precisely so a duplicate overwrites rather than double-counts.
* **Dead-letter queue.** After ``max_receive_count`` deliveries a message is
  moved aside rather than retried forever. A CARLA episode that reliably
  segfaults the simulator would otherwise consume a worker indefinitely, and the
  whole run would never finish.

Getting these three right locally is what makes the local backend a real test of
the orchestration logic rather than a happy-path stub. A naive in-memory list
would pass every test and then lose work the first time a worker crashed.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)

__all__ = [
    "LocalMessageQueue",
    "Message",
    "MessageQueue",
    "SqsMessageQueue",
    "build_queue",
    "ensure_queue",
]


@dataclass(frozen=True)
class Message:
    """One unit of work in flight."""

    message_id: str
    body: dict
    #: Opaque token required to delete the message. Named to match SQS, where it
    #: changes on every receive — code that stores a message id and tries to
    #: delete with it later is a bug this naming makes visible.
    receipt_handle: str
    receive_count: int = 1


class MessageQueue(ABC):
    """Queue interface shared by the local and SQS backends."""

    @abstractmethod
    def send(self, body: dict) -> str:
        """Enqueue one message; returns its id."""

    @abstractmethod
    def receive(self, *, max_messages: int = 1, wait_seconds: float = 0.0) -> list[Message]:
        """Receive up to ``max_messages``, hiding them for the visibility timeout."""

    @abstractmethod
    def delete(self, receipt_handle: str) -> None:
        """Acknowledge a message so it is not redelivered."""

    @abstractmethod
    def approximate_depth(self) -> int:
        """Roughly how many messages are waiting. Approximate by nature."""

    @abstractmethod
    def dead_letters(self) -> list[dict]:
        """Bodies of messages that exhausted their delivery budget."""

    @abstractmethod
    def purge(self) -> None:
        """Drop everything. Used between test runs."""


@dataclass
class _Entry:
    message_id: str
    body: dict
    receive_count: int = 0
    #: Monotonic time before which the message is hidden. 0 means visible.
    invisible_until: float = 0.0
    receipt_handle: str = ""


class LocalMessageQueue(MessageQueue):
    """Thread-safe in-process queue with SQS delivery semantics.

    Thread-safe (unlike Sentinel's cache, which is single-loop by design)
    because the local orchestrator runs workers as threads to exercise genuine
    concurrent contention. A queue that only works single-threaded would not
    test the thing it exists to test.
    """

    def __init__(
        self,
        *,
        visibility_timeout: float = 30.0,
        max_receive_count: int = 3,
    ) -> None:
        if visibility_timeout <= 0:
            raise ValueError(f"visibility_timeout must be positive, got {visibility_timeout}")
        if max_receive_count < 1:
            raise ValueError(f"max_receive_count must be >= 1, got {max_receive_count}")
        self.visibility_timeout = visibility_timeout
        self.max_receive_count = max_receive_count
        self._entries: list[_Entry] = []
        self._dead: list[dict] = []
        self._lock = threading.Lock()

    def send(self, body: dict) -> str:
        message_id = uuid.uuid4().hex
        with self._lock:
            self._entries.append(_Entry(message_id=message_id, body=body))
        return message_id

    def receive(self, *, max_messages: int = 1, wait_seconds: float = 0.0) -> list[Message]:
        deadline = time.monotonic() + wait_seconds
        while True:
            with self._lock:
                found = self._take_visible(max_messages)
            if found or time.monotonic() >= deadline:
                return found
            # Long-poll: sleep briefly rather than spin. Real SQS long-polling
            # blocks server-side; this approximates it without burning CPU.
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    def _take_visible(self, max_messages: int) -> list[Message]:
        """Caller must hold the lock."""
        now = time.monotonic()
        taken: list[Message] = []
        expired: list[_Entry] = []

        for entry in self._entries:
            if len(taken) >= max_messages:
                break
            if entry.invisible_until > now:
                continue

            entry.receive_count += 1
            if entry.receive_count > self.max_receive_count:
                # Budget exhausted: this message is poison. Move it aside so a
                # single bad episode cannot stall the run forever.
                expired.append(entry)
                continue

            entry.invisible_until = now + self.visibility_timeout
            entry.receipt_handle = uuid.uuid4().hex
            taken.append(
                Message(
                    message_id=entry.message_id,
                    body=entry.body,
                    receipt_handle=entry.receipt_handle,
                    receive_count=entry.receive_count,
                )
            )

        for entry in expired:
            self._entries.remove(entry)
            self._dead.append(entry.body)
            logger.warning(
                "message %s exceeded %d deliveries; moved to dead-letter queue",
                entry.message_id,
                self.max_receive_count,
            )
        return taken

    def delete(self, receipt_handle: str) -> None:
        with self._lock:
            for entry in self._entries:
                if entry.receipt_handle == receipt_handle:
                    self._entries.remove(entry)
                    return
        # Deleting an unknown handle is not an error in SQS either — it happens
        # when a worker finishes after its visibility timeout expired and the
        # message was already redelivered and acknowledged by someone else.
        logger.debug("delete for unknown receipt handle %s (already redelivered?)", receipt_handle)

    def approximate_depth(self) -> int:
        now = time.monotonic()
        with self._lock:
            return sum(1 for entry in self._entries if entry.invisible_until <= now)

    def in_flight(self) -> int:
        """Messages currently hidden by a visibility timeout."""
        now = time.monotonic()
        with self._lock:
            return sum(1 for entry in self._entries if entry.invisible_until > now)

    def dead_letters(self) -> list[dict]:
        with self._lock:
            return list(self._dead)

    def purge(self) -> None:
        with self._lock:
            self._entries.clear()
            self._dead.clear()


class SqsMessageQueue(MessageQueue):
    """Amazon SQS backend.

    Point ``endpoint_url`` at LocalStack to run this exact code against a real
    SQS API. Redrive (the dead-letter policy) is configured on the queue itself
    in AWS, so ``dead_letters`` reads the configured DLQ rather than tracking
    state locally.
    """

    def __init__(
        self,
        queue_url: str,
        *,
        dead_letter_queue_url: str | None = None,
        endpoint_url: str | None = None,
        region: str = "us-east-1",
        visibility_timeout: int = 30,
    ) -> None:
        import boto3

        self.queue_url = queue_url
        self.dead_letter_queue_url = dead_letter_queue_url
        self.visibility_timeout = visibility_timeout
        self._client = boto3.client("sqs", endpoint_url=endpoint_url, region_name=region)

    def send(self, body: dict) -> str:
        response = self._client.send_message(
            QueueUrl=self.queue_url, MessageBody=json.dumps(body)
        )
        return response["MessageId"]

    def receive(self, *, max_messages: int = 1, wait_seconds: float = 0.0) -> list[Message]:
        response = self._client.receive_message(
            QueueUrl=self.queue_url,
            # SQS caps a single ReceiveMessage at 10.
            MaxNumberOfMessages=min(max_messages, 10),
            WaitTimeSeconds=int(min(wait_seconds, 20)),  # SQS caps long-poll at 20s
            VisibilityTimeout=self.visibility_timeout,
            AttributeNames=["ApproximateReceiveCount"],
        )
        return [
            Message(
                message_id=item["MessageId"],
                body=json.loads(item["Body"]),
                receipt_handle=item["ReceiptHandle"],
                receive_count=int(item.get("Attributes", {}).get("ApproximateReceiveCount", 1)),
            )
            for item in response.get("Messages", [])
        ]

    def delete(self, receipt_handle: str) -> None:
        self._client.delete_message(QueueUrl=self.queue_url, ReceiptHandle=receipt_handle)

    def approximate_depth(self) -> int:
        response = self._client.get_queue_attributes(
            QueueUrl=self.queue_url, AttributeNames=["ApproximateNumberOfMessages"]
        )
        return int(response["Attributes"]["ApproximateNumberOfMessages"])

    def dead_letters(self) -> list[dict]:
        if not self.dead_letter_queue_url:
            return []
        bodies: list[dict] = []
        while True:
            response = self._client.receive_message(
                QueueUrl=self.dead_letter_queue_url, MaxNumberOfMessages=10
            )
            messages = response.get("Messages", [])
            if not messages:
                return bodies
            bodies.extend(json.loads(item["Body"]) for item in messages)

    def purge(self) -> None:
        self._client.purge_queue(QueueUrl=self.queue_url)


def ensure_queue(name: str, *, endpoint_url: str | None = None, region: str = "us-east-1") -> str:
    """Create the named SQS queue if it doesn't exist yet, and return its URL.

    ``create_queue`` is idempotent — called again on a queue that already
    exists (with the same attributes) it just returns that queue's URL. Meant
    for LocalStack-based local/CI use, where nothing provisions the queue
    ahead of time; real deployments provision queues via Terraform and pass an
    explicit ``queue_url`` to :func:`build_queue` instead.
    """
    import boto3

    client = boto3.client("sqs", endpoint_url=endpoint_url, region_name=region)
    return client.create_queue(QueueName=name)["QueueUrl"]


def build_queue(
    backend: str,
    *,
    queue_url: str = "",
    dead_letter_queue_url: str | None = None,
    endpoint_url: str | None = None,
    region: str = "us-east-1",
    visibility_timeout: float = 30.0,
    max_receive_count: int = 3,
) -> MessageQueue:
    """Construct the configured queue backend.

    Raises:
        ValueError: On an unknown backend, or an sqs backend with no queue URL.
    """
    normalized = backend.strip().lower()
    if normalized == "local":
        return LocalMessageQueue(
            visibility_timeout=visibility_timeout, max_receive_count=max_receive_count
        )
    if normalized == "sqs":
        if not queue_url:
            raise ValueError("sqs backend requires queue_url")
        return SqsMessageQueue(
            queue_url,
            dead_letter_queue_url=dead_letter_queue_url,
            endpoint_url=endpoint_url,
            region=region,
            visibility_timeout=int(visibility_timeout),
        )
    raise ValueError(f"unknown queue backend {backend!r}; expected local or sqs")
