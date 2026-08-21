"""
The one home (#35) for the queue/telemetry/object-store CLI flag groups.

Four CLIs configure the same cloud backends — scripts/run_worker.py,
scripts/enqueue_episodes.py, scripts/archive_telemetry.py, and
``python -m pathfinder.distributed_run`` — and before this module each
carried its own near-copy of the argparse blocks and the build-from-args
wiring (including the --queue-name -> ensure_queue fallback). A flag added
to one backend now lands in every CLI at once, and the #22 runbook's flag
names and defaults have exactly one place they could drift from.

What differs per CLI is confined to the keyword parameters: which backend a
CLI defaults to, whether telemetry is optional there, and the one help
sentence that describes what telemetry means in that CLI's context. Flag
names, the remaining defaults, and the builders are shared.
"""
from __future__ import annotations

import argparse
import logging

from pathfinder.cloud.objects import ObjectStore, build_store
from pathfinder.cloud.queue import MessageQueue, build_queue, ensure_queue
from pathfinder.cloud.stream import TelemetryStream, build_stream
from pathfinder.cloud.warehouse import TelemetryWarehouse

logger = logging.getLogger(__name__)

__all__ = [
    "add_object_store_args",
    "add_queue_args",
    "add_telemetry_args",
    "queue_from_args",
    "store_from_args",
    "stream_from_args",
    "warehouse_from_args",
]


def add_queue_args(
    parser: argparse.ArgumentParser,
    *,
    backend_default: str = "sqs",
    queue_name_default: str = "pathfinder-episodes",
) -> None:
    """Mount the work-queue flag group; :func:`queue_from_args` reads it back."""
    parser.add_argument("--queue-backend", choices=["local", "sqs"], default=backend_default)
    parser.add_argument(
        "--queue-url", type=str, default="",
        help="Explicit SQS queue URL. Alternative to --queue-name for a known/pre-provisioned queue.",
    )
    parser.add_argument(
        "--queue-name", type=str, default=queue_name_default,
        help="SQS queue name to create-or-resolve via --queue-endpoint-url (LocalStack convenience; "
        "real deployments should pass --queue-url to a Terraform-provisioned queue instead).",
    )
    parser.add_argument("--dead-letter-queue-url", type=str, default=None)
    parser.add_argument("--queue-endpoint-url", type=str, default=None, help="For LocalStack")
    parser.add_argument("--queue-region", type=str, default="us-east-1")


def queue_from_args(args: argparse.Namespace) -> MessageQueue:
    """Build the queue the parsed flags describe, resolving --queue-name to a
    URL via :func:`ensure_queue` when the SQS backend has no explicit
    --queue-url.

    Raises:
        SystemExit: When the SQS backend has neither a URL nor a name to
            resolve one from.
    """
    queue_url = args.queue_url
    if args.queue_backend == "sqs" and not queue_url:
        if not args.queue_name:
            raise SystemExit("--queue-backend sqs requires --queue-url or --queue-name")
        queue_url = ensure_queue(
            args.queue_name, endpoint_url=args.queue_endpoint_url, region=args.queue_region
        )
        logger.info("resolved queue %r to %s", args.queue_name, queue_url)
    return build_queue(
        args.queue_backend,
        queue_url=queue_url,
        dead_letter_queue_url=args.dead_letter_queue_url,
        endpoint_url=args.queue_endpoint_url,
        region=args.queue_region,
    )


def add_telemetry_args(
    parser: argparse.ArgumentParser,
    *,
    backend_default: str = "none",
    optional: bool = True,
    backend_help: str | None = None,
) -> None:
    """Mount the telemetry-stream flag group; :func:`stream_from_args` reads
    it back. ``optional=False`` drops the 'none' choice — for CLIs whose whole
    job is the telemetry stream, where skipping it is meaningless. The default
    ``backend_help`` follows ``optional``, so --help never advertises a 'none'
    value the parser would reject."""
    if backend_help is None:
        backend_help = (
            "Which telemetry stream backend to use; 'none' skips telemetry entirely."
            if optional
            else "Which telemetry stream backend to drain."
        )
    choices = (["none"] if optional else []) + ["local", "kinesis"]
    parser.add_argument(
        "--telemetry-backend", choices=choices, default=backend_default, help=backend_help
    )
    parser.add_argument("--telemetry-stream-name", type=str, default="pathfinder-telemetry")
    parser.add_argument("--telemetry-local-root", type=str, default="./telemetry")
    parser.add_argument("--telemetry-endpoint-url", type=str, default=None, help="For LocalStack")
    parser.add_argument("--telemetry-region", type=str, default="us-east-1")


def stream_from_args(args: argparse.Namespace) -> TelemetryStream | None:
    """Build the telemetry stream the parsed flags describe, or ``None`` for
    the 'none' backend."""
    if args.telemetry_backend == "none":
        return None
    return build_stream(
        args.telemetry_backend,
        local_root=args.telemetry_local_root,
        stream_name=args.telemetry_stream_name,
        endpoint_url=args.telemetry_endpoint_url,
        region=args.telemetry_region,
    )


def add_object_store_args(
    parser: argparse.ArgumentParser,
    *,
    backend_default: str,
) -> None:
    """Mount the object-store + warehouse flag group; :func:`store_from_args`
    and :func:`warehouse_from_args` read it back. ``backend_default`` is
    required because the existing CLIs genuinely disagree (the archiver
    defaults to s3, the run-report collector to local)."""
    parser.add_argument("--object-backend", choices=["local", "s3"], default=backend_default)
    parser.add_argument("--bucket", type=str, default="pathfinder")
    parser.add_argument("--object-local-root", type=str, default="./s3")
    parser.add_argument("--object-endpoint-url", type=str, default=None, help="For LocalStack")
    parser.add_argument("--object-region", type=str, default="us-east-1")
    parser.add_argument("--warehouse-prefix", type=str, default="telemetry")


def store_from_args(args: argparse.Namespace) -> ObjectStore:
    """Build the object store the parsed flags describe."""
    return build_store(
        args.object_backend,
        local_root=args.object_local_root,
        bucket=args.bucket,
        endpoint_url=args.object_endpoint_url,
        region=args.object_region,
    )


def warehouse_from_args(args: argparse.Namespace) -> TelemetryWarehouse:
    """Build a Parquet warehouse over :func:`store_from_args`'s store, rooted
    at --warehouse-prefix. The store is reachable as ``warehouse.store``."""
    return TelemetryWarehouse(store_from_args(args), prefix=args.warehouse_prefix)
