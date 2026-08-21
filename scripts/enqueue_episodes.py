"""
Seed a benchmark suite onto a queue for distributed workers to pull from.

Separate from the workers themselves: enqueuing is a one-shot operation (build
N episode specs, push them, exit), while a worker is a long-running consumer.
Conflating the two would mean every worker that started first would race to
also seed the queue.

Usage (LocalStack, the docker-compose default):
    python scripts/enqueue_episodes.py --count 8 --queue-backend sqs \\
        --queue-name pathfinder-episodes --queue-endpoint-url http://localstack:4566
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pathfinder.cloud.queue import build_queue, ensure_queue
from pathfinder.orchestration import build_episode_specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed episode specs onto a work queue")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--queue-backend", choices=["local", "sqs"], default="sqs")
    parser.add_argument("--queue-url", type=str, default="")
    parser.add_argument("--queue-name", type=str, default="pathfinder-episodes")
    parser.add_argument("--queue-endpoint-url", type=str, default=None, help="For LocalStack")
    parser.add_argument("--queue-region", type=str, default="us-east-1")
    parser.add_argument("--route-length-m", type=float, default=400.0)
    parser.add_argument(
        "--suite-out", type=Path, default=None,
        help="Also write the enqueued specs as a JSON manifest — the record "
        "the run-report collector (python -m pathfinder.distributed_run "
        "--suite) reads, so the report describes the suite that was actually "
        "enqueued rather than a reconstruction from matching flags.",
    )
    args = parser.parse_args()

    queue_url = args.queue_url
    if args.queue_backend == "sqs" and not queue_url:
        queue_url = ensure_queue(
            args.queue_name, endpoint_url=args.queue_endpoint_url, region=args.queue_region
        )
        print(f"resolved queue {args.queue_name!r} to {queue_url}")

    queue = build_queue(
        args.queue_backend,
        queue_url=queue_url,
        endpoint_url=args.queue_endpoint_url,
        region=args.queue_region,
    )

    specs = build_episode_specs(count=args.count, route_length_m=args.route_length_m)
    for spec in specs:
        queue.send(spec.to_dict())

    print(f"enqueued {len(specs)} episodes onto {queue_url or '(local, in-process)'}")

    if args.suite_out is not None:
        import json

        args.suite_out.parent.mkdir(parents=True, exist_ok=True)
        args.suite_out.write_text(
            json.dumps([spec.to_dict() for spec in specs], indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"suite manifest written to {args.suite_out}")


if __name__ == "__main__":
    main()
