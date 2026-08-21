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

from pathfinder.cloud import cli as cloud_cli
from pathfinder.orchestration import build_episode_specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed episode specs onto a work queue")
    parser.add_argument("--count", type=int, default=8)
    cloud_cli.add_queue_args(parser)
    parser.add_argument("--route-length-m", type=float, default=400.0)
    parser.add_argument(
        "--suite-out", type=Path, default=None,
        help="Also write the enqueued specs as a JSON manifest — the record "
        "the run-report collector (python -m pathfinder.distributed_run "
        "--suite) reads, so the report describes the suite that was actually "
        "enqueued rather than a reconstruction from matching flags.",
    )
    args = parser.parse_args()

    queue = cloud_cli.queue_from_args(args)

    specs = build_episode_specs(count=args.count, route_length_m=args.route_length_m)
    for spec in specs:
        queue.send(spec.to_dict())

    # Only the SQS backend has a URL; the local queue lives in this process.
    destination = getattr(queue, "queue_url", "") or "(local, in-process)"
    print(f"enqueued {len(specs)} episodes onto {destination}")

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
