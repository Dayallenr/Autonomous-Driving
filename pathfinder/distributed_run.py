"""
The distributed-run report: one artifact for a queue-coordinated benchmark.

    python -m pathfinder.distributed_run --coordinator localhost:50051 ...

A distributed run scatters its evidence across four places: the coordinator
holds result rows in memory, the queue holds redelivery and dead-letter
state, the telemetry stream holds per-frame rows, and the warehouse holds
what was archived. This module is the collector that gathers all four into
one report JSON plus a generated write-up (issue #17) — the artifact a
rehearsal (#22) or the real run (#26) actually hands over at the end.

The collector talks to every component over the same interfaces a remote
process would use — coordinator over gRPC, queue over its backend API,
telemetry through the stream/warehouse pair — so the in-process test, the
LocalStack rehearsal, and the real-AWS run all exercise this exact code with
only endpoint configuration differing.

Redeliveries are counted from the results themselves: every result carries
the delivery count of the message it arrived on (``receive_count``, SQS's
``ApproximateReceiveCount``), observed by the worker that completed it. A
count above 1 is the crash-recovery path actually firing — a worker died
mid-episode and the queue's visibility timeout handed the episode to a
survivor. Counting queue-side instead would need a backend-specific query
that SQS cannot answer after the fact.

Unlike the sibling report kinds, the collector does not checkpoint per
episode: it runs for milliseconds after the suite finishes, and a crashed
collection is repaired by re-running it, not by replaying a partial.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path

from pathfinder import reporting
from pathfinder.metrics.driving_score import aggregate
from pathfinder.rpc.client import CoordinatorClient, episode_score_from_proto
from pathfinder.sim.base import EpisodeSpec

logger = logging.getLogger(__name__)

__all__ = [
    "KIND",
    "archive_telemetry",
    "build_distributed_report",
    "collect_report",
    "queue_statistics",
    "result_row",
]

KIND = "distributed_run"


def result_row(result) -> dict:
    """One report row from an ``EpisodeResult`` proto: the score plus the
    provenance the wire message carries — worker, backend, model and dataset
    version, and the queue delivery count. Every row answers "who produced
    this number, on what, from which delivery" without a join."""
    score = episode_score_from_proto(result)
    return {
        "worker_id": result.worker_id,
        "simulator_backend": result.simulator_backend,
        "model_version": result.model_version,
        "dataset_version": result.dataset_version,
        # proto3 reads an unset int as 0, which would mark a result from a
        # writer predating the field as a negative redelivery. Reading it as 1
        # (no redelivery observed) underclaims, which is the safe direction.
        "receive_count": result.receive_count or 1,
        **score.to_dict(),
    }


def queue_statistics(queue, rows: list[dict]) -> dict:
    """The queue's side of the story: redeliveries observed on completed
    results, what is still sitting on the queue, and what was dead-lettered.

    Args:
        queue: Any ``MessageQueue`` — read for depth and dead letters.
        rows: :func:`result_row` dicts, whose ``receive_count`` is where
            redeliveries are counted from (see the module docstring).
    """
    dead = queue.dead_letters()
    redelivered = [row for row in rows if row["receive_count"] > 1]
    return {
        "redeliveries": sum(row["receive_count"] - 1 for row in redelivered),
        "redelivered_episodes": [
            {
                "episode_id": row["episode_id"],
                "completed_by": row["worker_id"],
                "receive_count": row["receive_count"],
            }
            for row in redelivered
        ],
        "approximate_depth": queue.approximate_depth(),
        "dead_letters": len(dead),
        "dead_letter_episode_ids": [
            body.get("episode_id", "<unknown>") for body in dead
        ],
    }


def archive_telemetry(stream, warehouse) -> dict:
    """Drain the telemetry stream into partitioned Parquet and describe what
    landed, in the shape :func:`build_distributed_report` records."""
    keys = warehouse.drain_stream(stream)
    return {
        "archived": True,
        "rows_written": warehouse.rows_written,
        "files_written": warehouse.files_written,
        "parquet_files": keys,
        "partitioning": (
            f"hive-style {warehouse.prefix}/frames/event_date=YYYY-MM-DD/"
            "part-*.parquet, one file per date partition per drain"
        ),
    }


def _scope_for(backends: list[str]) -> tuple[str, str, str]:
    """Backend label plus scope for the observed result backends. Only a run
    whose every result came from CARLA earns the driving-quality label; no
    results and mixed backends each get their own pipeline-only sentence."""
    if not backends:
        return (
            "none",
            reporting.SCOPE_PIPELINE_ONLY,
            "No results were collected, so there is no driving to label; "
            "this artifact records the run mechanics only.",
        )
    if len(backends) > 1:
        joined = " + ".join(backends)
        return (
            joined,
            reporting.SCOPE_PIPELINE_ONLY,
            f"Results mix simulator backends ({joined}), whose scores are "
            "not comparable, so the aggregate is not a driving-quality "
            "number. These results verify the distributed pipeline end to "
            "end; the real measurement comes from a CARLA-only run.",
        )
    scope, note = reporting.scope(
        backends[0],
        driving_quality_note=(
            "Generated from CARLA-backend episodes: the aggregate measures "
            "the recorded policy's driving quality over the suite the queue "
            "carried."
        ),
        pipeline_limits="neither simulates real physics nor renders real scenes",
        pipeline_label=(
            "distributed benchmark pipeline — queue, workers, coordinator, "
            "telemetry warehouse"
        ),
    )
    return backends[0], scope, note


def build_distributed_report(
    *,
    specs: list[EpisodeSpec],
    status,
    results,
    duplicate_submissions: int,
    queue,
    telemetry: dict | None = None,
    suite_source: str = "caller-provided specs",
) -> dict:
    """Assemble the one report dict from the run's four evidence sources.

    Args:
        specs: The suite that was enqueued, for the report's episode list and
            for cross-checking that results and suite agree.
        suite_source: Where ``specs`` came from, recorded in the artifact.
            The id-level cross-check below cannot catch a reconstructed suite
            whose ids match but whose seeds or route lengths differ from what
            was actually enqueued, so the report must say whether its suite
            is the enqueuer's own record or a reconstruction.
        status: The coordinator's ``RunStatusResponse``.
        results: The coordinator's ``EpisodeResult`` protos.
        duplicate_submissions: The coordinator's duplicate-submission count —
            at-least-once delivery surfacing as a second result for an
            episode already recorded.
        queue: The work queue, read for :func:`queue_statistics`.
        telemetry: An :func:`archive_telemetry` dict, or ``None`` when no
            telemetry was collected — recorded as such rather than omitted.
    """
    rows = sorted((result_row(result) for result in results), key=lambda row: row["episode_id"])
    scores = [episode_score_from_proto(result) for result in results]

    backends = reporting.unique_in_order(
        [row["simulator_backend"] or "unknown" for row in rows]
    )
    backend, scope, scope_note = _scope_for(backends)

    spec_ids = {spec.episode_id for spec in specs}
    result_ids = {row["episode_id"] for row in rows}

    return {
        "kind": KIND,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "backend": backend,
        "scope": scope,
        "scope_note": scope_note,
        "coordinator": {
            "run_id": status.run_id,
            "episodes_total": status.episodes_total,
            "episodes_completed": status.episodes_completed,
            "episodes_in_flight": status.episodes_in_flight,
            "elapsed_seconds": round(status.elapsed_seconds, 2),
            "duplicate_submissions": duplicate_submissions,
            "workers": [
                {
                    "worker_id": worker.worker_id,
                    "episodes_completed": worker.episodes_completed,
                    "current_episode_id": worker.current_episode_id,
                    "seconds_since_heartbeat": round(worker.seconds_since_heartbeat, 2),
                    "healthy": worker.healthy,
                }
                for worker in status.workers
            ],
        },
        "queue": queue_statistics(queue, rows),
        "telemetry": (
            telemetry
            if telemetry is not None
            else {
                "archived": False,
                "reason": "no telemetry stream was configured for collection",
            }
        ),
        "episodes": [spec.to_dict() for spec in specs],
        "suite_source": suite_source,
        "results": rows,
        "summary": aggregate(scores).to_dict() if scores else None,
        "failed_episodes": sorted(
            score.episode_id for score in scores if score.status != "completed"
        ),
        # Both directions of the suite/results cross-check. Episodes still
        # missing mean the run was collected early or lost work; results
        # outside the suite mean the queue carried something this report's
        # specs do not describe. Either non-empty list is a red flag the
        # artifact states rather than hides.
        "episodes_missing_results": sorted(spec_ids - result_ids),
        "results_not_in_suite": sorted(result_ids - spec_ids),
    }


async def collect_report(
    *,
    coordinator_address: str,
    specs: list[EpisodeSpec],
    queue,
    telemetry: dict | None = None,
    suite_source: str = "caller-provided specs",
) -> dict:
    """Query the coordinator over gRPC and assemble the report.

    Args:
        coordinator_address: ``host:port`` of the live coordinator.
        specs: The suite that was enqueued.
        queue: The work queue, read for statistics.
        telemetry: An :func:`archive_telemetry` dict, or ``None``.
        suite_source: Where ``specs`` came from — see
            :func:`build_distributed_report`.
    """
    client = CoordinatorClient.connect(coordinator_address)
    try:
        # The coordinator serves a single run, so an empty run_id resolves to
        # it; passing one anyway would require the caller to learn it first.
        status = await client.get_run_status("")
        results = await client.get_run_results(status.run_id)
    finally:
        await client.close()
    return build_distributed_report(
        specs=specs,
        status=status,
        results=list(results.results),
        duplicate_submissions=results.duplicate_submissions,
        queue=queue,
        telemetry=telemetry,
        suite_source=suite_source,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Collect a distributed benchmark run into one report artifact: "
            "coordinator status and results over gRPC, queue statistics, and "
            "the telemetry archived to Parquet."
        )
    )
    parser.add_argument("--coordinator", type=str, required=True, help="host:port")

    parser.add_argument(
        "--suite", type=Path, default=None,
        help="the suite manifest enqueue_episodes.py --suite-out wrote — the "
        "enqueuer's own record of what went onto the queue. Prefer this over "
        "the reconstruction flags below: reconstruction can silently record "
        "wrong seeds or route lengths if its flags drift from the enqueuer's.",
    )
    parser.add_argument(
        "--episodes", type=int, default=8,
        help="suite size, matching what was enqueued (enqueue_episodes.py --count); "
        "ignored when --suite is given",
    )
    parser.add_argument(
        "--route-length-m", type=float, default=400.0,
        help="suite route length, matching what was enqueued; ignored when "
        "--suite is given",
    )

    parser.add_argument("--queue-backend", choices=["local", "sqs"], default="sqs")
    parser.add_argument("--queue-url", type=str, default="")
    parser.add_argument("--queue-name", type=str, default="pathfinder-episodes")
    parser.add_argument("--dead-letter-queue-url", type=str, default=None)
    parser.add_argument("--queue-endpoint-url", type=str, default=None, help="For LocalStack")
    parser.add_argument("--queue-region", type=str, default="us-east-1")

    parser.add_argument(
        "--telemetry-backend", choices=["none", "local", "kinesis"], default="none",
        help="Drain this stream into Parquet before reporting; 'none' records "
        "the report with telemetry marked not archived.",
    )
    parser.add_argument("--telemetry-stream-name", type=str, default="pathfinder-telemetry")
    parser.add_argument("--telemetry-local-root", type=str, default="./telemetry")
    parser.add_argument("--telemetry-endpoint-url", type=str, default=None, help="For LocalStack")
    parser.add_argument("--telemetry-region", type=str, default="us-east-1")

    parser.add_argument("--object-backend", choices=["local", "s3"], default="local")
    parser.add_argument("--bucket", type=str, default="pathfinder")
    parser.add_argument("--object-local-root", type=str, default="./s3")
    parser.add_argument("--object-endpoint-url", type=str, default=None, help="For LocalStack")
    parser.add_argument("--object-region", type=str, default="us-east-1")
    parser.add_argument("--warehouse-prefix", type=str, default="telemetry")

    parser.add_argument(
        "--output", type=Path, default=Path("results/distributed/report.json")
    )
    args = parser.parse_args(argv)

    # Imported here rather than at module top for the same reason as in the
    # sibling CLIs: the library-level builder has no business importing the
    # queue, stream, and store backends its arguments duck-type.
    from pathfinder.cloud.objects import build_store
    from pathfinder.cloud.queue import build_queue, ensure_queue
    from pathfinder.cloud.stream import build_stream
    from pathfinder.cloud.warehouse import TelemetryWarehouse
    from pathfinder.orchestration import build_episode_specs

    if args.suite is not None:
        import json

        specs = [
            EpisodeSpec.from_dict(entry)
            for entry in json.loads(args.suite.read_text(encoding="utf-8"))
        ]
        suite_source = f"suite manifest {args.suite}"
    else:
        specs = build_episode_specs(count=args.episodes, route_length_m=args.route_length_m)
        suite_source = (
            "reconstructed from CLI flags (--episodes/--route-length-m), not "
            "read from the enqueuer's record; seeds and route lengths are "
            "only as right as those flags"
        )

    queue_url = args.queue_url
    if args.queue_backend == "sqs" and not queue_url:
        queue_url = ensure_queue(
            args.queue_name, endpoint_url=args.queue_endpoint_url, region=args.queue_region
        )
    queue = build_queue(
        args.queue_backend,
        queue_url=queue_url,
        dead_letter_queue_url=args.dead_letter_queue_url,
        endpoint_url=args.queue_endpoint_url,
        region=args.queue_region,
    )

    telemetry = None
    if args.telemetry_backend != "none":
        stream = build_stream(
            args.telemetry_backend,
            local_root=args.telemetry_local_root,
            stream_name=args.telemetry_stream_name,
            endpoint_url=args.telemetry_endpoint_url,
            region=args.telemetry_region,
        )
        store = build_store(
            args.object_backend,
            local_root=args.object_local_root,
            bucket=args.bucket,
            endpoint_url=args.object_endpoint_url,
            region=args.object_region,
        )
        telemetry = archive_telemetry(stream, TelemetryWarehouse(store, prefix=args.warehouse_prefix))

    report = asyncio.run(
        collect_report(
            coordinator_address=args.coordinator,
            specs=specs,
            queue=queue,
            telemetry=telemetry,
            suite_source=suite_source,
        )
    )

    from pathfinder.distributed_writeup import render_writeup

    artifact = reporting.ReportArtifact(args.output, label_key="worker_id")
    writeup = artifact.finish(report, render_writeup)

    coordinator = report["coordinator"]
    print(
        f"run {coordinator['run_id']}: {coordinator['episodes_completed']}"
        f"/{coordinator['episodes_total'] or '?'} episodes from "
        f"{len(coordinator['workers'])} worker(s)"
    )
    queue_stats = report["queue"]
    print(
        f"queue: {queue_stats['redeliveries']} redelivery(ies), "
        f"{queue_stats['dead_letters']} dead-lettered, "
        f"depth {queue_stats['approximate_depth']}"
    )
    telemetry_stats = report["telemetry"]
    if telemetry_stats["archived"]:
        print(
            f"telemetry: {telemetry_stats['rows_written']} row(s) across "
            f"{telemetry_stats['files_written']} Parquet file(s)"
        )
    reporting.print_cli_tail(report, args.output, writeup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
