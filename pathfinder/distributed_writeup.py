"""
Render a distributed-run report artifact into its Markdown write-up.

Generated from the report JSON for the same reason as the other report
kinds': issue #17 puts hard requirements on the artifact — redeliveries
counted, the dead-letter queue's state, every result row attributed to a
worker, backend, and Policy — and a generated document keeps those
properties under test instead of under memory.

    python -m pathfinder.distributed_writeup results/distributed/report.json

regenerates the sibling ``.md`` from an existing report. The collector CLI
calls :func:`render_writeup` itself, so normally the two artifacts land
together.
"""
from __future__ import annotations

from pathfinder.reporting import (
    generated_from,
    infraction_table,
    regenerate_writeup_main,
    scope_banner,
    suite_section,
)

__all__ = ["render_writeup"]


def _coordinator_section(report: dict) -> list[str]:
    coordinator = report["coordinator"]
    lines = [
        f"- Run id: `{coordinator['run_id']}`",
        f"- Episodes: {coordinator['episodes_completed']} completed of "
        f"{coordinator['episodes_total'] or 'an unbounded run'}",
        f"- Duplicate submissions: {coordinator['duplicate_submissions']}",
        f"- Elapsed: {coordinator['elapsed_seconds']} s",
        "",
        "| Worker | Episodes completed | Healthy at collection |",
        "|---|---|---|",
    ]
    lines += [
        f"| {worker['worker_id']} | {worker['episodes_completed']} | "
        f"{'yes' if worker['healthy'] else 'no'} |"
        for worker in coordinator["workers"]
    ]
    lines += [
        "",
        "A worker unhealthy at collection stopped heartbeating — either it "
        "finished and exited, or it died. Its completed episodes are counted "
        "above either way; an episode it held when it died reappears in the "
        "redelivery count below, which is the queue doing its one real job.",
    ]
    return lines


def _queue_section(report: dict) -> list[str]:
    queue = report["queue"]
    lines = [
        f"- Redeliveries: {queue['redeliveries']} (delivery is at-least-once; "
        "a count above zero means an episode outlived the worker that first "
        "received it and was completed by another)",
        f"- Dead-lettered: {queue['dead_letters']}",
        f"- Approximate depth after the run: {queue['approximate_depth']}",
    ]
    for entry in queue["redelivered_episodes"]:
        lines.append(
            f"  - `{entry['episode_id']}` was delivered "
            f"{entry['receive_count']} times and completed by "
            f"`{entry['completed_by']}`"
        )
    if queue["dead_letter_episode_ids"]:
        lines += [
            "",
            "Dead-lettered episodes exhausted their delivery budget and are "
            "**not** in the aggregate: "
            + ", ".join(f"`{episode_id}`" for episode_id in queue["dead_letter_episode_ids"])
            + ".",
        ]
    return lines


def _telemetry_section(report: dict) -> list[str]:
    telemetry = report["telemetry"]
    if not telemetry["archived"]:
        return [f"Not archived — {telemetry['reason']}."]
    return [
        f"- Rows written: {telemetry['rows_written']}",
        f"- Parquet files: {telemetry['files_written']}",
        f"- Partitioning: {telemetry['partitioning']}",
    ]


def _summary_section(report: dict) -> list[str]:
    summary = report["summary"]
    if summary is None:
        return ["No results were collected, so there is nothing to aggregate."]
    rows = [
        ("Episodes", summary["episodes"]),
        ("Driving score", summary["driving_score"]),
        ("Route completion", summary["route_completion"]),
        ("Infraction penalty", summary["infraction_penalty"]),
        ("Collisions per km", summary["collisions_per_km"]),
        ("Failures", summary["failures"]),
    ]
    lines = ["| Metric | Value |", "|---|---|"]
    lines += [f"| {label} | {value} |" for label, value in rows]
    return lines


def _infraction_section(report: dict) -> list[str]:
    summary = report["summary"]
    return infraction_table(
        [("Count", summary["infraction_totals"] if summary else None)],
        empty_message="No scoreable or tracked infraction was committed.",
    )


def _per_episode_table(report: dict) -> list[str]:
    lines = [
        "| Episode | Worker | Backend | Policy | Deliveries | Driving score | Status |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {row['episode_id']} | {row['worker_id']} | "
        f"{row['simulator_backend']} | {row['model_version']} | "
        f"{row['receive_count']} | {row['driving_score']} | {row['status']} |"
        for row in report["results"]
    ]
    return lines


def _cross_check_section(report: dict) -> list[str]:
    lines: list[str] = []
    if report["episodes_missing_results"]:
        lines.append(
            "**Suite episodes with no result** (collected early, or work "
            "lost): "
            + ", ".join(f"`{episode_id}`" for episode_id in report["episodes_missing_results"])
            + "."
        )
    if report["results_not_in_suite"]:
        lines.append(
            "**Results outside the recorded suite** (the queue carried "
            "episodes these specs do not describe): "
            + ", ".join(f"`{episode_id}`" for episode_id in report["results_not_in_suite"])
            + "."
        )
    if report["failed_episodes"]:
        lines.append(
            "Episode(s) "
            + ", ".join(f"`{episode_id}`" for episode_id in report["failed_episodes"])
            + " failed mid-run; their scores measure the failure, not the "
            "policy."
        )
    if not lines:
        lines.append(
            "Every enqueued episode has exactly one result, and every result "
            "belongs to the recorded suite."
        )
    return lines


def render_writeup(report: dict, *, source: str) -> str:
    """Render one distributed-run report dict as a self-labelling Markdown
    write-up.

    Args:
        report: A ``build_distributed_report`` dict, exactly as serialised.
        source: Path of the report artifact, as the write-up should cite it.
    """
    lines = [
        f"# Distributed benchmark run — {report['backend']} backend",
        "",
        generated_from(
            report, source=source, generator="pathfinder/distributed_writeup.py"
        ),
        "",
        scope_banner(report),
        "",
        "## What this measures",
        "",
        "One benchmark run through the distributed pipeline: episodes seeded "
        "onto a work queue, pulled by workers that register and heartbeat "
        "over gRPC, scored, submitted to the coordinator, and their "
        "telemetry archived to partitioned Parquet. The mechanics — "
        "redelivery after a worker death, dead-lettering, provenance on "
        "every row — are the subject; the driving aggregate means only what "
        "the scope banner above allows.",
        "",
        *suite_section(report["episodes"], episodes_label="Episodes"),
        "",
        "## Coordinator",
        "",
        *_coordinator_section(report),
        "",
        "## Queue",
        "",
        *_queue_section(report),
        "",
        "## Telemetry warehouse",
        "",
        *_telemetry_section(report),
        "",
        "## Aggregate",
        "",
        *_summary_section(report),
        "",
        "## Infraction breakdown",
        "",
        *_infraction_section(report),
        "",
        "## Per-episode results",
        "",
        *_per_episode_table(report),
        "",
        "## Suite cross-check",
        "",
        *_cross_check_section(report),
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    return regenerate_writeup_main(
        argv,
        kind="distributed_run",
        render_writeup=render_writeup,
        description="Regenerate the Markdown write-up for a distributed-run report.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
