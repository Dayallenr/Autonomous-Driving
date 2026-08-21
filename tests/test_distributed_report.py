"""
Unit tests for the distributed-run report builder and its write-up: issue
#17 requires the artifact to carry the redelivery count, the dead-letter
queue's state, and worker/backend/Policy provenance on every result row —
so those are what get pinned here, on fabricated protos with no network.
The end-to-end chaos test lives in test_distributed_e2e.py.
"""
from __future__ import annotations

import time

import pytest

from pathfinder.cloud.queue import LocalMessageQueue
from pathfinder.distributed_run import (
    build_distributed_report,
    queue_statistics,
    result_row,
)
from pathfinder.distributed_writeup import render_writeup
from pathfinder.metrics.driving_score import EpisodeScore
from pathfinder.reporting import SCOPE_DRIVING_QUALITY, SCOPE_PIPELINE_ONLY
from pathfinder.rpc.client import episode_score_to_proto
from pathfinder.rpc.generated import pathfinder_pb2
from pathfinder.sim.base import EpisodeSpec


def make_spec(index: int) -> EpisodeSpec:
    return EpisodeSpec(
        episode_id=f"ep-{index:04d}", route_length_m=80.0, max_steps=50, seed=100 + index
    )


def make_result(
    episode_id: str,
    *,
    worker_id: str = "worker-a",
    backend: str = "kinematic",
    model_version: str = "pure_pursuit",
    receive_count: int = 1,
    status: str = "completed",
) -> pathfinder_pb2.EpisodeResult:
    score = EpisodeScore(
        episode_id=episode_id,
        route_completion=0.5,
        infraction_penalty=1.0,
        driving_score=50.0,
        infractions={"lane_invasion": 1},
        distance_travelled_m=40.0,
        route_length_m=80.0,
        frames=50,
        duration_seconds=2.0,
        status=status,
    )
    return episode_score_to_proto(
        score,
        worker_id=worker_id,
        model_version=model_version,
        simulator_backend=backend,
        receive_count=receive_count,
    )


def make_status(*, run_id: str = "run-1", episodes_total: int = 2, completed: int = 2):
    return pathfinder_pb2.RunStatusResponse(
        run_id=run_id,
        episodes_total=episodes_total,
        episodes_completed=completed,
        episodes_in_flight=0,
        elapsed_seconds=3.5,
        workers=[
            pathfinder_pb2.WorkerStatus(
                worker_id="worker-a", episodes_completed=completed,
                seconds_since_heartbeat=0.1, healthy=True,
            ),
            pathfinder_pb2.WorkerStatus(
                worker_id="worker-chaos", episodes_completed=0,
                seconds_since_heartbeat=9.0, healthy=False,
            ),
        ],
    )


def make_report(**overrides) -> dict:
    defaults = dict(
        specs=[make_spec(0), make_spec(1)],
        status=make_status(),
        results=[
            make_result("ep-0000", receive_count=2),
            make_result("ep-0001"),
        ],
        duplicate_submissions=0,
        queue=LocalMessageQueue(),
        telemetry={
            "archived": True,
            "rows_written": 100,
            "files_written": 1,
            "parquet_files": ["telemetry/frames/event_date=2026-08-21/part-x.parquet"],
            "partitioning": "hive-style telemetry/frames/event_date=YYYY-MM-DD/part-*.parquet",
        },
    )
    defaults.update(overrides)
    return build_distributed_report(**defaults)


def test_result_row_carries_full_provenance():
    row = result_row(
        make_result("ep-0000", worker_id="worker-b", backend="kinematic",
                    model_version="pure_pursuit", receive_count=2)
    )
    assert row["worker_id"] == "worker-b"
    assert row["simulator_backend"] == "kinematic"
    assert row["model_version"] == "pure_pursuit"
    assert row["receive_count"] == 2
    assert row["driving_score"] == 50.0


def test_result_row_reads_a_missing_receive_count_as_one():
    # proto3 renders an unset int32 as 0; a writer predating the field must
    # read as "no redelivery observed", not as a negative redelivery.
    proto = make_result("ep-0000")
    proto.receive_count = 0
    assert result_row(proto)["receive_count"] == 1


def test_queue_statistics_count_redeliveries_from_result_rows():
    rows = [
        result_row(make_result("ep-0000", worker_id="survivor", receive_count=3)),
        result_row(make_result("ep-0001")),
    ]
    stats = queue_statistics(LocalMessageQueue(), rows)
    assert stats["redeliveries"] == 2
    assert stats["redelivered_episodes"] == [
        {"episode_id": "ep-0000", "completed_by": "survivor", "receive_count": 3}
    ]
    assert stats["dead_letters"] == 0
    assert stats["approximate_depth"] == 0


def test_queue_statistics_surface_dead_letters():
    queue = LocalMessageQueue(visibility_timeout=0.05, max_receive_count=1)
    queue.send({"episode_id": "ep-poison"})
    assert queue.receive()  # delivery 1 uses up the budget
    time.sleep(0.1)
    assert queue.receive() == []  # delivery 2 exceeds it -> dead-lettered
    stats = queue_statistics(queue, [])
    assert stats["dead_letters"] == 1
    assert stats["dead_letter_episode_ids"] == ["ep-poison"]


def test_report_aggregates_all_four_evidence_sources():
    report = make_report()
    assert report["kind"] == "distributed_run"
    assert report["backend"] == "kinematic"
    assert report["scope"] == SCOPE_PIPELINE_ONLY
    assert report["coordinator"]["run_id"] == "run-1"
    assert report["coordinator"]["duplicate_submissions"] == 0
    assert {worker["worker_id"] for worker in report["coordinator"]["workers"]} == {
        "worker-a", "worker-chaos",
    }
    assert report["queue"]["redeliveries"] == 1
    assert report["telemetry"]["rows_written"] == 100
    assert report["summary"]["episodes"] == 2
    assert report["episodes_missing_results"] == []
    assert report["results_not_in_suite"] == []
    assert report["failed_episodes"] == []


def test_report_cross_checks_suite_against_results():
    report = make_report(
        specs=[make_spec(0), make_spec(1), make_spec(2)],
        results=[
            make_result("ep-0000"),
            make_result("ep-9999"),
            make_result("ep-0001", status="failed"),
        ],
    )
    # ep-0001 failed but it *has* a result; only ep-0002 is truly missing.
    assert report["episodes_missing_results"] == ["ep-0002"]
    assert report["results_not_in_suite"] == ["ep-9999"]
    assert report["failed_episodes"] == ["ep-0001"]


def test_only_an_all_carla_run_earns_driving_quality():
    carla_only = make_report(
        results=[make_result("ep-0000", backend="carla"), make_result("ep-0001", backend="carla")]
    )
    assert carla_only["scope"] == SCOPE_DRIVING_QUALITY

    mixed = make_report(
        results=[make_result("ep-0000", backend="carla"), make_result("ep-0001")]
    )
    assert mixed["scope"] == SCOPE_PIPELINE_ONLY
    assert "mix simulator backends" in mixed["scope_note"]

    empty = make_report(results=[], status=make_status(completed=0))
    assert empty["scope"] == SCOPE_PIPELINE_ONLY
    assert empty["backend"] == "none"
    assert empty["summary"] is None


def test_missing_telemetry_is_recorded_not_omitted():
    report = make_report(telemetry=None)
    assert report["telemetry"]["archived"] is False
    assert "no telemetry stream" in report["telemetry"]["reason"]


@pytest.fixture()
def rendered() -> str:
    return render_writeup(make_report(), source="results/distributed/report.json")


def test_writeup_carries_the_required_sentences(rendered):
    assert "Scope: pipeline-only" in rendered
    assert "results/distributed/report.json" in rendered
    assert "at-least-once" in rendered
    assert "Redeliveries: 1" in rendered
    assert "Dead-lettered: 0" in rendered
    assert "`ep-0000` was delivered 2 times and completed by `worker-a`" in rendered


def test_writeup_attributes_every_result_row(rendered):
    # Worker, backend, and Policy provenance on each per-episode row.
    assert (
        "| ep-0000 | worker-a | kinematic | pure_pursuit | 2 | 50.0 | completed |"
        in rendered
    )
    assert (
        "| ep-0001 | worker-a | kinematic | pure_pursuit | 1 | 50.0 | completed |"
        in rendered
    )


def test_writeup_states_the_telemetry_totals(rendered):
    assert "Rows written: 100" in rendered
    assert "Parquet files: 1" in rendered

    unarchived = render_writeup(
        make_report(telemetry=None), source="results/distributed/report.json"
    )
    assert "Not archived" in unarchived


def test_writeup_renders_an_empty_run_without_dying():
    report = make_report(results=[], status=make_status(completed=0))
    text = render_writeup(report, source="results/distributed/report.json")
    assert "nothing to aggregate" in text


def test_cli_collects_from_a_live_coordinator_and_lands_both_artifacts(tmp_path):
    """The CLI wiring against a real socket: a coordinator with no results is
    the smallest honest run, and the collector must still land a report that
    says so rather than dying on the empty aggregate."""
    import asyncio
    import json
    import threading

    import grpc

    from pathfinder import distributed_run
    from pathfinder.rpc.coordinator import CoordinatorService
    from pathfinder.rpc.generated import pathfinder_pb2_grpc

    started = threading.Event()
    box: dict = {}

    def serve_in_thread() -> None:
        async def serve() -> None:
            service = CoordinatorService(episodes_total=2)
            server = grpc.aio.server()
            pathfinder_pb2_grpc.add_CoordinatorServicer_to_server(service, server)
            box["port"] = server.add_insecure_port("127.0.0.1:0")
            box["loop"] = asyncio.get_running_loop()
            box["stop"] = asyncio.Event()
            await server.start()
            started.set()
            await box["stop"].wait()
            await server.stop(None)

        asyncio.run(serve())

    thread = threading.Thread(target=serve_in_thread, daemon=True)
    thread.start()
    assert started.wait(5), "coordinator never came up"

    output = tmp_path / "report.json"
    try:
        code = distributed_run.main(
            [
                "--coordinator", f"127.0.0.1:{box['port']}",
                "--queue-backend", "local",
                "--episodes", "2",
                "--output", str(output),
            ]
        )
    finally:
        box["loop"].call_soon_threadsafe(box["stop"].set)
        thread.join(5)

    assert code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["kind"] == "distributed_run"
    assert report["scope"] == SCOPE_PIPELINE_ONLY
    assert report["summary"] is None
    assert report["telemetry"]["archived"] is False
    assert output.with_suffix(".md").exists()
