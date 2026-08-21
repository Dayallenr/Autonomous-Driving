"""
The Phase 5 chaos test (issue #17): the whole distributed pipeline, end to
end, with a worker killed mid-episode — CARLA-free, AWS-free, network-free
beyond loopback gRPC.

A seeded suite goes onto a local queue with real visibility-timeout
semantics; a coordinator serves over a real socket; a chaos worker takes the
first episode and dies mid-drive without acknowledging it; a survivor
drains the rest, picks up the redelivered episode after the visibility
timeout, and completes it. Telemetry from both workers lands in partitioned
Parquet, and the collector assembles the one report artifact the ticket
asks for — redelivery counted, dead-letter queue empty, provenance on every
row, warehouse totals matching what is actually on disk.

The kill is a ``BaseException`` raised from inside the driving loop: it
sails past ``run_episode``'s ``except Exception`` (which would score the
episode as failed and acknowledge it — a different, gentler failure), so
the worker dies with the message unacknowledged, exactly like a SIGKILL
between receive and delete. A plain Exception would test the wrong path.
"""
from __future__ import annotations

import asyncio
import io
import sys
from pathlib import Path

import grpc
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_worker import run_worker  # noqa: E402 - needs scripts/ on sys.path first

from pathfinder import reporting
from pathfinder.cloud.objects import LocalObjectStore
from pathfinder.cloud.queue import LocalMessageQueue
from pathfinder.cloud.stream import LocalTelemetryStream
from pathfinder.cloud.warehouse import TelemetryWarehouse
from pathfinder.distributed_run import archive_telemetry, collect_report
from pathfinder.distributed_writeup import render_writeup
from pathfinder.orchestration import build_episode_specs
from pathfinder.rpc.coordinator import CoordinatorService
from pathfinder.rpc.generated import pathfinder_pb2_grpc
from pathfinder.runner import PurePursuitPolicy

EPISODES = 4
CHAOS_KILL_AT_FRAME = 10
VISIBILITY_TIMEOUT = 1.5


class ChaosKill(BaseException):
    """The chaos monkey. BaseException on purpose — see the module docstring."""


class DiesMidEpisode:
    """PurePursuit until frame CHAOS_KILL_AT_FRAME, then the process 'dies'."""

    NAME = PurePursuitPolicy.NAME

    def __init__(self) -> None:
        self.inner = PurePursuitPolicy()
        self.frames_left = CHAOS_KILL_AT_FRAME

    def plan(self, state):
        if self.frames_left <= 0:
            raise ChaosKill("worker killed mid-episode")
        self.frames_left -= 1
        return self.inner.plan(state)


def test_chaos_kill_redelivery_parquet_and_report(tmp_path):
    specs = build_episode_specs(
        count=EPISODES, route_length_m=60.0, base_seed=500, max_steps=60
    )
    queue = LocalMessageQueue(visibility_timeout=VISIBILITY_TIMEOUT, max_receive_count=5)
    for spec in specs:
        queue.send(spec.to_dict())

    stream = LocalTelemetryStream(tmp_path / "telemetry")

    async def body():
        service = CoordinatorService(episodes_total=EPISODES, heartbeat_seconds=0.2)
        server = grpc.aio.server()
        pathfinder_pb2_grpc.add_CoordinatorServicer_to_server(service, server)
        port = server.add_insecure_port("127.0.0.1:0")
        await server.start()
        address = f"127.0.0.1:{port}"

        try:
            # Act 1: the chaos worker takes the first episode and dies
            # mid-drive, after emitting some telemetry but before
            # acknowledging the message or submitting a result.
            with pytest.raises(ChaosKill):
                await run_worker(
                    worker_id="worker-chaos",
                    coordinator_address=address,
                    queue=queue,
                    simulator_backend="kinematic",
                    policy=DiesMidEpisode(),
                    idle_timeout_seconds=0.5,
                    receive_wait_seconds=0.2,
                    telemetry_stream=stream,
                )

            # The episode it held is in flight, not lost and not visible yet.
            assert queue.in_flight() == 1
            assert queue.approximate_depth() == EPISODES - 1

            # Act 2: the survivor drains the queue. Its idle timeout outlasts
            # the visibility timeout, so it is still polling when the dead
            # worker's episode reappears, and completes it too.
            survivor_results = await run_worker(
                worker_id="worker-survivor",
                coordinator_address=address,
                queue=queue,
                simulator_backend="kinematic",
                idle_timeout_seconds=VISIBILITY_TIMEOUT + 2.0,
                receive_wait_seconds=0.2,
                telemetry_stream=stream,
            )
            assert len(survivor_results) == EPISODES

            # Act 3: archive telemetry and collect the report over real gRPC.
            store = LocalObjectStore(tmp_path / "store")
            warehouse = TelemetryWarehouse(store, prefix="telemetry")
            report = await collect_report(
                coordinator_address=address,
                specs=specs,
                queue=queue,
                telemetry=archive_telemetry(stream, warehouse),
            )
            return report, store, survivor_results, service
        finally:
            await server.stop(None)

    report, store, survivor_results, service = asyncio.run(body())

    # The killed worker's episode was redelivered and completed by the
    # survivor, and the report's redelivery count says so.
    killed_episode = specs[0].episode_id
    assert report["queue"]["redeliveries"] == 1
    assert report["queue"]["redelivered_episodes"] == [
        {"episode_id": killed_episode, "completed_by": "worker-survivor", "receive_count": 2}
    ]

    # Nothing was lost and nothing was poisoned: the queue is empty, the
    # dead-letter queue is empty, and suite and results match exactly.
    assert report["queue"]["approximate_depth"] == 0
    assert report["queue"]["dead_letters"] == 0
    assert report["episodes_missing_results"] == []
    assert report["results_not_in_suite"] == []
    assert report["coordinator"]["episodes_completed"] == EPISODES

    # Every result row carries worker id, simulator backend, and Policy
    # provenance — the survivor completed all of them.
    assert len(report["results"]) == EPISODES
    for row in report["results"]:
        assert row["worker_id"] == "worker-survivor"
        assert row["simulator_backend"] == "kinematic"
        assert row["model_version"] == "pure_pursuit"

    # Both workers registered; the dead one is unhealthy by collection time.
    workers = {worker["worker_id"]: worker for worker in report["coordinator"]["workers"]}
    assert set(workers) == {"worker-chaos", "worker-survivor"}
    assert workers["worker-chaos"]["healthy"] is False
    assert workers["worker-chaos"]["episodes_completed"] == 0

    # Telemetry: every frame both workers emitted — the survivor's full
    # episodes plus the chaos worker's partial one — is in partitioned
    # Parquet, and the report's row count matches what is actually on disk.
    expected_rows = sum(score.frames for score in survivor_results) + CHAOS_KILL_AT_FRAME
    assert report["telemetry"]["archived"] is True
    assert report["telemetry"]["rows_written"] == expected_rows
    parquet_rows = 0
    for key in report["telemetry"]["parquet_files"]:
        assert key.startswith("telemetry/frames/event_date=")
        table = pq.read_table(io.BytesIO(store.get(key)))
        parquet_rows += table.num_rows
        # Frame-level provenance survives into the warehouse: both the dead
        # and the surviving worker are attributable in the archived rows.
        assert set(table.column("worker_id").to_pylist()) <= {
            "worker-chaos", "worker-survivor",
        }
    assert parquet_rows == expected_rows

    # A kinematic run must never read as driving quality.
    assert report["scope"] == reporting.SCOPE_PIPELINE_ONLY

    # The report-writer lands the JSON artifact and its write-up together.
    output = tmp_path / "report.json"
    artifact = reporting.ReportArtifact(output, label_key="worker_id")
    writeup = artifact.finish(report, render_writeup)
    assert output.exists()
    assert writeup.read_text(encoding="utf-8").startswith("# Distributed benchmark run")
    assert f"`{killed_episode}` was delivered 2 times" in writeup.read_text(encoding="utf-8")
