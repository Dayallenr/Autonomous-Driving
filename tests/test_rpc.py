"""
Integration tests for the coordinator gRPC server + client.

These exercise real network calls over loopback (an ephemeral port, not a
mock channel) — the gap the audit found was that ``CoordinatorService`` had
never actually been bound to a socket or driven by a real client. Correctness
of the service's internal state machine (dedup, liveness math) already has
unit-style coverage in intent via these same code paths; what's new here is
proving the RPCs work end-to-end over the wire.

No pytest-asyncio dependency: each test wraps its body in ``asyncio.run``
directly, since nothing here needs async fixtures shared across tests.
"""
from __future__ import annotations

import asyncio

import grpc
import pytest

from pathfinder.metrics.driving_score import EpisodeScore
from pathfinder.rpc.client import CoordinatorClient, episode_score_to_proto
from pathfinder.rpc.coordinator import CoordinatorService
from pathfinder.rpc.generated import pathfinder_pb2_grpc


async def _start_server(
    *, episodes_total: int = 0, heartbeat_seconds: float = 0.2
) -> tuple[grpc.aio.Server, CoordinatorService, int]:
    service = CoordinatorService(episodes_total=episodes_total, heartbeat_seconds=heartbeat_seconds)
    server = grpc.aio.server()
    pathfinder_pb2_grpc.add_CoordinatorServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    await server.start()
    return server, service, port


def _sample_score(episode_id: str = "ep-0001") -> EpisodeScore:
    return EpisodeScore(
        episode_id=episode_id,
        route_completion=0.92,
        infraction_penalty=0.7,
        driving_score=64.4,
        infractions={"red_light": 1},
        distance_travelled_m=368.0,
        route_length_m=400.0,
        frames=1500,
        duration_seconds=75.0,
    )


def test_register_worker_over_real_grpc():
    async def body():
        server, service, port = await _start_server()
        client = CoordinatorClient.connect(f"127.0.0.1:{port}")
        try:
            response = await client.register_worker(
                "worker-a", hostname="host-a", simulator_backend="kinematic"
            )
            assert response.accepted
            assert response.run_id == service.run.run_id
            assert response.heartbeat_interval_seconds == 0.2
            assert "worker-a" in service.workers
        finally:
            await client.close()
            await server.stop(None)

    asyncio.run(body())


def test_heartbeat_reflected_in_run_status():
    async def body():
        server, service, port = await _start_server()
        client = CoordinatorClient.connect(f"127.0.0.1:{port}")
        try:
            await client.register_worker("worker-a")
            await client.heartbeat_once(
                "worker-a", current_episode_id="ep-0001", episode_progress=0.4, frames_completed=200
            )
            status = await client.get_run_status(service.run.run_id)
            assert len(status.workers) == 1
            worker = status.workers[0]
            assert worker.worker_id == "worker-a"
            assert worker.current_episode_id == "ep-0001"
            assert worker.episode_progress == pytest.approx(0.4)
            assert worker.healthy is True
        finally:
            await client.close()
            await server.stop(None)

    asyncio.run(body())


def test_submit_result_dedups_on_resubmit():
    async def body():
        server, service, port = await _start_server()
        client = CoordinatorClient.connect(f"127.0.0.1:{port}")
        try:
            proto_result = episode_score_to_proto(
                _sample_score(), worker_id="worker-a", simulator_backend="kinematic"
            )
            first = await client.submit_result(proto_result)
            second = await client.submit_result(proto_result)
            assert first.accepted and not first.duplicate
            assert second.accepted and second.duplicate

            status = await client.get_run_status(service.run.run_id)
            # A duplicate overwrites rather than appending, so this must stay 1
            # even though submit_result was called twice.
            assert status.episodes_completed == 1
        finally:
            await client.close()
            await server.stop(None)

    asyncio.run(body())


def test_run_results_returns_full_rows_with_provenance():
    async def body():
        server, service, port = await _start_server()
        client = CoordinatorClient.connect(f"127.0.0.1:{port}")
        try:
            redelivered = episode_score_to_proto(
                _sample_score("ep-0001"),
                worker_id="worker-b",
                model_version="pure_pursuit",
                simulator_backend="kinematic",
                receive_count=2,
            )
            fresh = episode_score_to_proto(
                _sample_score("ep-0002"), worker_id="worker-a", simulator_backend="kinematic"
            )
            await client.submit_result(redelivered)
            await client.submit_result(fresh)
            # A resubmission must show up as a duplicate, not a third row.
            await client.submit_result(fresh)

            response = await client.get_run_results(service.run.run_id)
            assert response.run_id == service.run.run_id
            assert response.duplicate_submissions == 1
            rows = {row.episode_id: row for row in response.results}
            assert set(rows) == {"ep-0001", "ep-0002"}
            assert rows["ep-0001"].worker_id == "worker-b"
            assert rows["ep-0001"].model_version == "pure_pursuit"
            assert rows["ep-0001"].simulator_backend == "kinematic"
            # The redelivery observation survives the wire — this is what the
            # distributed-run report counts redeliveries from.
            assert rows["ep-0001"].receive_count == 2
            assert rows["ep-0002"].receive_count == 1
        finally:
            await client.close()
            await server.stop(None)

    asyncio.run(body())


def test_cooperative_stop_is_seen_on_next_heartbeat():
    async def body():
        server, service, port = await _start_server()
        client = CoordinatorClient.connect(f"127.0.0.1:{port}")
        try:
            await client.register_worker("worker-a")
            before = await client.heartbeat_once("worker-a")
            assert before.should_stop is False

            service.request_stop("run cancelled by operator")
            after = await client.heartbeat_once("worker-a")
            assert after.should_stop is True
            assert after.reason == "run cancelled by operator"
        finally:
            await client.close()
            await server.stop(None)

    asyncio.run(body())


def test_watch_run_streams_until_episodes_complete():
    async def body():
        server, service, port = await _start_server(episodes_total=1)
        client = CoordinatorClient.connect(f"127.0.0.1:{port}")
        try:
            proto_result = episode_score_to_proto(
                _sample_score(), worker_id="worker-a", simulator_backend="kinematic"
            )
            await client.submit_result(proto_result)

            statuses = []
            async for status in client.watch_run(service.run.run_id):
                statuses.append(status)
                if status.episodes_completed >= status.episodes_total:
                    break

            assert statuses[-1].episodes_completed == 1
            assert statuses[-1].episodes_total == 1
        finally:
            await client.close()
            await server.stop(None)

    asyncio.run(body())
