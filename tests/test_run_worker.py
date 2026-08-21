"""
End-to-end test for scripts/run_worker.py: a real gRPC server, a real
LocalMessageQueue, and the kinematic simulator backend, wired together the
same way a container would run them. This is the path Docker/Kubernetes
(Phases 3-4) will actually invoke, so it needs coverage beyond the
server/client unit tests in test_rpc.py.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import grpc
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import run_worker as run_worker_module  # noqa: E402 - needs scripts/ on sys.path first
from run_worker import run_worker  # noqa: E402

from pathfinder.cloud.queue import LocalMessageQueue
from pathfinder.rpc.coordinator import CoordinatorService
from pathfinder.rpc.generated import pathfinder_pb2_grpc
from pathfinder.sim.base import EpisodeSpec


def test_worker_registers_runs_and_submits_over_real_grpc():
    async def body():
        service = CoordinatorService(episodes_total=1)
        server = grpc.aio.server()
        pathfinder_pb2_grpc.add_CoordinatorServicer_to_server(service, server)
        port = server.add_insecure_port("127.0.0.1:0")
        await server.start()

        queue = LocalMessageQueue()
        spec = EpisodeSpec(episode_id="ep-worker-test", route_length_m=80.0, max_steps=200, seed=1)
        queue.send(spec.to_dict())

        try:
            results = await run_worker(
                worker_id="test-worker",
                coordinator_address=f"127.0.0.1:{port}",
                queue=queue,
                simulator_backend="kinematic",
                idle_timeout_seconds=0.5,
                receive_wait_seconds=0.2,
            )
        finally:
            await server.stop(None)

        assert len(results) == 1
        assert results[0].episode_id == "ep-worker-test"

        # The coordinator must have received the result over the wire, not
        # just seen it returned locally.
        assert "ep-worker-test" in service.run.results
        assert service.run.results["ep-worker-test"].worker_id == "test-worker"
        assert "test-worker" in service.workers
        assert service.workers["test-worker"].episodes_completed == 1

    asyncio.run(body())


def test_chaos_kill_fires_mid_episode_and_leaves_the_message_unacknowledged(monkeypatch):
    """--chaos-kill-after-frames is the #22 runbook's deterministic worker
    kill: a kinematic episode takes ~10 ms, so no by-hand kill can land
    mid-episode. The real thing is os._exit — no cleanup, no queue delete —
    which a test cannot survive, so the seam ``_chaos_exit`` is patched to a
    BaseException stand-in; what the test pins is that the kill fires at the
    configured frame with the episode still unacknowledged on the queue."""

    class ChaosExit(BaseException):
        pass

    monkeypatch.setattr(
        run_worker_module, "_chaos_exit", lambda: (_ for _ in ()).throw(ChaosExit())
    )

    async def body():
        service = CoordinatorService(episodes_total=1)
        server = grpc.aio.server()
        pathfinder_pb2_grpc.add_CoordinatorServicer_to_server(service, server)
        port = server.add_insecure_port("127.0.0.1:0")
        await server.start()

        # Visibility timeout longer than the test, so the undeleted message
        # is observably in flight rather than already redelivered.
        queue = LocalMessageQueue(visibility_timeout=60.0)
        spec = EpisodeSpec(episode_id="ep-chaos-test", route_length_m=80.0, max_steps=200, seed=1)
        queue.send(spec.to_dict())

        try:
            with pytest.raises(ChaosExit):
                await run_worker(
                    worker_id="chaos-victim",
                    coordinator_address=f"127.0.0.1:{port}",
                    queue=queue,
                    simulator_backend="kinematic",
                    idle_timeout_seconds=0.5,
                    receive_wait_seconds=0.2,
                    chaos_kill_after_frames=50,
                )
        finally:
            await server.stop(None)

        # Killed mid-episode: the message is neither deleted nor visible yet —
        # exactly the state a SIGKILL'd worker leaves for the visibility
        # timeout to repair — and no result reached the coordinator.
        assert queue.in_flight() == 1
        assert queue.approximate_depth() == 0
        assert queue.dead_letters() == []
        assert service.run.results == {}

    asyncio.run(body())
