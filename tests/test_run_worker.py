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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_worker import run_worker  # noqa: E402 - needs scripts/ on sys.path first

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
