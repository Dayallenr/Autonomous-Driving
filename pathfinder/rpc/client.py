"""
Async client for the coordinator gRPC service.

One class wraps the five RPCs so a worker script reads as a sequence of verbs
(register, heartbeat, submit, watch) instead of channel/stub plumbing. The
heartbeat loop lives here too, as a background task, because every worker
needs the identical loop and getting the cadence wrong (too fast floods the
coordinator, too slow trips its liveness timeout) is exactly the kind of
detail that should be written once.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator

import grpc

from pathfinder.metrics.driving_score import EpisodeScore
from pathfinder.rpc.generated import pathfinder_pb2, pathfinder_pb2_grpc

logger = logging.getLogger(__name__)

__all__ = ["CoordinatorClient", "episode_score_to_proto"]


def episode_score_to_proto(
    score: EpisodeScore,
    *,
    worker_id: str,
    model_version: str = "",
    dataset_version: str = "",
    simulator_backend: str = "",
) -> pathfinder_pb2.EpisodeResult:
    """Convert a local :class:`EpisodeScore` into the wire message.

    ``EpisodeScore`` has no worker/model/backend fields — ``run_episode``
    doesn't know which worker or model produced it, only how the drive went.
    Those get attached here, at the point where the result is about to leave
    the process that does know.
    """
    return pathfinder_pb2.EpisodeResult(
        episode_id=score.episode_id,
        worker_id=worker_id,
        route_completion=score.route_completion,
        infraction_penalty=score.infraction_penalty,
        driving_score=score.driving_score,
        infractions=[
            pathfinder_pb2.InfractionCount(kind=kind, count=count)
            for kind, count in score.infractions.items()
        ],
        distance_travelled_m=score.distance_travelled_m,
        route_length_m=score.route_length_m,
        frames=score.frames,
        duration_seconds=score.duration_seconds,
        mean_fps=score.mean_fps,
        status=score.status,
        termination_reason=score.termination_reason,
        model_version=model_version,
        dataset_version=dataset_version,
        simulator_backend=simulator_backend,
    )


class CoordinatorClient:
    """Async wrapper around ``CoordinatorStub``, plus a managed heartbeat loop."""

    def __init__(self, channel: grpc.aio.Channel) -> None:
        self._channel = channel
        self._stub = pathfinder_pb2_grpc.CoordinatorStub(channel)
        self._heartbeat_task: asyncio.Task | None = None
        self._should_stop = False

    @classmethod
    def connect(cls, address: str) -> CoordinatorClient:
        """Open an insecure channel to ``host:port``. Does not block — gRPC
        channels connect lazily on first RPC."""
        return cls(grpc.aio.insecure_channel(address))

    async def close(self) -> None:
        self.stop_heartbeat()
        if self._heartbeat_task is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await self._heartbeat_task
        await self._channel.close()

    async def register_worker(
        self,
        worker_id: str,
        *,
        hostname: str = "",
        simulator_backend: str = "",
        carla_port: int = 0,
    ) -> pathfinder_pb2.RegisterResponse:
        request = pathfinder_pb2.RegisterRequest(
            worker_id=worker_id,
            hostname=hostname,
            simulator_backend=simulator_backend,
            carla_port=carla_port,
        )
        return await self._stub.RegisterWorker(request)

    async def heartbeat_once(
        self,
        worker_id: str,
        *,
        current_episode_id: str = "",
        frames_completed: int = 0,
        episode_progress: float = 0.0,
        episodes_completed: int = 0,
    ) -> pathfinder_pb2.HeartbeatResponse:
        request = pathfinder_pb2.HeartbeatRequest(
            worker_id=worker_id,
            current_episode_id=current_episode_id,
            frames_completed=frames_completed,
            episode_progress=episode_progress,
            episodes_completed=episodes_completed,
        )
        response = await self._stub.Heartbeat(request)
        self._should_stop = response.should_stop
        return response

    def start_heartbeat(
        self, worker_id: str, interval_seconds: float, state: WorkerHeartbeatState
    ) -> None:
        """Start a background task that heartbeats on ``interval_seconds`` using
        live values from ``state``, which the worker loop mutates as it works."""
        if self._heartbeat_task is not None:
            raise RuntimeError("heartbeat already started")
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(worker_id, interval_seconds, state)
        )

    def stop_heartbeat(self) -> None:
        if self._heartbeat_task is not None:
            self._heartbeat_task.cancel()

    async def _heartbeat_loop(
        self, worker_id: str, interval_seconds: float, state: WorkerHeartbeatState
    ) -> None:
        try:
            while True:
                await asyncio.sleep(interval_seconds)
                try:
                    await self.heartbeat_once(
                        worker_id,
                        current_episode_id=state.current_episode_id,
                        frames_completed=state.frames_completed,
                        episode_progress=state.episode_progress,
                        episodes_completed=state.episodes_completed,
                    )
                except grpc.aio.AioRpcError as error:
                    # A missed heartbeat is not fatal — the coordinator's own
                    # timeout multiplier exists precisely to absorb this.
                    logger.warning("heartbeat failed for %s: %s", worker_id, error)
        except asyncio.CancelledError:
            return

    @property
    def should_stop(self) -> bool:
        """True if the coordinator's most recent heartbeat response asked this
        worker to stop after its current episode."""
        return self._should_stop

    async def submit_result(
        self, result: pathfinder_pb2.EpisodeResult
    ) -> pathfinder_pb2.SubmitResultResponse:
        return await self._stub.SubmitResult(pathfinder_pb2.SubmitResultRequest(result=result))

    async def get_run_status(self, run_id: str) -> pathfinder_pb2.RunStatusResponse:
        return await self._stub.GetRunStatus(pathfinder_pb2.RunStatusRequest(run_id=run_id))

    async def watch_run(self, run_id: str) -> AsyncIterator[pathfinder_pb2.RunStatusResponse]:
        async for status in self._stub.WatchRun(pathfinder_pb2.RunStatusRequest(run_id=run_id)):
            yield status


class WorkerHeartbeatState:
    """Mutable box the worker loop updates and the heartbeat task reads.

    A plain object rather than passing five loose arguments around — the
    heartbeat task needs to see live progress, not the values at task-start.
    """

    __slots__ = (
        "current_episode_id",
        "frames_completed",
        "episode_progress",
        "episodes_completed",
    )

    def __init__(self) -> None:
        self.current_episode_id = ""
        self.frames_completed = 0
        self.episode_progress = 0.0
        self.episodes_completed = 0
