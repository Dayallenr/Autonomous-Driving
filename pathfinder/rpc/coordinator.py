"""
Coordinator gRPC service: worker registration, heartbeats, result collection.

State lives in memory here, deliberately. The coordinator is not the durable
record of a run — the queue owns work durability and the warehouse owns results.
If the coordinator restarts, workers re-register and keep going; nothing is lost
that matters. Adding a database behind this would introduce a second source of
truth about run state, and the two would drift.

Liveness
--------
A worker is "healthy" if it has heartbeated within
``HEARTBEAT_TIMEOUT_MULTIPLIER`` intervals. That multiplier exists because a
worker mid-episode is doing blocking simulator work and a single missed
heartbeat is normal; declaring death on one miss produces constant false
positives and, worse, encourages someone to "fix" it by killing healthy workers.

Note that worker liveness here is **advisory only** — it drives the dashboard.
Actual work recovery is the queue's visibility timeout, which does not depend on
the coordinator being up at all.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field

import grpc

from pathfinder.rpc.generated import pathfinder_pb2, pathfinder_pb2_grpc

logger = logging.getLogger(__name__)

__all__ = ["CoordinatorService", "DEFAULT_HEARTBEAT_SECONDS"]

DEFAULT_HEARTBEAT_SECONDS = 5.0

#: Missed intervals before a worker is considered unhealthy.
HEARTBEAT_TIMEOUT_MULTIPLIER = 3.0


@dataclass
class _Worker:
    worker_id: str
    hostname: str
    simulator_backend: str
    registered_at: float
    last_heartbeat: float
    current_episode_id: str = ""
    episode_progress: float = 0.0
    episodes_completed: int = 0
    frames_completed: int = 0


@dataclass
class _Run:
    run_id: str
    started_at: float
    episodes_total: int
    results: dict[str, pathfinder_pb2.EpisodeResult] = field(default_factory=dict)
    duplicates: int = 0


class CoordinatorService(pathfinder_pb2_grpc.CoordinatorServicer):
    """Tracks workers and collects episode results for one benchmark run."""

    def __init__(
        self,
        *,
        episodes_total: int = 0,
        heartbeat_seconds: float = DEFAULT_HEARTBEAT_SECONDS,
        run_id: str | None = None,
    ) -> None:
        self.heartbeat_seconds = heartbeat_seconds
        self.run = _Run(
            run_id=run_id or uuid.uuid4().hex[:12],
            started_at=time.monotonic(),
            episodes_total=episodes_total,
        )
        self.workers: dict[str, _Worker] = {}
        self._stop_requested = False
        self._stop_reason = ""
        # Guards workers/results against concurrent RPC handlers. grpc.aio runs
        # handlers on one event loop, but an await inside a handler is a
        # suspension point, so mutations still need to be atomic across them.
        self._lock = asyncio.Lock()

    def request_stop(self, reason: str = "run cancelled") -> None:
        """Ask every worker to stop after its current episode.

        Cooperative rather than forceful: a worker killed mid-episode loses a
        result that took minutes to produce, and the queue would then redeliver
        that episode to someone else. Letting it finish is strictly cheaper.
        """
        self._stop_requested = True
        self._stop_reason = reason

    async def RegisterWorker(self, request, context):  # noqa: N802 - gRPC naming
        if not request.worker_id:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "worker_id is required")
        async with self._lock:
            now = time.monotonic()
            self.workers[request.worker_id] = _Worker(
                worker_id=request.worker_id,
                hostname=request.hostname,
                simulator_backend=request.simulator_backend,
                registered_at=now,
                last_heartbeat=now,
            )
        logger.info(
            "worker %s registered from %s (backend=%s)",
            request.worker_id, request.hostname or "?", request.simulator_backend or "?",
        )
        return pathfinder_pb2.RegisterResponse(
            accepted=True,
            run_id=self.run.run_id,
            heartbeat_interval_seconds=self.heartbeat_seconds,
        )

    async def Heartbeat(self, request, context):  # noqa: N802
        async with self._lock:
            worker = self.workers.get(request.worker_id)
            if worker is None:
                # Re-register implicitly: the coordinator may have restarted, and
                # rejecting the heartbeat would strand a perfectly healthy worker.
                worker = _Worker(
                    worker_id=request.worker_id,
                    hostname="",
                    simulator_backend="",
                    registered_at=time.monotonic(),
                    last_heartbeat=time.monotonic(),
                )
                self.workers[request.worker_id] = worker
                logger.info("worker %s re-registered via heartbeat", request.worker_id)

            worker.last_heartbeat = time.monotonic()
            worker.current_episode_id = request.current_episode_id
            worker.episode_progress = request.episode_progress
            worker.episodes_completed = request.episodes_completed
            worker.frames_completed = request.frames_completed

        return pathfinder_pb2.HeartbeatResponse(
            should_stop=self._stop_requested, reason=self._stop_reason
        )

    async def SubmitResult(self, request, context):  # noqa: N802
        result = request.result
        if not result.episode_id:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "episode_id is required")
        async with self._lock:
            duplicate = result.episode_id in self.run.results
            if duplicate:
                self.run.duplicates += 1
                logger.info(
                    "duplicate result for %s from %s (at-least-once delivery; overwriting)",
                    result.episode_id, result.worker_id,
                )
            self.run.results[result.episode_id] = result
        return pathfinder_pb2.SubmitResultResponse(accepted=True, duplicate=duplicate)

    def _status(self) -> pathfinder_pb2.RunStatusResponse:
        now = time.monotonic()
        timeout = self.heartbeat_seconds * HEARTBEAT_TIMEOUT_MULTIPLIER
        results = list(self.run.results.values())

        return pathfinder_pb2.RunStatusResponse(
            run_id=self.run.run_id,
            episodes_total=self.run.episodes_total,
            episodes_completed=len(results),
            episodes_in_flight=sum(
                1 for worker in self.workers.values() if worker.current_episode_id
            ),
            mean_driving_score=(
                sum(r.driving_score for r in results) / len(results) if results else 0.0
            ),
            mean_route_completion=(
                sum(r.route_completion for r in results) / len(results) if results else 0.0
            ),
            workers=[
                pathfinder_pb2.WorkerStatus(
                    worker_id=worker.worker_id,
                    current_episode_id=worker.current_episode_id,
                    episode_progress=worker.episode_progress,
                    episodes_completed=worker.episodes_completed,
                    seconds_since_heartbeat=now - worker.last_heartbeat,
                    healthy=(now - worker.last_heartbeat) < timeout,
                )
                for worker in sorted(self.workers.values(), key=lambda w: w.worker_id)
            ],
            elapsed_seconds=now - self.run.started_at,
        )

    async def GetRunStatus(self, request, context):  # noqa: N802
        async with self._lock:
            return self._status()

    async def WatchRun(self, request, context):  # noqa: N802
        """Stream status until the run completes or the client disconnects."""
        while True:
            async with self._lock:
                status = self._status()
            yield status
            if (
                self.run.episodes_total
                and status.episodes_completed >= self.run.episodes_total
            ):
                return
            try:
                await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                # Client disconnected; ending the generator is the correct
                # response, not an error.
                return
