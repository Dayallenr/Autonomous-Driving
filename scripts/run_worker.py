"""
Distributed benchmark worker: registers with the coordinator over gRPC, pulls
episodes off a queue, drives them, and reports results back over gRPC.

This is the process that actually runs inside a worker container/pod (Phases
3-4). It is a different thing from ``pathfinder.orchestration.EpisodeWorker``:
that class runs in-process threads for a local single-machine benchmark, while
this script is a standalone process that talks to a coordinator and queue it
does not share memory with — the shape a real distributed deployment needs.

Usage:
    python scripts/run_worker.py \\
        --worker-id worker-1 \\
        --coordinator localhost:50051 \\
        --queue-backend local \\
        --simulator-backend kinematic

The kinematic backend is the only one that makes sense in a container: CARLA
needs a GPU and a running CarlaUE4 server this process does not have.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import logging
import signal
import socket
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import grpc

from pathfinder.cloud import cli as cloud_cli
from pathfinder.cloud.queue import MessageQueue
from pathfinder.cloud.stream import TelemetryStream
from pathfinder.metrics.driving_score import EpisodeScore
from pathfinder.policies import (
    DEFAULT_POLICY,
    POLICY_NAMES,
    build_policy,
    result_label,
)
from pathfinder.rpc.client import CoordinatorClient, WorkerHeartbeatState, episode_score_to_proto
from pathfinder.runner import Policy, run_episode
from pathfinder.sim.base import EpisodeSpec
from pathfinder.sim.carla_backend import build_simulator

logger = logging.getLogger(__name__)


async def _drain_one(
    queue: MessageQueue,
    simulator,
    policy,
    *,
    worker_id: str,
    state: WorkerHeartbeatState,
    wait_seconds: float,
    model_version: str = "",
    telemetry_stream: TelemetryStream | None = None,
) -> tuple[EpisodeScore, int] | None:
    """Receive and run at most one episode. Returns the score together with
    the message's delivery count (above 1 means the queue redelivered it —
    crash recovery actually firing — which the result carries to the
    coordinator so the run report can count redeliveries). Returns ``None``
    if the queue had nothing to offer within ``wait_seconds``.

    ``queue.receive``/``run_episode`` are both blocking calls (network I/O or
    CPU-bound simulation); running them via ``asyncio.to_thread`` keeps the
    event loop free to send heartbeats while an episode is in flight — a
    multi-minute episode must not make this worker look dead.
    """
    messages = await asyncio.to_thread(queue.receive, max_messages=1, wait_seconds=wait_seconds)
    if not messages:
        return None

    message = messages[0]
    spec = EpisodeSpec.from_dict(message.body)
    state.current_episode_id = spec.episode_id
    state.episode_progress = 0.0

    telemetry_sink = None
    if telemetry_stream is not None:
        # Partitioned by episode_id — see cloud/stream.py's module docstring
        # for why: it's what keeps one episode's frames strictly ordered.
        def telemetry_sink(row: dict) -> None:
            telemetry_stream.put(row["episode_id"], row)

    try:
        score = await asyncio.to_thread(
            run_episode, simulator, spec, policy,
            worker_id=worker_id, model_version=model_version, telemetry_sink=telemetry_sink,
        )
        # Delete only after a result exists, so a crash mid-episode leaves the
        # message for another worker rather than silently losing the episode.
        queue.delete(message.receipt_handle)
        return score, message.receive_count
    finally:
        state.current_episode_id = ""


async def run_worker(
    *,
    worker_id: str,
    coordinator_address: str,
    queue: MessageQueue,
    simulator_backend: str = "kinematic",
    policy_name: str = DEFAULT_POLICY,
    model_version: str | None = None,
    idle_timeout_seconds: float = 5.0,
    receive_wait_seconds: float = 2.0,
    stop_event: asyncio.Event | None = None,
    telemetry_stream: TelemetryStream | None = None,
    policy: Policy | None = None,
) -> list[EpisodeScore]:
    """Register, then loop episodes until the queue drains, the coordinator
    asks this worker to stop, or ``stop_event`` is set (e.g. by a SIGTERM
    handler — a Kubernetes scale-down sends this, and finishing the in-flight
    episode before exiting is strictly cheaper than losing it to redelivery).
    Returns everything this worker completed.

    ``policy`` injects a caller-built Policy instead of building one from
    ``policy_name`` — the same seam the DAgger loop exposes (#15), for tests
    and for Policies whose construction needs more than a name. The caller
    owns label honesty: ``policy_name``/``model_version`` still stamp every
    result, so they must describe what the injected Policy actually is."""
    client = CoordinatorClient.connect(coordinator_address)
    results: list[EpisodeScore] = []
    state = WorkerHeartbeatState()

    registration = await client.register_worker(
        worker_id, hostname=socket.gethostname(), simulator_backend=simulator_backend
    )
    if not registration.accepted:
        raise RuntimeError(f"coordinator rejected registration: {registration.reason}")
    logger.info(
        "worker %s registered for run %s (heartbeat every %.1fs)",
        worker_id, registration.run_id, registration.heartbeat_interval_seconds,
    )
    client.start_heartbeat(worker_id, registration.heartbeat_interval_seconds, state)

    simulator = build_simulator(simulator_backend)
    label = result_label(policy_name, model_version)
    idle_since: float | None = None

    try:
        # After the simulator: a Policy that drives the simulator's own actor
        # needs it to exist first. Anything misconfigured — CARLA's behaviour
        # agent asked to drive the kinematic backend, say — raises here, before
        # a single message is taken off the queue, rather than turning every
        # Episode into a scored failure. Inside the try so teardown still runs.
        if policy is None:
            policy = build_policy(policy_name, simulator=simulator)
        while not client.should_stop and (stop_event is None or not stop_event.is_set()):
            drained = await _drain_one(
                queue, simulator, policy,
                worker_id=worker_id, state=state, wait_seconds=receive_wait_seconds,
                model_version=label, telemetry_stream=telemetry_stream,
            )
            if drained is None:
                # idle_timeout_seconds <= 0 means "never give up" — the shape
                # a Kubernetes Deployment needs (a pod that exits looks like a
                # crash to the controller, which just restarts it into another
                # empty queue forever). idle_timeout_seconds > 0 is batch mode:
                # drain a fixed suite and exit, which is what docker-compose's
                # one-shot demo wants.
                if idle_timeout_seconds <= 0:
                    continue
                loop_time = asyncio.get_running_loop().time()
                if idle_since is None:
                    idle_since = loop_time
                elif loop_time - idle_since >= idle_timeout_seconds:
                    logger.info("worker %s: queue empty for %.1fs, stopping", worker_id, idle_timeout_seconds)
                    break
                continue

            idle_since = None
            score, receive_count = drained
            results.append(score)
            state.episodes_completed += 1
            state.frames_completed += score.frames
            logger.info(
                "worker %s finished %s: driving_score=%.1f frames=%d",
                worker_id, score.episode_id, score.driving_score, score.frames,
            )
            proto_result = episode_score_to_proto(
                score,
                worker_id=worker_id,
                model_version=label,
                simulator_backend=simulator_backend,
                receive_count=receive_count,
            )
            submit_response = await client.submit_result(proto_result)
            if submit_response.duplicate:
                logger.info("coordinator already had a result for %s", score.episode_id)
    finally:
        simulator.close()
        if telemetry_stream is not None:
            # put() only flushes automatically at batch_size records — a
            # worker that finishes with a partial batch buffered would
            # otherwise lose those rows silently.
            written = telemetry_stream.close()
            logger.info("worker %s flushed %d telemetry record(s)", worker_id, written)
        # A worker that finishes quickly can exit before the periodic
        # heartbeat task ever fires (its interval is server-assigned and may
        # be several seconds), leaving the coordinator's last view of this
        # worker stale even though every result was submitted correctly. One
        # last heartbeat before disconnecting closes that gap.
        with contextlib.suppress(grpc.aio.AioRpcError):
            await client.heartbeat_once(
                worker_id,
                frames_completed=state.frames_completed,
                episodes_completed=state.episodes_completed,
            )
        await client.close()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="PathFinder distributed benchmark worker")
    parser.add_argument("--worker-id", type=str, default=f"worker-{uuid.uuid4().hex[:8]}")
    parser.add_argument("--coordinator", type=str, required=True, help="host:port")
    cloud_cli.add_queue_args(parser, backend_default="local", queue_name_default="")
    parser.add_argument("--simulator-backend", choices=["kinematic", "carla", "auto"], default="kinematic")
    parser.add_argument(
        "--policy", choices=list(POLICY_NAMES), default=DEFAULT_POLICY,
        help="Which Policy to score. 'carla_builtin_behavior_agent' is CARLA's own "
        "behaviour agent, kept as a reference baseline rather than as project work; "
        "it needs --simulator-backend carla.",
    )
    parser.add_argument(
        "--model-version", type=str, default=None,
        help="Label recorded against every result. Defaults to --policy; set it "
        "explicitly when the Policy has trained weights whose version its name "
        "does not carry.",
    )
    parser.add_argument("--idle-timeout-seconds", type=float, default=5.0)
    cloud_cli.add_telemetry_args(
        parser,
        backend_help="Stream per-frame telemetry as episodes run. 'none' skips it entirely "
        "(cheapest — no telemetry work at all).",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    queue = cloud_cli.queue_from_args(args)
    telemetry_stream = cloud_cli.stream_from_args(args)

    results = asyncio.run(_run_with_signal_handling(args, queue, telemetry_stream))
    logger.info("worker %s completed %d episodes", args.worker_id, len(results))


async def _run_with_signal_handling(
    args: argparse.Namespace, queue: MessageQueue, telemetry_stream: TelemetryStream | None
) -> list[EpisodeScore]:
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:
            pass  # add_signal_handler is unix-only; fine, this is a container workload.

    return await run_worker(
        worker_id=args.worker_id,
        coordinator_address=args.coordinator,
        queue=queue,
        simulator_backend=args.simulator_backend,
        policy_name=args.policy,
        model_version=args.model_version,
        idle_timeout_seconds=args.idle_timeout_seconds,
        stop_event=stop_event,
        telemetry_stream=telemetry_stream,
    )


if __name__ == "__main__":
    main()
