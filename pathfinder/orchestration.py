
"""
Multi-worker benchmark orchestration over a work queue.

The problem this solves
-----------------------
A benchmark is hundreds of CARLA episodes. Each takes minutes, each needs its
own simulator instance, and simulators crash. Running them sequentially takes
hours; pre-sharding the list across N workers loses a shard whenever a worker
dies.

So episodes go on a queue and workers pull. This gives three properties that
static sharding cannot:

* **Automatic load balancing.** Episodes vary in length by an order of
  magnitude. A fast worker simply pulls more, with no coordination.
* **Crash recovery.** A dead worker's in-flight episode reappears after its
  visibility timeout and another worker takes it.
* **Elastic scale.** Workers can join a run already in progress.

Idempotency
-----------
Delivery is at-least-once, so an episode can run twice — most often when a
worker is merely slow, not dead, and its visibility timeout expires. Results are
therefore keyed by ``episode_id`` and a duplicate **overwrites** rather than
appends. Getting this wrong inflates the episode count and silently biases every
aggregate, which is precisely the kind of bug that survives to publication.
"""
from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field

from pathfinder.cloud.queue import MessageQueue
from pathfinder.metrics.driving_score import BenchmarkSummary, EpisodeScore, aggregate
from pathfinder.planners import DEFAULT_PLANNER, build_planner, result_label
from pathfinder.runner import Planner, run_episode
from pathfinder.sim.base import EpisodeSpec
from pathfinder.sim.carla_backend import build_simulator

logger = logging.getLogger(__name__)

__all__ = ["BenchmarkCoordinator", "BenchmarkRun", "EpisodeWorker", "build_episode_specs"]


def build_episode_specs(
    *,
    count: int,
    towns: tuple[str, ...] = ("Town01", "Town03", "Town05"),
    weathers: tuple[str, ...] = ("ClearNoon", "WetNoon", "HardRainNoon", "ClearSunset"),
    route_length_m: float = 400.0,
    base_seed: int = 1000,
    max_steps: int = 1500,
) -> list[EpisodeSpec]:
    """Build a deterministic benchmark suite.

    Town and weather cycle at different rates (a prime-ish stride) so the suite
    covers combinations rather than pairing town *i* with weather *i* forever.
    Every spec carries its own seed, so the suite is reproducible and any single
    episode can be re-run in isolation to debug it.

    Raises:
        ValueError: If ``count`` is not positive.
    """
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")
    return [
        EpisodeSpec(
            episode_id=f"ep-{index:04d}",
            town=towns[index % len(towns)],
            weather=weathers[(index * 3) % len(weathers)],
            route_length_m=route_length_m,
            seed=base_seed + index,
            max_steps=max_steps,
        )
        for index in range(count)
    ]


@dataclass
class BenchmarkRun:
    """Outcome of a distributed benchmark."""

    run_id: str
    scores: dict[str, EpisodeScore] = field(default_factory=dict)
    duplicates: int = 0
    dead_letters: list[dict] = field(default_factory=list)
    wall_seconds: float = 0.0
    workers: int = 0

    @property
    def summary(self) -> BenchmarkSummary:
        return aggregate(list(self.scores.values()))

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "episodes_completed": len(self.scores),
            "duplicate_deliveries": self.duplicates,
            "dead_letters": len(self.dead_letters),
            "wall_seconds": round(self.wall_seconds, 2),
            "workers": self.workers,
            "summary": self.summary.to_dict(),
        }


class EpisodeWorker:
    """Pulls episodes off the queue and runs them until the queue drains.

    One simulator instance per worker, created once and reused across episodes:
    CARLA world loading dominates episode cost, so constructing a simulator per
    episode would make the benchmark measure startup rather than driving.
    """

    def __init__(
        self,
        worker_id: str,
        queue: MessageQueue,
        *,
        planner_name: str = DEFAULT_PLANNER,
        simulator_backend: str = "auto",
        telemetry_sink=None,
        model_version: str | None = None,
        dataset_version: str = "",
        idle_timeout_seconds: float = 2.0,
    ) -> None:
        self.worker_id = worker_id
        self.queue = queue
        self.planner_name = planner_name
        self.simulator_backend = simulator_backend
        self.telemetry_sink = telemetry_sink
        self.model_version = result_label(planner_name, model_version)
        self.dataset_version = dataset_version
        self.idle_timeout_seconds = idle_timeout_seconds

        self.episodes_run = 0
        self.results: list[EpisodeScore] = []
        self._stop = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def run(self) -> list[EpisodeScore]:
        """Consume episodes until the queue is empty for ``idle_timeout_seconds``."""
        simulator = build_simulator(self.simulator_backend)
        idle_since: float | None = None

        try:
            # After the simulator, because a Policy that drives the simulator's
            # own actor needs it to exist first — and inside the try, so a
            # mismatched pair still restores a CARLA server to async mode on the
            # way out. Anything misconfigured raises here, before a single
            # message leaves the queue.
            planner: Planner = build_planner(self.planner_name, simulator=simulator)
            while not self._stop.is_set():
                messages = self.queue.receive(max_messages=1, wait_seconds=0.2)
                if not messages:
                    # Drain detection: the queue may briefly look empty while
                    # other workers hold messages in flight, so require it to
                    # stay empty for a window before concluding the run is done.
                    if idle_since is None:
                        idle_since = time.monotonic()
                    elif time.monotonic() - idle_since >= self.idle_timeout_seconds:
                        break
                    continue

                idle_since = None
                message = messages[0]
                spec = EpisodeSpec.from_dict(message.body)

                try:
                    score = run_episode(
                        simulator,
                        spec,
                        planner,
                        telemetry_sink=self.telemetry_sink,
                        worker_id=self.worker_id,
                        model_version=self.model_version,
                        dataset_version=self.dataset_version,
                    )
                    self.results.append(score)
                    self.episodes_run += 1
                    # Acknowledge only after a result exists. Deleting on
                    # receipt would drop the episode if the worker died mid-run,
                    # which is the exact failure the queue is here to survive.
                    self.queue.delete(message.receipt_handle)
                except Exception as error:
                    # Do not delete: let the visibility timeout redeliver it.
                    # After max_receive_count it lands in the DLQ, so a reliably
                    # crashing episode cannot stall the run forever.
                    logger.error(
                        "worker %s failed episode %s (leaving for redelivery): %s",
                        self.worker_id, spec.episode_id, error,
                    )
        finally:
            simulator.close()
        return self.results


class BenchmarkCoordinator:
    """Enqueues a suite, runs workers concurrently, and aggregates results."""

    def __init__(
        self,
        queue: MessageQueue,
        *,
        planner_name: str = DEFAULT_PLANNER,
        simulator_backend: str = "auto",
        telemetry_sink=None,
        model_version: str | None = None,
        dataset_version: str = "",
    ) -> None:
        self.queue = queue
        self.planner_name = planner_name
        self.simulator_backend = simulator_backend
        self.telemetry_sink = telemetry_sink
        self.model_version = result_label(planner_name, model_version)
        self.dataset_version = dataset_version

    def run(self, specs: list[EpisodeSpec], *, workers: int = 4) -> BenchmarkRun:
        """Run a suite across ``workers`` concurrent workers.

        Raises:
            ValueError: If ``workers`` is not positive or ``specs`` is empty.
        """
        if not specs:
            raise ValueError("cannot run a benchmark with no episodes")
        if workers <= 0:
            raise ValueError(f"workers must be positive, got {workers}")

        run = BenchmarkRun(run_id=uuid.uuid4().hex[:12], workers=workers)
        for spec in specs:
            self.queue.send(spec.to_dict())
        logger.info("enqueued %d episodes for run %s", len(specs), run.run_id)

        instances = [
            EpisodeWorker(
                f"worker-{index}",
                self.queue,
                planner_name=self.planner_name,
                simulator_backend=self.simulator_backend,
                telemetry_sink=self.telemetry_sink,
                model_version=self.model_version,
                dataset_version=self.dataset_version,
            )
            for index in range(workers)
        ]

        started = time.perf_counter()
        threads = [
            threading.Thread(target=worker.run, name=worker.worker_id, daemon=True)
            for worker in instances
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        run.wall_seconds = time.perf_counter() - started

        # Key by episode_id so an at-least-once duplicate overwrites instead of
        # inflating the count and biasing every aggregate.
        for worker in instances:
            for score in worker.results:
                if score.episode_id in run.scores:
                    run.duplicates += 1
                run.scores[score.episode_id] = score

        run.dead_letters = self.queue.dead_letters()
        logger.info(
            "run %s finished: %d episodes in %.1fs across %d workers (%d duplicates, %d dead-lettered)",
            run.run_id, len(run.scores), run.wall_seconds, workers,
            run.duplicates, len(run.dead_letters),
        )
        return run
