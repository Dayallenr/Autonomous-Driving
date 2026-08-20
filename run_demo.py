#!/usr/bin/env python
"""
PathFinder end-to-end demo.

    python run_demo.py

Runs the whole stack against local backends — no CARLA, no AWS, no Docker.

Stages
------
1. Simulator     — kinematic backend, forward-view rendering
2. Orchestration — episodes over an SQS-semantics queue, concurrent workers
3. Resilience    — worker crash recovery and dead-letter handling
4. Telemetry     — Kinesis-shaped stream → partitioned Parquet → analytics query
5. Datasets      — content-addressed versioning, digest-verified download
6. Training      — a SageMaker-contract job run locally
7. Perception    — the KITTI detector's measured throughput
8. DDL           — the Redshift/Athena schema the telemetry lands in

Every number printed is measured on this machine. Where a number is not
comparable to a published one — kinematic backend rather than CARLA, this CPU/GPU
rather than a benchmark rig — the demo says so instead of leaving it implied.
"""
from __future__ import annotations

import json
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

BOLD, DIM, RESET = "\033[1m", "\033[2m", "\033[0m"
GREEN, YELLOW, CYAN = "\033[32m", "\033[33m", "\033[36m"


def header(number: int, title: str) -> None:
    print(f"\n{BOLD}{'─' * 76}{RESET}")
    print(f"{BOLD} {number}. {title}{RESET}")
    print(f"{BOLD}{'─' * 76}{RESET}\n")


def note(message: str) -> None:
    print(f"   {DIM}{message}{RESET}")


def stage_simulator() -> None:
    header(1, "Simulator — kinematic backend with forward-view rendering")

    from pathfinder.runner import PurePursuitPolicy, run_episode
    from pathfinder.sim import EpisodeSpec, KinematicSimulator, carla_available

    print(f"   CARLA available on this machine: {carla_available()}")
    if not carla_available():
        note("CARLA does not ship for Apple Silicon and needs a GPU, so the")
        note("kinematic backend runs instead. It exercises orchestration,")
        note("telemetry, and scoring — not perception or CARLA physics.")

    simulator = KinematicSimulator(render=True)
    spec = EpisodeSpec(episode_id="demo-0", route_length_m=400, max_steps=1500, seed=42)
    score = run_episode(simulator, spec, PurePursuitPolicy())
    simulator.close()

    print(f"\n   episode {spec.episode_id}: {spec.town} / {spec.weather} / {spec.route_length_m:.0f} m")
    for key in ("driving_score", "route_completion", "collisions", "lane_invasions",
                "red_lights_run", "frames", "termination_reason"):
        print(f"     {key:22} {score.to_dict()[key]}")

    # Determinism is what makes a distributed benchmark reproducible.
    simulator = KinematicSimulator()
    repeat = run_episode(simulator, spec, PurePursuitPolicy())
    simulator.close()
    identical = abs(repeat.driving_score - score.driving_score) < 1e-9
    print(f"\n   re-running the same seed gives an identical score: "
          f"{GREEN if identical else YELLOW}{identical}{RESET}")
    note("episodes are a pure function of their seed, so any result can be bisected")


def stage_orchestration(telemetry_stream) -> tuple[object, float]:
    header(2, "Orchestration — episodes over a queue, concurrent workers")

    from pathfinder.cloud import LocalMessageQueue
    from pathfinder.orchestration import BenchmarkCoordinator, build_episode_specs

    specs = build_episode_specs(count=24, route_length_m=400, max_steps=1500)
    note(f"{len(specs)} episodes across "
         f"{len({s.town for s in specs})} towns and {len({s.weather for s in specs})} weathers")

    def sink(row: dict) -> None:
        # Partition by episode so one episode's frames stay strictly ordered.
        telemetry_stream.put(row["episode_id"], row)

    results = {}
    for worker_count in (1, 4):
        queue = LocalMessageQueue(visibility_timeout=60)
        coordinator = BenchmarkCoordinator(
            queue,
            simulator_backend="kinematic",
            telemetry_sink=sink if worker_count == 4 else None,
        )
        started = time.perf_counter()
        run = coordinator.run(specs, workers=worker_count)
        results[worker_count] = (run, time.perf_counter() - started)

    print(f"\n   {'workers':>9} {'wall (s)':>10} {'episodes':>10} {'speedup':>9}")
    print(f"   {'-' * 42}")
    baseline = results[1][1]
    for worker_count, (run, elapsed) in results.items():
        print(f"   {worker_count:>9} {elapsed:>10.2f} {len(run.scores):>10} "
              f"{baseline / elapsed:>8.2f}x")

    # Read that speedup column honestly: it is below 1.0, and it should be.
    note("speedup here is <1x, and that is the expected result — not a bug to")
    note("explain away. These workers are threads, and the kinematic backend is")
    note("pure-Python CPU work, so the GIL serialises them and adds contention.")
    note("What this stage proves is correctness under concurrency — every")
    note("episode runs exactly once across contending workers, and results")
    note("survive redelivery. Throughput scaling needs workers in separate")
    note("processes (in deployment, separate pods), which is a different axis.")

    run, elapsed = results[4]
    summary = run.summary.to_dict()
    print(f"\n   {BOLD}benchmark result (4 workers){RESET}")
    for key in ("episodes", "driving_score", "route_completion", "infraction_penalty",
                "collisions_per_km", "total_distance_km", "failures"):
        print(f"     {key:22} {summary[key]}")
    print(f"     {'infractions':22} {json.dumps(summary['infraction_totals'])}")

    print(f"\n   duplicate deliveries: {run.duplicates}   dead-lettered: {len(run.dead_letters)}")
    note("results are keyed by episode_id, so an at-least-once duplicate")
    note("overwrites rather than inflating the episode count")
    return run, elapsed


def stage_resilience() -> None:
    header(3, "Resilience — crash recovery and poison-message handling")

    from pathfinder.cloud import LocalMessageQueue

    # A worker takes a message and dies without acknowledging it.
    queue = LocalMessageQueue(visibility_timeout=0.4, max_receive_count=3)
    queue.send({"episode_id": "ep-crash"})

    taken = queue.receive(max_messages=1)[0]
    print(f"   worker A received {taken.body['episode_id']} (delivery {taken.receive_count})")
    print(f"   queue depth while in flight: {queue.approximate_depth()} "
          f"(hidden, not lost)")
    print("   ...worker A dies without acknowledging...")

    time.sleep(0.5)  # visibility timeout expires
    redelivered = queue.receive(max_messages=1)[0]
    print(f"   worker B received {redelivered.body['episode_id']} "
          f"(delivery {redelivered.receive_count}) {GREEN}recovered{RESET}")
    queue.delete(redelivered.receipt_handle)
    print(f"   after acknowledgement, depth: {queue.approximate_depth()}")

    # A message that always fails must not stall the run forever.
    print(f"\n   {DIM}poison message: fails on every delivery{RESET}")
    poison_queue = LocalMessageQueue(visibility_timeout=0.05, max_receive_count=3)
    poison_queue.send({"episode_id": "ep-poison"})
    deliveries = 0
    for _ in range(8):
        messages = poison_queue.receive(max_messages=1)
        if not messages:
            time.sleep(0.06)
            continue
        deliveries += 1  # never acknowledged, simulating repeated failure
        time.sleep(0.06)
    print(f"   delivered {deliveries} times, then dead-lettered: "
          f"{len(poison_queue.dead_letters())}")
    note("without a delivery budget, one broken episode consumes a worker forever")


def stage_telemetry(telemetry_stream, workspace: Path) -> None:
    header(4, "Telemetry — stream → partitioned Parquet → analytics query")

    from pathfinder.cloud import LocalObjectStore, TelemetryWarehouse

    telemetry_stream.flush()
    partitions = telemetry_stream.partitions()
    print(f"   stream: {telemetry_stream.records_written:,} records "
          f"across {len(partitions)} partitions (one per episode)")
    note("partitioning by episode_id keeps each episode's frames strictly ordered")

    store = LocalObjectStore(workspace / "s3")
    warehouse = TelemetryWarehouse(store)

    started = time.perf_counter()
    keys = warehouse.drain_stream(telemetry_stream, limit=200_000)
    elapsed = time.perf_counter() - started

    print(f"\n   wrote {warehouse.rows_written:,} rows to {warehouse.files_written} "
          f"Parquet file(s) in {elapsed:.2f}s")
    for key in keys[:3]:
        print(f"     {key}")

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        note("pyarrow not installed — skipping read-back")
        return

    tables = [pq.read_table(workspace / "s3" / key) for key in keys]
    combined = pa.concat_tables(tables)
    print(f"\n   read back {combined.num_rows:,} rows, {combined.num_columns} columns")

    # The kind of query the offline-regression dashboard runs.
    print(f"\n   {DIM}query: policy latency and speed by navigation command{RESET}")
    commands = combined.column("command").to_pylist()
    latencies = combined.column("policy_latency_ms").to_pylist()
    speeds = combined.column("speed_mps").to_pylist()
    names = {0: "FOLLOW_LANE", 1: "TURN_LEFT", 2: "TURN_RIGHT", 3: "GO_STRAIGHT"}

    buckets: dict[int, list[tuple[float, float]]] = {}
    for command, latency, speed in zip(commands, latencies, speeds, strict=True):
        buckets.setdefault(command, []).append((latency, speed))

    print(f"     {'command':<14}{'frames':>9}{'mean ms':>10}{'mean m/s':>10}")
    for command in sorted(buckets):
        rows = buckets[command]
        print(f"     {names.get(command, command):<14}{len(rows):>9}"
              f"{sum(r[0] for r in rows) / len(rows):>10.3f}"
              f"{sum(r[1] for r in rows) / len(rows):>10.2f}")


def stage_datasets(workspace: Path):
    header(5, "Datasets — content-addressed versioning with digest verification")

    from pathfinder.cloud import DatasetRegistry, LocalObjectStore

    registry = DatasetRegistry(LocalObjectStore(workspace / "s3"))

    files = {f"frame_{i:03d}.txt": f"synthetic sample {i}".encode() for i in range(20)}
    first = registry.publish("kitti-mini", files, metadata={"note": "demo"})
    print(f"   published kitti-mini@{first.version} "
          f"({first.file_count} files, {first.total_bytes} bytes)")

    # Republishing identical content must not create a new version.
    again = registry.publish("kitti-mini", files)
    print(f"   republishing identical content -> {again.version} "
          f"{GREEN}(same version, no duplicate){RESET}")

    # One changed byte must produce a different version.
    changed = dict(files)
    changed["frame_000.txt"] = b"synthetic sample 0 (edited)"
    second = registry.publish("kitti-mini", changed)
    print(f"   one byte changed          -> {second.version} "
          f"{YELLOW}(new version){RESET}")
    note("a training job records the version it consumed, so 'what data produced")
    note("these weights?' is answerable rather than a matter of memory")

    local = registry.download(first, workspace / "materialized")
    print(f"\n   downloaded and digest-verified {first.file_count} files to {local.name}/")
    return registry, first


def stage_training(registry, dataset, workspace: Path) -> None:
    header(6, "Training — a SageMaker-contract job, run locally")

    from pathfinder.cloud import LocalTrainingJobRunner

    entry_point = workspace / "train_entry.py"
    entry_point.write_text(
        '''"""Training entry point honouring the SageMaker container contract."""
import json
import os
from pathlib import Path

channel = Path(os.environ["SM_CHANNEL_TRAINING"])
model_dir = Path(os.environ["SM_MODEL_DIR"])
hyperparameters = json.loads(Path(os.environ["SM_HYPERPARAMETERS"]).read_text())

files = sorted(channel.rglob("*.txt"))
print(f"read {len(files)} training files from {channel}")
print(f"hyperparameters: {hyperparameters}")
print(f"dataset version: {os.environ.get('PATHFINDER_DATASET_VERSION')}")

model_dir.mkdir(parents=True, exist_ok=True)
(model_dir / "model.json").write_text(json.dumps({
    "trained_on_files": len(files),
    "hyperparameters": hyperparameters,
    "dataset_version": os.environ.get("PATHFINDER_DATASET_VERSION", ""),
}))
print("wrote model artifact")
''',
        encoding="utf-8",
    )

    runner = LocalTrainingJobRunner(registry, workspace=workspace / "job")
    job = runner.fit(
        job_name="cil-demo",
        entry_point=entry_point,
        dataset=dataset,
        hyperparameters={"epochs": 3, "learning_rate": 3e-4, "batch_size": 64},
        timeout_seconds=120,
    )

    for key, value in job.to_dict().items():
        if key != "hyperparameters":
            print(f"   {key:20} {value}")
    print(f"\n   {DIM}entry-point output:{RESET}")
    for line in job.log_tail:
        print(f"     {line}")

    note("the script reads files from a directory and writes to SM_MODEL_DIR —")
    note("the same script runs unchanged on real SageMaker, which is the only")
    note("property that makes this abstraction worth having")

    # A failing job must be reported, not raised.
    failing = workspace / "fail_entry.py"
    failing.write_text("import sys\nprint('deliberate failure')\nsys.exit(3)\n", encoding="utf-8")
    failed = runner.fit(job_name="cil-fail", entry_point=failing, timeout_seconds=30)
    print(f"\n   failing job: status={failed.status} exit={failed.exit_code} "
          f"reason={failed.failure_reason!r}")


def stage_perception() -> None:
    header(7, "Perception — KITTI detector throughput")

    from pathfinder.benchmark_detector import (
        DEFAULT_MODEL,
        load_model,
        run_throughput,
        select_device,
    )

    if not DEFAULT_MODEL.exists():
        print(f"   {YELLOW}no perception model at {DEFAULT_MODEL} — skipping{RESET}")
        note("train one with the KITTI pipeline (see README.md)")
        return

    device = select_device("auto")
    try:
        model = load_model(DEFAULT_MODEL, device)
    except ImportError as error:
        print(f"   {YELLOW}{error}{RESET}")
        return

    print(f"   model: {DEFAULT_MODEL.name}  device: {device}")
    print(f"   classes: {', '.join(model.names.values())}")
    print(f"\n   {DIM}measuring (this takes ~20s)...{RESET}")

    throughput = run_throughput(
        model, device=device, batch_size=1, input_size=640, iterations=20, warmup=5
    )
    print(f"\n   {CYAN}{throughput.summary()}{RESET}")
    note("FPS is a property of (model, hardware, batch, resolution, precision).")
    note("This is not comparable to a figure measured on a discrete NVIDIA GPU")
    note("with TensorRT, which is where a 30+ FPS number would come from.")


def stage_ddl() -> None:
    header(8, "DDL — generated from the same column definitions as the Parquet")

    from pathfinder.cloud import athena_ddl, redshift_ddl

    for line in redshift_ddl().splitlines():
        if any(token in line for token in ("CREATE TABLE", "DISTKEY", "SORTKEY", "DISTSTYLE ALL")):
            print(f"     {line}")
    print()
    for line in athena_ddl().splitlines():
        if any(token in line for token in ("PARTITIONED BY", "STORED AS", "LOCATION")):
            print(f"     {line}")
    note("Arrow schema, Redshift DDL, and Athena DDL are projections of one")
    note("column list, so a COPY cannot silently null a drifted column")


def main() -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    print(f"\n{BOLD}{'═' * 76}{RESET}")
    print(f"{BOLD}  PATHFINDER — end-to-end demo{RESET}")
    print(f"{BOLD}{'═' * 76}{RESET}")
    note("local backends only: no CARLA, no AWS, no Docker")

    workspace = Path(tempfile.mkdtemp(prefix="pathfinder-demo-"))
    from pathfinder.cloud import LocalTelemetryStream

    telemetry_stream = LocalTelemetryStream(workspace / "kinesis")

    try:
        stage_simulator()
        stage_orchestration(telemetry_stream)
        stage_resilience()
        stage_telemetry(telemetry_stream, workspace)
        registry, dataset = stage_datasets(workspace)
        stage_training(registry, dataset, workspace)
        stage_perception()
        stage_ddl()

        print(f"\n{BOLD}{'═' * 76}{RESET}")
        print(f"{BOLD}  demo complete{RESET}")
        print(f"{BOLD}{'═' * 76}{RESET}")
        print(
            "\nDriving-quality numbers above come from the kinematic backend and are\n"
            "not comparable to CARLA leaderboard results. The orchestration,\n"
            "telemetry, dataset, and training numbers are real measurements of the\n"
            "pipeline on this machine.\n"
        )
        return 0
    finally:
        telemetry_stream.close()
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
