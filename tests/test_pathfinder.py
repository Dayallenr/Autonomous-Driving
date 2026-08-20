"""
PathFinder test suite: cloud abstractions, simulator, scoring, orchestration.

The queue tests deliberately exercise *timing* behaviour (visibility timeouts,
redelivery, dead-lettering) rather than just the happy path. Those semantics are
the entire reason for using a queue, and a test suite that only checks
send/receive would pass against a plain list while losing work on every crash.
"""
from __future__ import annotations

import threading
import time
from datetime import UTC, datetime

import numpy as np
import pytest

from pathfinder.cloud import (
    DatasetRegistry,
    LocalMessageQueue,
    LocalObjectStore,
    LocalTelemetryStream,
    LocalTrainingJobRunner,
    TelemetryWarehouse,
    athena_ddl,
    build_queue,
    build_runner,
    build_store,
    build_stream,
    redshift_ddl,
)
from pathfinder.cloud.warehouse import EPISODE_COLUMNS, FRAME_COLUMNS
from pathfinder.metrics.detection import (
    BoxMatch,
    average_precision,
    compute_map,
    iou_matrix,
    measure_throughput,
)
from pathfinder.metrics.driving_score import (
    INFRACTION_PENALTIES,
    aggregate,
    score_episode,
)
from pathfinder.orchestration import BenchmarkCoordinator, build_episode_specs
from pathfinder.runner import PurePursuitPolicy, run_episode
from pathfinder.sim import EpisodeSpec, Infraction, KinematicSimulator, build_simulator
from pathfinder.sim.render import RENDER_HEIGHT, RENDER_WIDTH

# ─────────────────────────────────────────────────────────────────────────────
# Queue — SQS semantics
# ─────────────────────────────────────────────────────────────────────────────


def test_queue_roundtrip():
    queue = LocalMessageQueue()
    queue.send({"episode_id": "a"})
    messages = queue.receive(max_messages=1)
    assert len(messages) == 1 and messages[0].body["episode_id"] == "a"
    queue.delete(messages[0].receipt_handle)
    assert queue.approximate_depth() == 0


def test_received_message_is_hidden_not_deleted():
    """The distinction that makes crash recovery possible."""
    queue = LocalMessageQueue(visibility_timeout=10.0)
    queue.send({"episode_id": "a"})
    queue.receive(max_messages=1)
    assert queue.approximate_depth() == 0, "should be hidden"
    assert queue.in_flight() == 1, "but still present"


def test_unacknowledged_message_is_redelivered():
    """A worker that dies mid-episode must not lose the episode."""
    queue = LocalMessageQueue(visibility_timeout=0.1)
    queue.send({"episode_id": "a"})
    first = queue.receive(max_messages=1)[0]
    assert first.receive_count == 1

    time.sleep(0.15)
    second = queue.receive(max_messages=1)[0]
    assert second.receive_count == 2
    assert second.body["episode_id"] == "a"
    # Receipt handles change on every receive, so the stale one must not delete.
    assert first.receipt_handle != second.receipt_handle


def test_poison_message_goes_to_dead_letter_queue():
    """Without a delivery budget, one broken episode stalls a worker forever."""
    queue = LocalMessageQueue(visibility_timeout=0.02, max_receive_count=2)
    queue.send({"episode_id": "poison"})
    for _ in range(6):
        queue.receive(max_messages=1)
        time.sleep(0.03)
    assert len(queue.dead_letters()) == 1
    assert queue.dead_letters()[0]["episode_id"] == "poison"
    assert queue.approximate_depth() == 0


def test_deleting_unknown_receipt_handle_is_not_an_error():
    """Happens legitimately when a slow worker finishes after redelivery."""
    LocalMessageQueue().delete("not-a-real-handle")


def test_queue_is_thread_safe():
    """Workers run concurrently, so every message must go to exactly one."""
    queue = LocalMessageQueue(visibility_timeout=30.0)
    for index in range(200):
        queue.send({"n": index})

    received: list[int] = []
    lock = threading.Lock()

    def consume() -> None:
        while True:
            messages = queue.receive(max_messages=1)
            if not messages:
                return
            with lock:
                received.append(messages[0].body["n"])
            queue.delete(messages[0].receipt_handle)

    threads = [threading.Thread(target=consume) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(received) == list(range(200)), "message delivered zero or twice"


def test_long_poll_waits_for_a_message():
    queue = LocalMessageQueue()
    started = time.monotonic()
    assert queue.receive(max_messages=1, wait_seconds=0.2) == []
    assert time.monotonic() - started >= 0.18


def test_build_queue_validates_backend():
    with pytest.raises(ValueError, match="unknown queue backend"):
        build_queue("rabbitmq")
    with pytest.raises(ValueError, match="queue_url"):
        build_queue("sqs")


def test_queue_rejects_invalid_config():
    with pytest.raises(ValueError, match="visibility_timeout"):
        LocalMessageQueue(visibility_timeout=0)
    with pytest.raises(ValueError, match="max_receive_count"):
        LocalMessageQueue(max_receive_count=0)


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry stream
# ─────────────────────────────────────────────────────────────────────────────


def test_stream_preserves_order_within_a_partition(tmp_path):
    """Kinesis guarantees ordering within a shard; episodes rely on it."""
    stream = LocalTelemetryStream(tmp_path)
    for index in range(50):
        stream.put("ep-1", {"frame_index": index})
    stream.flush()
    records = stream.read(partition_key="ep-1")
    assert [r.data["frame_index"] for r in records] == list(range(50))


def test_stream_separates_partitions(tmp_path):
    stream = LocalTelemetryStream(tmp_path)
    stream.put("ep-1", {"v": 1})
    stream.put("ep-2", {"v": 2})
    stream.flush()
    assert set(stream.partitions()) == {"ep-1", "ep-2"}
    assert len(stream.read(partition_key="ep-1")) == 1


def test_stream_serializes_datetimes(tmp_path):
    """Regression test: telemetry rows carry timestamps, and a JSON encoder that
    cannot handle them silently drops every row — the producer treats telemetry
    failures as non-fatal, so the symptom is an empty warehouse, not a crash."""
    stream = LocalTelemetryStream(tmp_path)
    stream.put("ep-1", {"frame_index": 0, "timestamp": datetime.now(UTC)})
    stream.flush()
    records = stream.read(partition_key="ep-1")
    assert len(records) == 1
    assert isinstance(records[0].data["timestamp"], str)


def test_stream_rejects_oversized_record(tmp_path):
    """Enforced locally so it fails fast rather than in the cloud."""
    stream = LocalTelemetryStream(tmp_path)
    with pytest.raises(ValueError, match="over the"):
        stream.put("ep-1", {"blob": "x" * (1024 * 1024 + 10)})


def test_stream_auto_flushes_on_batch_size(tmp_path):
    stream = LocalTelemetryStream(tmp_path, batch_size=10)
    for index in range(10):
        stream.put("ep-1", {"i": index})
    assert stream.records_written == 10, "should have flushed without an explicit call"


def test_build_stream_validates_backend(tmp_path):
    with pytest.raises(ValueError, match="unknown telemetry backend"):
        build_stream("pubsub", local_root=tmp_path)


# ─────────────────────────────────────────────────────────────────────────────
# Object store and dataset versioning
# ─────────────────────────────────────────────────────────────────────────────


def test_object_store_roundtrip(tmp_path):
    store = LocalObjectStore(tmp_path)
    store.put("a/b.txt", b"hello")
    assert store.get("a/b.txt") == b"hello"
    assert store.list("a/") == ["a/b.txt"]


def test_object_store_rejects_traversal(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        LocalObjectStore(tmp_path).put("../evil", b"x")


def test_build_store_validates_backend(tmp_path):
    with pytest.raises(ValueError, match="unknown object store backend"):
        build_store("gcs", local_root=tmp_path)


def test_dataset_version_is_content_addressed(tmp_path):
    """Identical content must yield an identical version, or 'did the data
    change?' becomes an investigation instead of a comparison."""
    registry = DatasetRegistry(LocalObjectStore(tmp_path))
    files = {"a.txt": b"one", "b.txt": b"two"}
    first = registry.publish("ds", files)
    second = registry.publish("ds", dict(files))
    assert first.version == second.version


def test_changed_byte_yields_new_version(tmp_path):
    registry = DatasetRegistry(LocalObjectStore(tmp_path))
    first = registry.publish("ds", {"a.txt": b"one"})
    second = registry.publish("ds", {"a.txt": b"onE"})
    assert first.version != second.version
    assert set(registry.list_versions("ds")) == {first.version, second.version}


def test_dataset_resolve_latest_follows_pointer(tmp_path):
    registry = DatasetRegistry(LocalObjectStore(tmp_path))
    registry.publish("ds", {"a.txt": b"one"})
    newest = registry.publish("ds", {"a.txt": b"two"})
    assert registry.resolve("ds", "latest").version == newest.version


def test_dataset_download_verifies_digests(tmp_path):
    """A truncated download trains a model on corrupt data with no error; the
    only symptom is a metric that is quietly worse than it should be."""
    store = LocalObjectStore(tmp_path / "store")
    registry = DatasetRegistry(store)
    manifest = registry.publish("ds", {"a.txt": b"original"})

    # Corrupt the stored object behind the registry's back.
    store.put(f"datasets/ds/versions/{manifest.version}/data/a.txt", b"tampered")
    with pytest.raises(ValueError, match="failed verification"):
        registry.download(manifest, tmp_path / "out")


def test_publishing_empty_dataset_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="empty dataset"):
        DatasetRegistry(LocalObjectStore(tmp_path)).publish("ds", {})


# ─────────────────────────────────────────────────────────────────────────────
# Training jobs
# ─────────────────────────────────────────────────────────────────────────────


def test_local_training_job_honours_sagemaker_contract(tmp_path):
    """The only property that makes this abstraction worth having: a script
    written for SageMaker runs unchanged locally."""
    store = LocalObjectStore(tmp_path / "store")
    registry = DatasetRegistry(store)
    dataset = registry.publish("ds", {f"f{i}.txt": b"data" for i in range(5)})

    entry = tmp_path / "entry.py"
    entry.write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        "channel = Path(os.environ['SM_CHANNEL_TRAINING'])\n"
        "model_dir = Path(os.environ['SM_MODEL_DIR'])\n"
        "hp = json.loads(Path(os.environ['SM_HYPERPARAMETERS']).read_text())\n"
        "assert len(list(channel.rglob('*.txt'))) == 5, 'channel not materialized'\n"
        "assert hp['epochs'] == 7, 'hyperparameters not delivered'\n"
        "model_dir.mkdir(parents=True, exist_ok=True)\n"
        "(model_dir / 'model.bin').write_text('trained')\n",
        encoding="utf-8",
    )

    job = LocalTrainingJobRunner(registry, workspace=tmp_path / "job").fit(
        job_name="t", entry_point=entry, dataset=dataset, hyperparameters={"epochs": 7}
    )
    assert job.succeeded, job.failure_reason + "\n" + "\n".join(job.log_tail)
    assert job.dataset_version == dataset.version


def test_failing_training_job_is_reported_not_raised(tmp_path):
    entry = tmp_path / "fail.py"
    entry.write_text("import sys\nsys.exit(4)\n", encoding="utf-8")
    job = LocalTrainingJobRunner(workspace=tmp_path / "job").fit(
        job_name="t", entry_point=entry
    )
    assert not job.succeeded
    assert job.exit_code == 4


def test_training_job_timeout_is_reported(tmp_path):
    entry = tmp_path / "slow.py"
    entry.write_text("import time\ntime.sleep(30)\n", encoding="utf-8")
    job = LocalTrainingJobRunner(workspace=tmp_path / "job").fit(
        job_name="t", entry_point=entry, timeout_seconds=0.5
    )
    assert not job.succeeded and "timeout" in job.failure_reason


def test_missing_entry_point_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        LocalTrainingJobRunner().fit(job_name="t", entry_point=tmp_path / "nope.py")


def test_sagemaker_runner_validates_spot_configuration():
    """SageMaker rejects max_wait < max_run with an opaque message; catching it
    here turns a confusing API error into a clear one."""
    with pytest.raises(ValueError, match="max_wait_seconds"):
        build_runner(
            "sagemaker",
            role_arn="arn:aws:iam::1:role/r",
            image_uri="img",
            output_path="s3://b/",
            max_run_seconds=7200,
            max_wait_seconds=3600,
        )


def test_build_runner_validates_backend():
    with pytest.raises(ValueError, match="unknown training backend"):
        build_runner("vertex")
    with pytest.raises(ValueError, match="sagemaker backend requires"):
        build_runner("sagemaker")


# ─────────────────────────────────────────────────────────────────────────────
# Driving score
# ─────────────────────────────────────────────────────────────────────────────


def test_perfect_run_scores_100():
    score = score_episode(
        episode_id="e", distance_travelled_m=500, route_length_m=500, infractions=[]
    )
    assert score.route_completion == 1.0
    assert score.driving_score == pytest.approx(100.0)


def test_penalties_are_multiplicative():
    """Two collisions must be much worse than twice one — a subtractive penalty
    would let a long route 'earn back' the cost of hitting a pedestrian."""
    one = score_episode(
        episode_id="e", distance_travelled_m=500, route_length_m=500,
        infractions=[Infraction.COLLISION_VEHICLE],
    )
    two = score_episode(
        episode_id="e", distance_travelled_m=500, route_length_m=500,
        infractions=[Infraction.COLLISION_VEHICLE] * 2,
    )
    assert one.infraction_penalty == pytest.approx(0.60)
    assert two.infraction_penalty == pytest.approx(0.36)


def test_pedestrian_collision_is_the_harshest_penalty():
    assert INFRACTION_PENALTIES[Infraction.COLLISION_PEDESTRIAN] == min(
        INFRACTION_PENALTIES.values()
    )


def test_terminal_infractions_carry_no_coefficient():
    """Off-road truncates route completion; penalising it multiplicatively too
    would count the same failure twice."""
    assert Infraction.OFF_ROAD not in INFRACTION_PENALTIES
    assert Infraction.AGENT_BLOCKED not in INFRACTION_PENALTIES
    assert Infraction.LANE_INVASION not in INFRACTION_PENALTIES


def test_partial_route_scales_score():
    score = score_episode(
        episode_id="e", distance_travelled_m=250, route_length_m=500, infractions=[]
    )
    assert score.route_completion == 0.5
    assert score.driving_score == pytest.approx(50.0)


def test_overshooting_route_is_not_extra_credit():
    score = score_episode(
        episode_id="e", distance_travelled_m=900, route_length_m=500, infractions=[]
    )
    assert score.route_completion == 1.0


def test_zero_route_length_is_rejected():
    """Completion would be undefined; silently returning 0 or 1 corrupts every
    aggregate that includes the episode."""
    with pytest.raises(ValueError, match="route_length_m"):
        score_episode(episode_id="e", distance_travelled_m=1, route_length_m=0, infractions=[])


def test_aggregate_uses_arithmetic_mean_over_episodes():
    """The leaderboard definition. A distance-weighted mean would let long easy
    routes mask failures on short hard ones."""
    scores = [
        score_episode(episode_id="a", distance_travelled_m=1000, route_length_m=1000, infractions=[]),
        score_episode(episode_id="b", distance_travelled_m=0.1, route_length_m=100, infractions=[]),
    ]
    summary = aggregate(scores)
    assert summary.driving_score == pytest.approx((100.0 + 0.1) / 2, abs=1e-6)


def test_aggregate_rejects_empty():
    with pytest.raises(ValueError, match="empty score list"):
        aggregate([])


# ─────────────────────────────────────────────────────────────────────────────
# Simulator
# ─────────────────────────────────────────────────────────────────────────────


def test_episode_is_deterministic_in_its_seed():
    """Without this a distributed benchmark cannot be reproduced or bisected."""
    spec = EpisodeSpec(episode_id="e", route_length_m=200, max_steps=400, seed=99)
    scores = []
    for _ in range(2):
        simulator = KinematicSimulator()
        scores.append(run_episode(simulator, spec, PurePursuitPolicy()).driving_score)
        simulator.close()
    assert scores[0] == pytest.approx(scores[1])


def test_different_seeds_give_different_episodes():
    results = []
    for seed in (1, 2):
        simulator = KinematicSimulator()
        spec = EpisodeSpec(episode_id="e", route_length_m=200, max_steps=400, seed=seed)
        results.append(run_episode(simulator, spec, PurePursuitPolicy()).to_dict())
        simulator.close()
    assert results[0] != results[1]


def test_step_before_reset_raises():
    with pytest.raises(RuntimeError, match="reset"):
        KinematicSimulator().step(0.5, 0.0, 0.0)


def test_controls_are_clamped_not_rejected():
    """A policy emitting out-of-range controls is a bug worth surviving, and
    saturating is what real actuators do."""
    simulator = KinematicSimulator()
    simulator.reset(EpisodeSpec(episode_id="e", route_length_m=100))
    result = simulator.step(throttle=99.0, steer=-50.0, brake=-3.0)
    assert result.state.speed_mps >= 0.0


def test_baseline_planner_completes_routes():
    """A baseline that cannot finish a route is not a baseline."""
    simulator = KinematicSimulator()
    scores = [
        run_episode(
            simulator,
            EpisodeSpec(episode_id=f"e{i}", route_length_m=300, max_steps=1200, seed=200 + i),
            PurePursuitPolicy(),
        )
        for i in range(6)
    ]
    simulator.close()
    summary = aggregate(scores)
    assert summary.route_completion > 0.9, f"baseline only completed {summary.route_completion:.2f}"
    assert summary.failures == 0


def test_renderer_produces_model_shaped_frames():
    simulator = KinematicSimulator(render=True)
    state = simulator.reset(EpisodeSpec(episode_id="e", route_length_m=200, seed=3))
    simulator.close()
    assert state.image is not None
    assert state.image.shape == (RENDER_HEIGHT, RENDER_WIDTH, 3)
    assert state.image.dtype == np.uint8


def test_rendering_is_off_by_default():
    """Rendering costs ~10x a bare step; orchestration benchmarks do not need it."""
    simulator = KinematicSimulator()
    state = simulator.reset(EpisodeSpec(episode_id="e", route_length_m=100))
    simulator.close()
    assert state.image is None


def test_build_simulator_falls_back_to_kinematic():
    assert build_simulator("kinematic").name == "kinematic"
    assert build_simulator("auto").name in ("kinematic", "carla")


def test_build_simulator_validates_backend():
    with pytest.raises(ValueError, match="unknown simulator backend"):
        build_simulator("gazebo")


def test_telemetry_sink_failure_does_not_abort_episode():
    """Telemetry is derived data. Losing a row is acceptable; losing the episode
    to a logging failure is not."""
    def broken_sink(_row: dict) -> None:
        raise RuntimeError("sink is down")

    simulator = KinematicSimulator()
    score = run_episode(
        simulator,
        EpisodeSpec(episode_id="e", route_length_m=150, max_steps=400),
        PurePursuitPolicy(),
        telemetry_sink=broken_sink,
    )
    simulator.close()
    assert score.status == "completed"


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────


def test_all_episodes_complete_across_workers():
    specs = build_episode_specs(count=12, route_length_m=150, max_steps=400)
    run = BenchmarkCoordinator(
        LocalMessageQueue(visibility_timeout=60), simulator_backend="kinematic"
    ).run(specs, workers=4)
    assert len(run.scores) == len(specs)
    assert set(run.scores) == {spec.episode_id for spec in specs}
    assert not run.dead_letters


def test_results_are_keyed_by_episode_so_duplicates_do_not_inflate():
    """At-least-once delivery means duplicates happen; counting them would
    silently bias every aggregate."""
    specs = build_episode_specs(count=6, route_length_m=120, max_steps=300)
    run = BenchmarkCoordinator(
        LocalMessageQueue(visibility_timeout=60), simulator_backend="kinematic"
    ).run(specs, workers=3)
    assert len(run.scores) == 6


def test_orchestration_rejects_bad_arguments():
    coordinator = BenchmarkCoordinator(LocalMessageQueue(), simulator_backend="kinematic")
    with pytest.raises(ValueError, match="no episodes"):
        coordinator.run([], workers=2)
    with pytest.raises(ValueError, match="workers"):
        coordinator.run(build_episode_specs(count=1), workers=0)


def test_build_episode_specs_is_deterministic_and_varied():
    first = build_episode_specs(count=12)
    second = build_episode_specs(count=12)
    assert [s.to_dict() for s in first] == [s.to_dict() for s in second]
    assert len({s.town for s in first}) > 1
    assert len({s.weather for s in first}) > 1
    assert len({s.seed for s in first}) == 12


def test_build_episode_specs_rejects_zero():
    with pytest.raises(ValueError, match="count"):
        build_episode_specs(count=0)


# ─────────────────────────────────────────────────────────────────────────────
# Detection metrics
# ─────────────────────────────────────────────────────────────────────────────


def test_iou_of_identical_boxes_is_one():
    box = np.array([[0.0, 0.0, 10.0, 10.0]])
    assert iou_matrix(box, box)[0, 0] == pytest.approx(1.0)


def test_iou_of_disjoint_boxes_is_zero():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[20.0, 20.0, 30.0, 30.0]])
    assert iou_matrix(a, b)[0, 0] == 0.0


def test_iou_half_overlap():
    a = np.array([[0.0, 0.0, 10.0, 10.0]])
    b = np.array([[5.0, 0.0, 15.0, 10.0]])
    # intersection 50, union 150
    assert iou_matrix(a, b)[0, 0] == pytest.approx(50 / 150)


def test_iou_handles_empty_and_degenerate_boxes():
    assert iou_matrix(np.zeros((0, 4)), np.ones((2, 4))).shape == (0, 2)
    degenerate = np.array([[5.0, 5.0, 5.0, 5.0]])
    assert iou_matrix(degenerate, degenerate)[0, 0] == 0.0


def test_perfect_detector_scores_ap_one():
    boxes = np.array([[0.0, 0.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]])
    predictions = [{"boxes": boxes, "scores": np.array([0.9, 0.8]), "classes": np.array([0, 0])}]
    truth = [{"boxes": boxes, "classes": np.array([0, 0])}]
    metrics = compute_map(predictions, truth, {0: "car"}, iou_thresholds=(0.5,))
    assert metrics.map_50 == pytest.approx(1.0)


def test_detector_that_finds_nothing_scores_zero():
    predictions = [{"boxes": np.zeros((0, 4)), "scores": np.zeros(0), "classes": np.zeros(0, int)}]
    truth = [{"boxes": np.array([[0.0, 0.0, 10.0, 10.0]]), "classes": np.array([0])}]
    metrics = compute_map(predictions, truth, {0: "car"}, iou_thresholds=(0.5,))
    assert metrics.map_50 == 0.0


def test_absent_classes_are_excluded_from_map():
    """Averaging in a class that never appears penalizes the model for something
    it was never asked to do."""
    boxes = np.array([[0.0, 0.0, 10.0, 10.0]])
    predictions = [{"boxes": boxes, "scores": np.array([0.9]), "classes": np.array([0])}]
    truth = [{"boxes": boxes, "classes": np.array([0])}]
    metrics = compute_map(predictions, truth, {0: "car", 1: "tram"}, iou_thresholds=(0.5,))
    assert metrics.map_50 == pytest.approx(1.0)
    assert "tram" in metrics.absent_classes


def test_duplicate_detection_is_matched_as_a_false_positive():
    """One ground-truth box cannot satisfy two detections.

    The duplicate is counted — ``detections`` sees both — but it is matched as a
    false positive rather than stealing the already-matched ground truth.
    """
    box = np.array([[0.0, 0.0, 10.0, 10.0]])
    predictions = [
        {
            "boxes": np.vstack([box, box]),
            "scores": np.array([0.9, 0.85]),
            "classes": np.array([0, 0]),
        }
    ]
    truth = [{"boxes": box, "classes": np.array([0])}]
    metrics = compute_map(predictions, truth, {0: "car"}, iou_thresholds=(0.5,))

    assert metrics.detections == 2
    assert metrics.ground_truths == 1


def test_a_trailing_false_positive_does_not_lower_ap():
    """AP is an area under the PR curve, so a *lower-confidence* false positive
    is free.

    It adds no recall, so it contributes zero width to the integral. This is
    true under COCO's interpolated precision and under raw integration alike —
    it is a property of the metric, not of this implementation. Duplicate
    suppression therefore has to be judged by precision at a fixed operating
    point; expecting mAP to catch it is a misreading of the metric.
    """
    ranked_correctly = [BoxMatch(0.9, True), BoxMatch(0.85, False)]
    assert average_precision(ranked_correctly, num_ground_truth=1) == pytest.approx(1.0)


def test_a_false_positive_ranked_above_a_hit_does_lower_ap():
    """The case AP *does* punish: a confident wrong box outranking a real one.

    Recall only reaches 1.0 after two detections, so precision there is 0.5 and
    the area halves. This is the counterpart to the test above and the reason
    confidence calibration matters to mAP at all.
    """
    ranked_badly = [BoxMatch(0.9, False), BoxMatch(0.85, True)]
    assert average_precision(ranked_badly, num_ground_truth=1) == pytest.approx(0.5)


def test_compute_map_rejects_length_mismatch():
    with pytest.raises(ValueError, match="prediction sets"):
        compute_map([{}], [{}, {}], {0: "car"})


def test_average_precision_with_no_ground_truth_is_zero():
    assert average_precision([], 0) == 0.0


def test_measure_throughput_reports_context():
    """FPS without hardware/batch/resolution is not interpretable."""
    metrics = measure_throughput(
        lambda: time.sleep(0.001), iterations=5, warmup=2, batch_size=4, device="cpu"
    )
    assert metrics.fps > 0
    assert metrics.batch_size == 4 and metrics.device == "cpu"
    assert metrics.p99_latency_ms >= metrics.p50_latency_ms
    assert "batch=4" in metrics.summary()


def test_measure_throughput_rejects_zero_iterations():
    with pytest.raises(ValueError, match="iterations"):
        measure_throughput(lambda: None, iterations=0)


# ─────────────────────────────────────────────────────────────────────────────
# Warehouse
# ─────────────────────────────────────────────────────────────────────────────


def test_warehouse_writes_partitioned_parquet(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    store = LocalObjectStore(tmp_path)
    warehouse = TelemetryWarehouse(store)
    rows = [
        {
            "episode_id": "e1", "frame_index": i, "timestamp": datetime.now(UTC),
            "simulation_time": i * 0.05, "worker_id": "w0", "town": "Town01",
            "weather": "ClearNoon", "x": 0.0, "y": 0.0, "yaw_degrees": 0.0,
            "speed_mps": 5.0, "throttle": 0.4, "brake": 0.0, "steer": 0.0,
            "command": 0, "detections": 1, "nearest_object_m": 20.0,
            "policy_latency_ms": 0.1, "perception_latency_ms": 0.2,
            "infraction": "", "event_date": "2026-04-01",
        }
        for i in range(10)
    ]
    keys = warehouse.write_frames(rows)
    assert len(keys) == 1 and "event_date=2026-04-01" in keys[0]

    table = pq.read_table(tmp_path / keys[0])
    assert table.num_rows == 10
    assert table.schema.names == [column.name for column in FRAME_COLUMNS]


def test_warehouse_drain_parses_iso_timestamps(tmp_path):
    pytest.importorskip("pyarrow")
    stream = LocalTelemetryStream(tmp_path / "stream")
    stream.put("e1", {"episode_id": "e1", "frame_index": 0, "timestamp": datetime.now(UTC)})
    stream.flush()

    warehouse = TelemetryWarehouse(LocalObjectStore(tmp_path / "s3"))
    keys = warehouse.drain_stream(stream)
    assert len(keys) == 1 and warehouse.rows_written == 1


def test_warehouse_skips_malformed_records(tmp_path):
    """One bad frame must not cost an entire benchmark run's telemetry."""
    pytest.importorskip("pyarrow")
    stream = LocalTelemetryStream(tmp_path / "stream")
    stream.put("e1", {"nonsense": True})
    stream.put("e1", {"episode_id": "e1", "frame_index": 0})
    stream.flush()

    warehouse = TelemetryWarehouse(LocalObjectStore(tmp_path / "s3"))
    warehouse.drain_stream(stream)
    assert warehouse.rows_written == 1


def test_ddl_covers_every_column():
    redshift, athena = redshift_ddl(), athena_ddl()
    for column in list(FRAME_COLUMNS) + list(EPISODE_COLUMNS):
        assert column.name in redshift
        assert column.name in athena


def test_redshift_ddl_commas_precede_comments():
    """A trailing ``-- comment`` swallows the rest of the line, so a comma after
    one merges every following column into the comment."""
    ddl = redshift_ddl()
    for line in ddl.splitlines():
        stripped = line.strip()
        if not stripped.startswith(tuple(c.name for c in FRAME_COLUMNS)):
            continue
        if "--" in line and "," in line:
            assert line.index(",") < line.index("--"), f"comma after comment: {line!r}"


def test_redshift_ddl_sets_distribution_strategy():
    ddl = redshift_ddl()
    assert "DISTKEY(episode_id)" in ddl
    assert "DISTSTYLE ALL" in ddl  # small episode table replicated for joins
