# What this project claims, and the evidence

<!-- claims: checked -->

This is the claim-of-record: every load-bearing claim the project makes, each
mapped to the artifact that backs it. Numbers are written in the citation
convention of [docs/CLAIMS.md](docs/CLAIMS.md) — every quoted value is a link
naming its artifact and the JSON field it quotes — and the claim checker
(`pathfinder/claims.py`, run by `tests/test_claims_checker.py` in the ordinary
suite) verifies each one in CI. A claim that drifts from its artifact is a
failing build.

Two kinds of row appear below:

- **Machine-checked** — the quoted number is verified against the artifact
  field it cites, rounded to the quote's own decimal count.
- **Prose-audited** — negative and boundary claims ("never applied", "not
  project work") have no number to check; the checker verifies the cited
  evidence exists, and the sentence itself is held true by review.

Each section opens with the command that reproduces its artifacts and names
the test that pins the claims' mechanism (the geometry, the report shape, the
debounce rule) so it cannot silently regress. The numbers themselves are
pinned by this document: regenerating an artifact without updating the claim
fails the build. Reproduction commands that need KITTI on disk, a GPU, or a
live CARLA server say so; everything else runs on any laptop.

Audit the whole table in one command:

```
.venv/bin/python -m pathfinder.claims
```

---

## 1. The KITTI split leak

Reproduce: `python scripts/prepare_kitti.py --report-baseline` (needs raw
KITTI on disk; deterministic). Pinned by `tests/test_kitti_data.py`.

| Claim | Kind |
|---|---|
| KITTI's [7481](data/manifest.json "claim:frames") object-detection frames come from [141](data/manifest.json "claim:drives") continuous 10 Hz video drives, so a random train/val split puts near-duplicate frames on both sides. | machine-checked |
| In Ultralytics' bundled random split, a fraction [0.34](data/manifest.json "claim:baseline_random_split.temporal_adjacency.1.fraction") of val frames sit within one frame (0.1 s) of a training frame, [0.76](data/manifest.json "claim:baseline_random_split.temporal_adjacency.2.fraction") within two, and [0.97](data/manifest.json "claim:baseline_random_split.temporal_adjacency.10.fraction") within ten. | machine-checked |
| [106](data/manifest.json "claim:baseline_random_split.drives_shared") of [141](data/manifest.json "claim:baseline_random_split.drives_total") drives straddle the random split. | machine-checked |
| This repo's split is sequence-disjoint: [0](data/manifest.json "claim:leakage.drives_shared") drives are shared between train ([5978](data/manifest.json "claim:split.train_frames") frames) and val ([1503](data/manifest.json "claim:split.val_frames") frames). | machine-checked |
| The measured cost of the leak is ≈49 mAP points: the legacy leaky-split checkpoint scores [0.963](results/perception/eval-yolo26m-legacy/report.json "claim:map50") mAP@0.5 on the sequence-disjoint val set, against the honestly trained model's [0.475](results/perception/yolov8m/report.json "claim:map50"). | machine-checked (the ≈49 is the difference of the two checked numbers) |
| The legacy checkpoint scores *higher* on the "held-out" drives than its original random-split figure — [it had already memorised them](results/perception/eval-yolo26m-legacy/report.json "claim:prose"). | prose-audited |

## 2. The honest detector (YOLOv8m, sequence-disjoint val)

Reproduce: `python scripts/eval_detector.py --weights results/perception/yolov8m/weights/best.pt`
(needs KITTI on disk; GPU strongly recommended). Report shape pinned by
`tests/test_detection_report.py`.

| Claim | Kind |
|---|---|
| YOLOv8m scores mAP@0.5 [0.475](results/perception/yolov8m/report.json "claim:map50"), mAP@0.5:0.95 [0.307](results/perception/yolov8m/report.json "claim:map50_95"), precision [0.586](results/perception/yolov8m/report.json "claim:precision"), recall [0.440](results/perception/yolov8m/report.json "claim:recall") over [1376](results/perception/yolov8m/report.json "claim:images") images and [9079](results/perception/yolov8m/report.json "claim:instances") instances. | machine-checked |
| Per-class AP@0.5: car [0.872](results/perception/yolov8m/report.json "claim:per_class.0.ap50"), truck [0.794](results/perception/yolov8m/report.json "claim:per_class.4.ap50"), pedestrian [0.625](results/perception/yolov8m/report.json "claim:per_class.1.ap50"), tram [0.529](results/perception/yolov8m/report.json "claim:per_class.6.ap50"), van [0.442](results/perception/yolov8m/report.json "claim:per_class.2.ap50"), cyclist [0.425](results/perception/yolov8m/report.json "claim:per_class.3.ap50"), misc [0.106](results/perception/yolov8m/report.json "claim:per_class.5.ap50"), person_sitting [0.005](results/perception/yolov8m/report.json "claim:per_class.7.ap50"). | machine-checked |
| The detector's problem is recall, not classification: [missed detections dominate the confusion matrix](results/perception/yolov8m/confusion_matrix.png "claim:prose"); what it does detect it rarely mislabels. | prose-audited |
| The protocol is strictly harder than published KITTI numbers: [difficulty tiers and DontCare regions were dropped by the conversion](scripts/prepare_kitti.py "claim:prose"), so every object counts, however occluded or truncated. These figures are not comparable to leaderboard KITTI. | prose-audited |
| More epochs do not help: [a 60-epoch run peaks at epoch 10 and declines](results/perception/yolov8m/results.csv "claim:prose") — overfitting on [5978](data/manifest.json "claim:split.train_frames") training images, confirmed by experiment. Prose-audited because the epoch curve lives in `results.csv`, and the checker cites JSON field paths only — read the mAP columns there directly. | machine-checked + prose-audited |
| `person_sitting` is structurally unmeasurable: its whole val set is [1](results/perception/yolov8m/report.json "claim:per_class.7.drives") drive with [56](results/perception/yolov8m/report.json "claim:per_class.7.instances") instances, so any AP for it is high-variance; [the report flags it low-confidence automatically](results/perception/yolov8m/report.json "claim:prose"). | machine-checked + prose-audited |

## 3. What imperfect perception costs on the road (GT-vs-Detector ablation)

Reproduce: `python -m pathfinder.ablation --backend carla` (needs a live CARLA
0.9.16 server on the Windows machine). Mechanism pinned CARLA-free by
`tests/test_ablation.py`; write-up by `tests/test_ablation_writeup.py`.

| Claim | Kind |
|---|---|
| Over [10](results/ablation/carla_report.json "claim:baseline.summary.episodes") seeded episodes per arm (Towns 01/03/05, four weathers, seeds from [1000](results/ablation/carla_report.json "claim:episodes.0.seed"), [0](results/ablation/carla_report.json "claim:candidate.summary.failures") failures), privileged perception scores driving score [25.2](results/ablation/carla_report.json "claim:baseline.summary.driving_score") and the real detector [8.86](results/ablation/carla_report.json "claim:candidate.summary.driving_score"): imperfect perception costs [16.34](results/ablation/carla_report.json "claim:difference.driving_score") points. | machine-checked |
| Blindness reads as boldness: the detector arm completes *more* route ([0.481](results/ablation/carla_report.json "claim:candidate.summary.route_completion") vs [0.396](results/ablation/carla_report.json "claim:baseline.summary.route_completion")) but collides ten times as often ([26.5](results/ablation/carla_report.json "claim:candidate.summary.collisions_per_km") vs [2.5](results/ablation/carla_report.json "claim:baseline.summary.collisions_per_km") per km, [47](results/ablation/carla_report.json "claim:candidate.summary.infraction_totals.collision_vehicle") vehicle collisions against none in the privileged arm). | machine-checked |
| Mean detector inference is [8.35](results/ablation/carla_report.json "claim:candidate.mean_perception_latency_ms") ms/frame in the loop. | machine-checked |
| Under the pre-registered rule, perception is **not** the binding constraint: [the baseline's own 74.8-point shortfall from a perfect score exceeds perception's 16.34](results/ablation/carla_report.json "claim:prose") — PurePursuit is the bottleneck, so the deferred fine-tuning trigger did not fire. | prose-audited |

## 4. CARLA backend: determinism, throughput, validation

Reproduce: `python scripts/probe_carla.py` and
`python scripts/validate_carla_backend.py` (both need a live CARLA 0.9.16
server). Route geometry pinned CARLA-free by `tests/test_route.py`; collision
debounce by `tests/test_collision_debounce.py`.

| Claim | Kind |
|---|---|
| CARLA 0.9.16 is bit-reproducible on the Windows machine: two runs of one seed diverged [0.0](results/carla/probe.json "claim:drive.max_divergence_m") m over [120](results/carla/probe.json "claim:drive.steps") ticks, at [172.0](results/carla/probe.json "claim:drive.ticks_per_second.1")–[180.85](results/carla/probe.json "claim:drive.ticks_per_second.0") ticks/sec. | machine-checked |
| Determinism holds over full episodes with traffic and pedestrians, not just the probe: two runs of seed [42](results/carla/backend_validation.json "claim:spec.seed") over [1792](results/carla/backend_validation.json "claim:episode1.steps_run") steps ended at identical completion [0.471](results/carla/backend_validation.json "claim:episode1.final_completion") = [0.471](results/carla/backend_validation.json "claim:episode2.final_completion") — but only with fixed delta, synchronous mode, a seeded traffic manager **and** [`world.set_pedestrians_seed()`](pathfinder/sim/carla_backend.py "claim:prose") together. | machine-checked + prose-audited |
| PurePursuit completes [0.471](results/carla/backend_validation.json "claim:episode1.final_completion") of a [351](results/carla/backend_validation.json "claim:episode1.route_length_m") m Town05 route before `agent_blocked` — a driving-quality ceiling of the controller, [not a backend defect](results/carla/backend_validation.json "claim:prose"). | machine-checked + prose-audited |
| "All four navigation branches" is checked as a planner property, not an episode property: across [27](results/carla/backend_validation.json "claim:command_survey.routes_checked") sampled routes [the planner plus command mapping produce all four branches](results/carla/backend_validation.json "claim:prose"), and a driven episode only surfaces commands its own route planned. | machine-checked + prose-audited |
| CARLA's collision sensor re-fires every tick during sustained contact; [the backend debounces per actor with a 5 s cooldown mirroring the Leaderboard's rule](tests/test_collision_debounce.py "claim:prose"). The run that exposed this (one pile-up counted as 475 raw "collisions", underflowing 9 of 10 candidate episodes to a 0.0 score) predates the fix and was deliberately not kept as an artifact, so those numbers are historical and cannot be machine-checked — the debounce rule itself is what the test pins. | prose-audited |

## 5. Monocular range

| Claim | Kind |
|---|---|
| Monocular range saturates in the near field: below the camera's `min_measurable_range_m` the obstacle's ground contact leaves the frame and the estimate over-reads — the unsafe direction. The floor is exposed on `CameraGeometry` and [pinned so it cannot regress into a claim that the near field works](tests/test_range_geometry.py "claim:prose"). | prose-audited |

## 6. gRPC service latency

Reproduce: `python scripts/bench_rpc_latency.py` (any machine, loopback).
Service pinned by `tests/test_rpc.py`.

| Claim | Kind |
|---|---|
| Coordinator round-trip p50 over loopback, [500](results/rpc/latency_report.json "claim:calls_per_rpc") calls per RPC: RegisterWorker [0.26](results/rpc/latency_report.json "claim:results.0.p50_ms") ms, Heartbeat [0.26](results/rpc/latency_report.json "claim:results.1.p50_ms") ms, SubmitResult [0.26](results/rpc/latency_report.json "claim:results.2.p50_ms") ms, GetRunStatus [0.82](results/rpc/latency_report.json "claim:results.3.p50_ms") ms. | machine-checked |

## 7. The BehaviorAgent reference ceiling and the training gate

Reproduce: `python -m pathfinder.reference_run --backend carla` (needs a live
CARLA 0.9.16 server on the Windows machine). Mechanism pinned CARLA-free by
`tests/test_reference_run.py`; write-up by `tests/test_reference_writeup.py`.

| Claim | Kind |
|---|---|
| CARLA's built-in behaviour agent — **not project work** — scores driving score [31.33](results/reference/carla_report.json "claim:summary.driving_score") over the ablation's exact seeded suite ([10](results/reference/carla_report.json "claim:summary.episodes") episodes, [0](results/reference/carla_report.json "claim:summary.failures") failures), route completion [0.368](results/reference/carla_report.json "claim:summary.route_completion"), with [3](results/reference/carla_report.json "claim:summary.infraction_totals.collision_vehicle") vehicle collisions and [6](results/reference/carla_report.json "claim:summary.infraction_totals.agent_blocked") of 10 episodes ended agent-blocked. | machine-checked |
| The pre-registered training gate: reference [31.33](results/reference/carla_report.json "claim:floor_gate.reference_driving_score") over the recorded privileged-PurePursuit floor [25.2](results/reference/carla_report.json "claim:floor_gate.floor_driving_score") is a margin of [6.13](results/reference/carla_report.json "claim:floor_gate.margin") points, short of the required [10.0](results/reference/carla_report.json "claim:floor_gate.required_margin"). | machine-checked |
| [The gate's verdict is **stop-and-reassess**](results/reference/carla_report.json "claim:prose"), under the rule pre-registered before the run; the margin is computed by the gate code from the ablation artifact, never hardcoded. The reassessment resolved on 2026-08-21: **Phase 3 training stopped by decision** (#11) — the DAgger/comparison machinery remains as pipeline work. | prose-audited |

## 8. The distributed pipeline rehearsal (LocalStack, #22)

Reproduce: `docs/SETUP_WINDOWS.md` §9 (LocalStack + two terminals, ~2
minutes; kinematic workers, so any laptop). Mechanism pinned CARLA-free by
`tests/test_distributed_e2e.py` and `tests/test_run_worker.py`; queue
provisioning by `tests/test_provision_sqs.py`.

| Claim | Kind |
|---|---|
| The runbook rehearsal completed [8](results/distributed/localstack_rehearsal.json "claim:coordinator.episodes_completed") of [8](results/distributed/localstack_rehearsal.json "claim:coordinator.episodes_total") episodes against LocalStack SQS: the chaos-killed worker's episode was redelivered by the visibility timeout ([1](results/distributed/localstack_rehearsal.json "claim:queue.redeliveries") redelivery, delivered [2](results/distributed/localstack_rehearsal.json "claim:queue.redelivered_episodes.0.receive_count") times) and completed by the surviving worker, with [0](results/distributed/localstack_rehearsal.json "claim:queue.dead_letters") messages dead-lettered and [0](results/distributed/localstack_rehearsal.json "claim:queue.approximate_depth") left on the queue. | machine-checked |
| [The rehearsal artifact is pipeline evidence only](results/distributed/localstack_rehearsal.json "claim:prose") — kinematic backend, scope pipeline-only, run id `localstack-rehearsal`: it proves queue/coordinator/warehouse mechanics against a real SQS API, and its aggregate is never a driving-quality or benchmark number. No live AWS queue was used (that remains #18/#26). | prose-audited |

## 9. Boundaries and negative claims

These are the claims about what this project is *not*. They are all
prose-audited: the checker verifies the evidence exists; review holds the
sentences true.

| Claim | Evidence |
|---|---|
| All Terraform ([EKS](terraform/eks.tf "claim:prose"), ECR, SQS+DLQ, S3, KMS, IRSA, GitHub OIDC) is written and validated in CI but has **never been applied to real AWS**; nothing in this repo has ever provisioned a paid cloud resource. | `terraform/` |
| [Kinesis is provisioned in Terraform and exercised against moto](terraform/kinesis.tf "claim:prose"), but no real stream has ever been created — "Kinesis, never applied to real AWS". The default telemetry backend is local. | `terraform/kinesis.tf`, `tests/test_cloud_aws.py` |
| [The SQS queue code carries real visibility-timeout and DLQ semantics](pathfinder/cloud/queue.py "claim:prose") and is tested against emulation; no live AWS queue has been used yet (that is ticket #18/#26). | `pathfinder/cloud/queue.py` |
| Kubernetes orchestration runs on [kind](k8s/kind-config.yaml "claim:prose"), with an EKS overlay — "Kubernetes, with EKS-ready Terraform", **not** "deployed on EKS". | `k8s/` |
| [SageMaker integration is the real SageMaker SDK in local mode](pathfinder/cloud/training.py "claim:prose") — local Docker, zero spend. | `pathfinder/cloud/training.py` |
| [`carla_builtin_behavior_agent` is CARLA's own behaviour agent, **not project work**](pathfinder/policies.py "claim:prose"): its driving score is a reference upper bound only — measured live once by the #16 reference baseline (section 7) and quotable only as a ceiling, never as this project's driving. | `pathfinder/policies.py`, `results/reference/carla_report.json` |
| [No learned policy has been trained, and none will be under the current plan](pathfinder/dagger.py "claim:prose"): the DAgger loop, checkpointing, and scored comparison are proven CARLA-free with untrained weights and kept as pipeline machinery; training was stopped by decision when the pre-registered teacher-quality gate returned stop-and-reassess (section 7; #16, #11). | `pathfinder/dagger.py`, `results/reference/carla_report.json` |
| The detector's KITTI numbers are on synthetic-free real imagery; [driving it in CARLA is out-of-domain](results/ablation/carla_report.json "claim:prose"), which the ablation quantifies rather than hides. | `results/ablation/carla_report.json` |

---

## Adding a claim

New claims land here first, then in any other document. Write the sentence,
cite every number with the [convention](docs/CLAIMS.md), and run
`.venv/bin/python -m pathfinder.claims` — the suite fails until the claim and
its artifact agree.
