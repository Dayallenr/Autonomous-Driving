# PathFinder — project rules and status

Autonomous-driving portfolio project for Dayallen Ragunathan (SWE + ML roles).
This file loads every session. Read it before doing anything.

---

## Prime directive: truthfulness over convenience

Every claim in the README, and eventually on a resume, must correspond to real
working code and a reproducible artifact in this repo — a script, a report JSON,
a plot, a logged metric. If a claim cannot be made true without unreasonable
effort, **say so explicitly** and propose either a scoped-down honest version or
what it would take to make it true. Never leave a claim standing that the repo
does not back up.

This is not a style preference. The project's entire value is that a hiring
manager can open it, read a number, and verify it. A single invented figure
destroys that.

**Corollary: never report a number you have not personally seen produced.** If
results exist only on another machine or in a chat message, they do not exist.

---

## Ground rules

1. **Real end-to-end, not a tool checklist.** Data → cleaning → training →
   evaluation with real plots → infrastructure that does a genuine job. If a
   cloud service has no load-bearing role, cut it and say so rather than
   name-dropping it.

2. **Audit before building.** Verify by running things. Do not assume code
   works because it exists — a large fraction of this repo was written without
   ever being executed, and much of it was subtly wrong.

3. **Zero spend by default.** Do not provision paid cloud resources. Concretely:
   - **AWS SQS**: real AWS. Its free tier (1M requests/month, permanent) covers
     this project. Set a $1 budget alarm as a safety net.
   - **EKS → local Kubernetes** (kind). Write real, deployable EKS Terraform;
     validate with `terraform validate`/`plan` + `checkov`/`trivy`; never
     `apply` to real AWS. EKS control plane is ~$0.10/hr and not free-tier.
   - **SageMaker → SageMaker SDK local mode.** Real SDK, local Docker, $0.
   - **S3 / Kinesis / DynamoDB → LocalStack.**
   - If anything would cost money, **stop and ask first.**

4. **Checkpoint with the user** before: implementing cloud pieces, anything that
   costs money, and when a claim cannot be made honestly true.

5. **Tool honesty in write-ups.** SQS runs live, so "AWS SQS" is fair. The
   Kubernetes orchestration runs on kind, so say "Kubernetes, with EKS-ready
   Terraform" — not "deployed on EKS". SageMaker local mode is genuinely the
   SageMaker SDK, but say it runs in local mode. Kinesis is **not** cut:
   `terraform/kinesis.tf` provisions an on-demand stream and
   `KinesisTelemetryStream` is exercised against moto in
   `tests/test_cloud_aws.py` — but no real stream has ever been created, so
   say "Kinesis, never applied to real AWS". The default telemetry backend
   is `LocalTelemetryStream`. The `carla_builtin_behavior_agent` Policy is
   **CARLA's own behaviour agent, not project work** — never present its driving
   score as this project's result; it is a reference upper bound only, and it
   has never been run against a live server.

6. **Resume bullets are deferred.** The user has said explicitly: do not spend
   effort reframing resume bullets. Focus on making the project as good as it
   can be — metrics, breadth of tooling, and engineering quality. Bullets come
   later, at the end.

---

## Environment

| | |
|---|---|
| Windows desktop | RTX 5070 12 GB (Blackwell, **sm_120**), CARLA 0.9.16 working |
| Mac | Apple M2, no CUDA, **cannot run CARLA** — kinematic backend only |
| Repo | https://github.com/Dayallenr/Autonomous-Driving (dir is `PathFinder`) |

**PyTorch on the 5070 must come from the cu130 channel** — `cu128` is stale
(tops out at torch 2.11) and `cu126` has no sm_120 kernels:
`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130`

CARLA's `agents` package is auto-discovered by `pathfinder/sim/carla_paths.py`
via `CARLA_ROOT` (set on the Windows machine to the CARLA_0.9.16 folder).

Always run `ruff check` and `pytest` before claiming work is done.

---

## Established findings — do not re-litigate these

**The standard KITTI split leaks its validation set.** KITTI's 7,481 images come
from 141 continuous 10 Hz video drives. Ultralytics' bundled `kitti.yaml` splits
them randomly, so **34% of val frames sit within one frame (0.1 s) of a training
frame**, 76% within two, 97% within ten. 106 of 141 drives straddle that split.
`scripts/prepare_kitti.py --report-baseline` measures this.

**Quantified cost of the leak: ~49 mAP points.** The legacy YOLO26m checkpoint
scores 0.963 mAP@0.5 on the leaky split and the honestly-trained YOLOv8m scores
0.475 on a sequence-disjoint split. Evaluating the legacy checkpoint on the
"held-out" drives gives 0.963 — *higher* than its original 0.919 — proving it had
already memorised them.

**YOLOv8m honest result: mAP@0.5 = 0.475**, mAP@0.5:0.95 = 0.307, precision
0.586, recall 0.440. Per-class AP@0.5: car 0.872, truck 0.794, pedestrian 0.626,
tram 0.529, van 0.442, cyclist 0.425, misc 0.106, person_sitting 0.005.

**The detector's problem is recall, not classification.** The confusion matrix
shows missed detections dominate: 82% of `person_sitting` and 41–45% of
`misc`/`cyclist`/`tram` go to background. The model rarely misclassifies what it
does detect.

**Partly a protocol artifact.** Ultralytics' KITTI conversion dropped KITTI's
difficulty tiers (easy/moderate/hard) and `DontCare` regions. We score every
object including severely occluded and truncated ones with no `DontCare`
masking — strictly harder than any published KITTI number. **Recovering that
metadata from raw KITTI is a known, unstarted improvement.**

**More epochs do not help.** A full 60-epoch run peaks at **epoch 10** and
declines; the mosaic-off fine-tuning phase made it worse. This is overfitting on
5,978 training images, confirmed by experiment. `best.pt` already holds the best
weights. Early stopping is disabled by default (`--patience 0`) for a documented
reason — read the comment in `scripts/train_detector.py` before changing it.

**`person_sitting` is structurally unmeasurable.** It occupies 3 of 141 drives,
so its entire validation set is one drive (56 instances). Any AP for it is
high-variance. Reports flag single-drive classes automatically.

**Monocular range saturates in the near field.** `range_from_box` inverts the
renderer's ground-plane projection, and below `min_measurable_range_m` (3.5 m on
the kinematic camera) the obstacle's ground contact falls outside the frame. The
detected box stops descending, so a vehicle at 2.5 m measures 3.58 m — it
over-reads, which is the unsafe direction. The information is not in the image
and no monocular method recovers it; the floor is exposed on `CameraGeometry`
and pinned by `tests/test_range_geometry.py` so it cannot regress into a claim
that the near field works.

**CARLA 0.9.16 is bit-reproducible on the Windows machine.** Two runs of one
seed gave 0.0 m divergence over 120 ticks, at **172–181 ticks/sec**
(`results/carla/probe.json`) — so 1,000 episodes is roughly 2.5 hours
single-threaded. This makes "replay any result bit-identically" a supportable
claim, *provided* fixed delta, synchronous mode, and a seeded traffic manager
are all set together.

---

## Status by phase

| Phase | State |
|---|---|
| 0 Audit | Done |
| 1 KITTI data pipeline | **Done** — sequence-disjoint split, reproducible on Mac + Windows |
| 2 Perception | **Done** — YOLOv8m trained, evaluated, `results/perception/yolov8m/report.json` |
| — CARLA backend rewrite | **Written, never executed against a live server** ← next |
| 3 CIL Policy + DAgger | Not started — needs CARLA |
| 4 GT-vs-YOLO ablation | **Mechanism done** — `python -m pathfinder.ablation` runs both arms over identical seeded specs, records observed provenance per Episode, and labels kinematic reports pipeline-only (`results/ablation/kinematic_report.json`). The measurement bias #10 recorded is fixed: the CARLA privileged arm now measures forward-frustum, camera-origin, ground-plane range to the obstacle's nearest visible surface — the same convention the Detector arm inverts — pinned by `tests/test_range_convention.py`. The real number needs CARLA (#10) |
| 5 Distributed benchmark (SQS, telemetry, Parquet) | Queue/telemetry/warehouse code exists and is tested; needs real CARLA episodes and real AWS SQS |
| 6 gRPC service | **Done** — `pathfinder/rpc/server.py` binds a port; latency measured over loopback at p50 0.26 ms (RegisterWorker/Heartbeat/SubmitResult) and 0.82 ms (GetRunStatus), 500 calls each, `results/rpc/latency_report.json` |
| 7 Terraform / LocalStack / kind | **Written, never applied** — `terraform/` passes `fmt -check`, `init -backend=false`, `validate`, and `checkov` in CI; provisions EKS, ECR, SQS+DLQ, S3, **Kinesis**, KMS, IRSA, GitHub OIDC. `k8s/` carries kind manifests and an `eks/` overlay. Nothing has been applied to real AWS |
| 8 CI/CD | **Done** — `.github/workflows/ci.yml` (ruff, pytest, hadolint, Docker build, trivy, compose validate, terraform validate + checkov) and `deploy.yml` (manual `workflow_dispatch` only) |
| 9 README + demo | Not started; current README describes the old project |
| 10 Claim-to-artifact mapping | Deferred by user request |

256 tests pass; `ruff check` clean.

---

## Immediate next step

`pathfinder/sim/carla_backend.py` was just rewritten and **has never run against
a live CARLA server.** The previous version was a skeleton missing traffic,
camera, routes, navigation commands, traffic-light infractions, and world reuse.
All six are now implemented but unverified.

Validate it on the Windows machine before building anything on top:

1. Start `CarlaUE4.exe`.
2. Run a short routed episode via `build_simulator("carla")` and confirm:
   route completion advances monotonically, `command` varies across all four
   branches (not stuck on `FOLLOW_LANE`), `state.image` is a 88×200×3 array,
   traffic and pedestrians spawn, collisions and red lights register, and
   teardown restores async mode.
3. Re-run one seed twice and confirm identical results.

Expect bugs — this is 400 lines written blind against an API that could not be
tested locally. Fix them against the real server, and add regression tests to
`tests/test_route.py` (pure geometry, runs anywhere) rather than to CARLA-only
paths.

---

## Repo map

```
pathfinder/
  sim/        base.py (interfaces) · kinematic.py (portable backend, no CARLA)
              carla_backend.py (real CARLA) · route.py (pure geometry, tested)
              carla_paths.py (locates CARLA's agents pkg) · render.py (200x88)
  cloud/      queue.py (SQS + local, real visibility-timeout/DLQ semantics)
              stream.py · objects.py (S3 + dataset registry) · warehouse.py
              training.py (SageMaker contract)
  data/       kitti.py — provenance, sequence-disjoint split, rebalancing
  detection/  evaluate.py — per-class reports with instance/image/drive support
  metrics/    detection.py (mAP) · driving_score.py (CARLA Leaderboard)
  perception/ base.py (Perception protocol + PerceivedScene; documents what
              stays privileged: localization + traffic lights) · privileged.py
              (ground-truth passthrough, behaviour-preserving by
              characterisation test) · geometry.py — monocular range from a
              detected box (pure geometry, round-tripped against render.py's
              forward projection; saturates below min_measurable_range_m) ·
              detector.py (Detector seam + DetectorPerception + YoloDetector;
              Detector loaded once per process, eval mode, no TTA, pinned device)
  planning/   cil_model.py — 4-branch ResNet-18 CIL model (moved from
              top-level planning/, which no longer exists). The package keeps
              its name; ADR-0002 explains why the interface rename stopped here
  rpc/        coordinator.py (servicer) · server.py (binds a port)
              client.py · generated stubs
  policies.py — build_policy(name) registry for the `Policy` protocol;
              ModularPolicy composes a Perception with an inner controller
              (unregistered — `pathfinder/ablation.py` composes it directly);
              CarlaBehaviorAgentPolicy wraps CARLA's own BehaviorAgent as the
              reference baseline, registered as `carla_builtin_behavior_agent`
              (never run against a live server)
  orchestration.py · runner.py · dagger.py · benchmark_detector.py
  ablation.py — the GT-vs-Detector ablation: run_ablation() + CLI; provenance
              observed from telemetry, kinematic reports labelled pipeline-only
scripts/      prepare_kitti.py · train_detector.py · eval_detector.py
              plot_data_report.py · probe_carla.py · generate_protos.py
              bench_rpc_latency.py · run_worker.py · enqueue_episodes.py
              archive_telemetry.py
docs/         DATA.md · SETUP_WINDOWS.md · adr/ (0001 modular, 0002
              Policy naming) · agents/
terraform/    EKS · ECR · SQS+DLQ · S3 · Kinesis · KMS · IRSA · OIDC
              (validated, never applied)
k8s/          kind manifests + eks/ overlay
.github/      workflows/ci.yml · workflows/deploy.yml (manual only)
results/      data/ (figures) · perception/ (reports) · carla/probe.json
              rpc/latency_report.json · ablation/ (pipeline-only kinematic report)
```

**The legacy classical-ADS stack is gone.** `agent.py`, top-level
`perception/`, `localization/`, `prediction/`, top-level `planning/`,
`control/`, `sensors/`, and `main.py` were deleted — they were a closed loop
that only imported each other, and `pathfinder/` had already superseded all
of it.
`planning/cil_model.py` was the one genuinely shared file; it now lives at
`pathfinder/planning/cil_model.py`, and `pathfinder/dagger.py` plus the
Colab notebook (`notebooks/train_cil_dagger.ipynb`) import it from there.

---

## Working style the user has asked for

- Keep instructions to the user **short and concrete**. They have limited time
  and are the only one who can run CARLA/GPU work.
- Batch their tasks so one sitting accomplishes a lot; long jobs should be
  unattended.
- Do the code, IaC, tests, and docs yourself. Only ask them for GPU runs,
  CARLA runs, and credentials (they must run `aws configure` themselves).
- When something breaks, diagnose it properly rather than working around it.
  Several real bugs so far were found by taking failures seriously: a wrong mAP
  test expectation, a loop-variable closure, an off-by-2 curvature formula, an
  Ultralytics `project` path being resolved relative to `runs_dir`, and early
  stopping truncating a training schedule.

---

## Agent skills

### Issue tracker

Issues live as GitHub issues on `Dayallenr/Autonomous-Driving`, driven via the
`gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default vocabulary — `needs-triage`, `needs-info`, `ready-for-agent`,
`ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context — one `CONTEXT.md` and `docs/adr/` at the repo root. Both
exist; they are extended by `/domain-modeling`. See `docs/agents/domain.md`.
