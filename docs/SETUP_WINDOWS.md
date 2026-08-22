# Windows + RTX 5070 setup

For training the detector and, later, running CARLA. Run these in PowerShell.

## 1. Python and the repo

Install **Python 3.13** from [python.org](https://www.python.org/downloads/windows/)
(tick "Add python.exe to PATH"), then:

```powershell
git clone https://github.com/Dayallenr/Autonomous-Driving.git PathFinder
cd PathFinder
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

## 2. PyTorch — the version matters

The RTX 5070 is **Blackwell**, compute capability **sm_120**. CUDA 12.8 was the
first release with sm_120 kernels, so a default `pip install torch` — which
serves a CPU build — or a CUDA 12.6 build will either run on the CPU or die with
`no kernel image is available for execution on the device`.

Install from the **cu130** channel:

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install ultralytics pyyaml matplotlib
```

> Older guides say `cu128`. That channel has stopped receiving updates — it tops
> out at torch 2.11 — while `cu126` and `cu130` carry current releases and only
> `cu130` has sm_120. Use cu130.

## 3. Verify the GPU before training anything

This is worth 20 seconds. It confirms the build actually has kernels for your
card rather than silently falling back:

```powershell
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('device:', torch.cuda.get_device_name(0)); print('capability: sm_%d%d' % torch.cuda.get_device_capability(0)); print('compiled for:', torch.cuda.get_arch_list()); x = torch.randn(4000, 4000, device='cuda'); print('matmul ok:', float((x @ x).sum()))"
```

Expected: `capability: sm_120`, `sm_120` present in `compiled for:`, and
`matmul ok:` printing a number. If `sm_120` is missing from the compiled list,
the install is wrong — redo step 2. `scripts/train_detector.py` also checks this
and warns before wasting an hour.

## 4. Build the dataset

```powershell
python scripts/prepare_kitti.py --report-baseline
```

Downloads KITTI (390 MB), recovers frame provenance, and writes the
sequence-disjoint split. Takes a few minutes; idempotent.

**Re-running this on Windows is required, not optional.** `data/splits/*.txt`
hold absolute paths, so the copies generated on another machine will not
resolve. The split itself is deterministic — the same drives land on the same
sides on every machine — so only the paths differ.

## 5. Train

```powershell
python scripts/train_detector.py --model yolov8m.pt --epochs 60 --batch 16
```

Roughly an hour on a 5070. If you hit CUDA out-of-memory, drop `--batch` to 8.

On finishing it evaluates on held-out drives and writes:

- `results/perception/yolov8m/report.json` — per-class AP with instance, image, and drive counts
- `results/perception/yolov8m/curves.csv` — per-epoch training curves
- `models/yolov8m.pt` — the best checkpoint

Then push the results back:

```powershell
git add results/ models/yolov8m.pt
git commit -m "Phase 2: YOLOv8m trained on sequence-disjoint KITTI"
git push
```

## 6. CARLA (separate, whenever you get to it)

Download the **CARLA 0.9.16 Windows** build from the
[releases page](https://github.com/carla-simulator/carla/releases), unzip, and
run `CarlaUE4.exe`. If a city loads and WASD moves the camera, it works.

The CARLA 0.9.16 wheels are **cp312-only**, so the CARLA venv is Python
3.12 (this is the repo's `.venv` on the Windows machine — separate from any
3.13 training venv):

```powershell
pip install carla
```

Set `CARLA_ROOT` to the unzipped CARLA_0.9.16 folder so
`pathfinder/sim/carla_paths.py` can find CARLA's `agents` package.

## 7. Run the perception ablation on CARLA (issue #10)

One sitting, unattended once started. Everything it needs is in the repo after
`git pull` — the trained weights live at
`results/perception/yolov8m/weights/best.pt` and are committed.

```powershell
git pull
.\.venv\Scripts\Activate.ps1   # the CARLA venv, Python 3.12
```

Once per machine, put the Detector's dependencies into that same venv (they
must coexist with `carla` in one process):

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install ultralytics
```

Verify `sm_120` with the one-liner in step 3. Then start `CarlaUE4.exe`, wait
for the town to load, and run:

```powershell
python -m pathfinder.ablation --backend carla --episodes 10 --device cuda
```

That is 20 episodes of simulation (10 per arm, identical seeds) across
Town01/03/05 and four weathers. At the recorded 172–181 ticks/s
(`results/carla/probe.json`) the privileged arm is minutes; the Detector arm
runs YOLOv8m on every frame and dominates the wall-clock — budget an hour or
two and let it run unattended. Every finished
episode is checkpointed to `results/ablation/carla_report.partial.jsonl`, so a
crash costs one episode of progress, not the run — but there is no resume:
re-run the same command after fixing whatever crashed.

It writes two artifacts:

- `results/ablation/carla_report.json` — every number, spec, and seed
- `results/ablation/carla_report.md` — the generated write-up (scope label,
  perception boundary, infraction breakdown, fine-tuning-trigger verdict)

Then push them back:

```powershell
git add results/ablation
git commit -m "Run the GT-vs-Detector ablation on CARLA (#10)"
git push
```

## 8. BehaviorAgent reference Episode on CARLA (issue #16)

One short sitting, unattended once started — batch it at the start of any
CARLA sitting (it is the prerequisite for the DAgger sitting, #25). It scores
CARLA's own behaviour agent — **not project work**, recorded as such in the
artifact — over the ablation's exact 10 seeded episodes, and prints the
go / stop-and-reassess verdict against the recorded PurePursuit floor.

Start `CarlaUE4.exe`, wait for the town to load, then in the CARLA venv:

```powershell
git pull
.\.venv\Scripts\Activate.ps1   # the CARLA venv, Python 3.12
python -m pathfinder.reference_run --backend carla
```

No Detector runs, so this is the fast kind of arm — at the recorded 172–181
ticks/s (`results/carla/probe.json`) budget well under an hour. Every
finished episode's score lands in
`results/reference/carla_report.partial.jsonl` as it completes, so no data
is silently lost — but there is no resume: a crash means re-running the
whole command from episode 0 after fixing whatever crashed.

It writes two artifacts:

- `results/reference/carla_report.json` — every number, spec, and seed, plus
  the computed floor gate
- `results/reference/carla_report.md` — the generated write-up (scope label,
  not-project-work banner, infraction breakdown, gate verdict)

Then push them back:

```powershell
git add results/reference
git commit -m "Record the BehaviorAgent reference baseline on CARLA (#16)"
git push
```

Then tell the agent the run is done — the review is agent work: quoting the
score and the printed go / stop-and-reassess verdict on #16 and on #11,
**before** any training happens (the gate paragraph in #16 requires it).

## 9. Distributed-run runbook (issues #22 → #26)

The full distributed benchmark, as one replayable sitting: seed the suite
onto SQS, run the coordinator and workers, kill a worker mid-Episode, watch
the visibility timeout redeliver its Episode to a survivor, archive
telemetry to Parquet, and collect the one report artifact.

**Everything that differs between the LocalStack rehearsal (#22) and the
real-SQS run (#26) lives in the configuration block of step 9.1 — endpoint,
credentials, queue URLs, and the run/artifact labels. Every command from
step 9.2 on is identical in both sittings.** Two declared scope boundaries,
neither an endpoint difference: the CARLA worker (step 9.5) only runs in the
real sitting, because the rehearsal machine cannot run CARLA — the rehearsal
proves the same worker command with the kinematic backend; and telemetry +
warehouse stay on the `local` backends in **both** sittings, because Kinesis
and S3 are never applied to real AWS (zero-spend rule — SQS's permanent free
tier is the one exception).

Executed end to end against LocalStack on 2026-08-21 (macOS, kinematic
workers; the variable-assignment and line-continuation syntax below is
PowerShell's — on a POSIX shell drop the `$env:`/backtick decoration — but
the command tokens are identical):
`results/distributed/localstack_rehearsal.{json,md}` is the rehearsal's
artifact — 8/8 episodes, the killed worker's Episode redelivered and
completed by the survivor, DLQ empty.

Two terminals: **A** for the coordinator, **B** for everything else (the
real sitting adds **C** for the CARLA worker). Steps 9.3–9.7 run unattended;
total rehearsal time is ~2 minutes, dominated by the 30 s visibility
timeout.

### 9.1 Configuration (the only rehearsal↔real difference)

Rehearsal (LocalStack) — in terminal B:

```powershell
docker compose up -d localstack        # rehearsal only; wait for healthy
$env:AWS_ACCESS_KEY_ID = "test"        # LocalStack accepts any credentials
$env:AWS_SECRET_ACCESS_KEY = "test"
$SQS_ENDPOINT = "http://localhost:4566"
python scripts/provision_sqs.py --endpoint-url $SQS_ENDPOINT
# ^ the rehearsal stand-in for #18's Terraform apply: creates
#   pathfinder-episodes + pathfinder-episodes-dlq shaped exactly like
#   terraform/sqs.tf, and prints the two URLs to paste here:
$QUEUE_URL = "<QUEUE_URL printed above>"
$DLQ_URL = "<DLQ_URL printed above>"
$RUN_ID = "localstack-rehearsal"
$REPORT = "results/distributed/localstack_rehearsal.json"
```

Real run — credentials from `aws configure` (issue #18's wizard —
`bash scripts/sqs_apply_wizard.sh`, run once on any machine with Terraform;
then `aws configure` on *this* machine with the same keys if the wizard ran
elsewhere). The wizard prints this exact block with the real URLs filled in
— keep the queues at its closing keep-or-destroy decision, since destroying
them invalidates the printed URLs until the wizard is re-run:

```powershell
$SQS_ENDPOINT = "https://sqs.us-east-1.amazonaws.com"
$QUEUE_URL = "<episodes queue URL from the #18 Terraform output>"
$DLQ_URL = "<DLQ URL from the #18 Terraform output>"
$RUN_ID = "sqs-live"
$REPORT = "results/distributed/live_report.json"
```

The endpoint flag is passed in **both** sittings — the real run points it at
the real SQS endpoint — so the commands below never change shape.

### 9.2 Coordinator — terminal A

```powershell
python -m pathfinder.rpc.server --port 50051 --episodes-total 8 --run-id $RUN_ID
```

Leave it running; every later step talks to it. (`$RUN_ID` is defined in
terminal B — when using a separate terminal, type the label literally.)

### 9.3 Seed the queue — terminal B

```powershell
python scripts/enqueue_episodes.py --count 8 `
    --queue-backend sqs --queue-url $QUEUE_URL --queue-endpoint-url $SQS_ENDPOINT `
    --suite-out results/distributed/suite.json
```

The suite manifest is the enqueuer's own record of what went onto the queue;
step 9.6 reads it back so the report describes what was actually enqueued.

### 9.4 The chaos kill — terminal B

A kinematic Episode takes ~10 ms, so no by-hand kill can land mid-Episode;
`--chaos-kill-after-frames` is the deterministic version. This worker takes
one Episode off the queue, drives 200 frames, and dies the way a SIGKILL'd
worker dies (exit code 70, no cleanup, no queue delete, no telemetry flush).
Its Episode stays invisible for the 30 s visibility timeout, then redelivers.
Keep the frame count under the telemetry batch size (500) so the victim's
partial telemetry dies with it — above that, its already-flushed rows would
land in the warehouse alongside the survivor's re-run of the same Episode:

```powershell
python scripts/run_worker.py --worker-id chaos-victim --coordinator localhost:50051 `
    --queue-backend sqs --queue-url $QUEUE_URL --queue-endpoint-url $SQS_ENDPOINT `
    --simulator-backend kinematic --policy pure_pursuit `
    --telemetry-backend local --telemetry-local-root ./telemetry `
    --chaos-kill-after-frames 200
```

### 9.5 The surviving worker(s) — terminal B (and C in the real sitting)

`--idle-timeout-seconds 45` must outlast the 30 s visibility timeout: the
survivor drains the seven visible Episodes in seconds, then keeps polling
until the victim's Episode reappears, completes it, and exits.

```powershell
python scripts/run_worker.py --worker-id kinematic-1 --coordinator localhost:50051 `
    --queue-backend sqs --queue-url $QUEUE_URL --queue-endpoint-url $SQS_ENDPOINT `
    --simulator-backend kinematic --policy pure_pursuit `
    --telemetry-backend local --telemetry-local-root ./telemetry `
    --idle-timeout-seconds 45
```

**Real sitting only** — the CARLA worker, in terminal C, started *before*
step 9.4 so it holds an Episode while the kinematic worker drains the rest
(CarlaUE4.exe running, same flags as terminal B's config block):

```powershell
python scripts/run_worker.py --worker-id carla-1 --coordinator localhost:50051 `
    --queue-backend sqs --queue-url $QUEUE_URL --queue-endpoint-url $SQS_ENDPOINT `
    --simulator-backend carla --policy pure_pursuit `
    --telemetry-backend local --telemetry-local-root ./telemetry `
    --idle-timeout-seconds 45
```

### 9.6 Archive telemetry + collect the report — terminal B

One command: drains the telemetry stream into partitioned Parquet under
`./warehouse`, queries the coordinator for status and results, reads the
queue's redelivery/DLQ state, and lands report JSON + generated write-up
together:

```powershell
python -m pathfinder.distributed_run --coordinator localhost:50051 `
    --suite results/distributed/suite.json `
    --queue-backend sqs --queue-url $QUEUE_URL --dead-letter-queue-url $DLQ_URL `
    --queue-endpoint-url $SQS_ENDPOINT `
    --telemetry-backend local --telemetry-local-root ./telemetry `
    --object-backend local --object-local-root ./warehouse `
    --output $REPORT
```

Expected in the printed tail and the artifact: 8/8 episodes, **1
redelivery** (the victim's Episode, completed by a survivor, delivered 2
times), **0 dead-lettered**, depth 0. Anything else is a finding — read the
report's suite cross-check section before trusting the run.

### 9.7 Shut down and check in — terminal B

Stop the coordinator (Ctrl+C in terminal A). Rehearsal only:
`docker compose stop localstack`. Then commit the artifact pair and the
suite manifest:

```powershell
git add results/distributed
git commit -m "Execute the distributed-run rehearsal against LocalStack (#22)"
git push
```

(The real sitting commits `live_report.{json,md}` and quotes its numbers on
#12/#26 instead; `./telemetry` and `./warehouse` are working data and stay
untracked.)

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `no kernel image is available` | torch built without sm_120 | Reinstall from cu130 (step 2) |
| `torch.cuda.is_available()` is False | CPU-only wheel | Reinstall from cu130; check `--index-url` was used |
| CUDA out of memory | batch too large for 12 GB | `--batch 8` |
| `cannot be loaded because running scripts is disabled` | PowerShell execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Dataloader hangs at 0% | Windows worker spawn | `--workers 0` |
| Images not found during training | split files from another machine | Re-run step 4 |
