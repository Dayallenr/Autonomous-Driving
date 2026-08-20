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

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `no kernel image is available` | torch built without sm_120 | Reinstall from cu130 (step 2) |
| `torch.cuda.is_available()` is False | CPU-only wheel | Reinstall from cu130; check `--index-url` was used |
| CUDA out of memory | batch too large for 12 GB | `--batch 8` |
| `cannot be loaded because running scripts is disabled` | PowerShell execution policy | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |
| Dataloader hangs at 0% | Windows worker spawn | `--workers 0` |
| Images not found during training | split files from another machine | Re-run step 4 |
