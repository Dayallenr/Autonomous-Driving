# %% [markdown]
# # PathFinder — Conditional Imitation Learning + DAgger (GPU training)
#
# Trains the 4-branch ResNet-18 CIL planner with DAgger against a privileged
# pure-pursuit expert, inside the deterministic kinematic simulator.
#
# When it finishes, download `cil_dagger.pt` from the last cell and drop it at:
#
# ```
# Autonomous-Driving/models/cil_dagger.pt
# ```
#
# **Runtime:** `Runtime → Change runtime type → T4 GPU` (or better).
#
# ---
# ## Why this needs a GPU
#
# A CPU run of `pathfinder.dagger` is a smoke test, not a result. On a few
# thousand frames with a randomly-initialised backbone, the student is badly
# underfit and expert-student disagreement typically *rises* across iterations.
# That is the expected ordering of two competing effects:
#
# * as β decays the student drives more and reaches worse states, where the
#   expert takes more extreme corrective actions → disagreement **up**;
# * as the network fits the aggregate dataset it predicts those corrections
#   better → disagreement **down**.
#
# The second only wins once the model can actually fit: ImageNet initialisation,
# tens of thousands of frames, tens of epochs. That is this notebook.
#
# ## Why the expert is "privileged"
#
# The expert reads exact cross-track error, heading error, and path curvature
# straight from the simulator. The student sees only rendered pixels. The student
# is therefore learning to *infer from images* what the expert is handed
# directly — the privileged-expert setup used by LBC and Roach.

# %%
# Unlike Sentinel's notebook, this one cannot be self-contained: DAgger needs the
# simulator, renderer, expert, and metrics. Cloning is far more reliable than
# pasting several thousand lines into cells.
REPO_URL = "https://github.com/Dayallenr/Autonomous-Driving.git"

import os
import subprocess
import sys

if not os.path.exists("Autonomous-Driving"):
    result = subprocess.run(["git", "clone", "--depth", "1", REPO_URL], capture_output=True, text=True)
    print(result.stdout or result.stderr)
    if result.returncode != 0:
        print(
            "\nClone failed. If the repo is private, either:\n"
            "  1. upload the project as a zip and run:  !unzip -q project.zip\n"
            "  2. or mount Drive and copy it in.\n"
            "Then make sure the working directory below points at the project root."
        )

os.chdir("Autonomous-Driving")
sys.path.insert(0, os.getcwd())
print("cwd:", os.getcwd())

# %%
# torch/torchvision are preinstalled on Colab; only the small extras are needed.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "opencv-python-headless"], check=False)

import torch

print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    DEVICE = "cuda"
else:
    print("WARNING: no GPU — this will be slow and may not converge in reasonable time.")
    DEVICE = "cpu"

# %% [markdown]
# ## Sanity check: the simulator and renderer
#
# Confirm the environment produces the `(88, 200, 3)` frames the ResNet expects
# before committing to a long run.

# %%
import matplotlib.pyplot as plt

from pathfinder.runner import PurePursuitPlanner
from pathfinder.sim import EpisodeSpec, KinematicSimulator

simulator = KinematicSimulator(render=True)
state = simulator.reset(EpisodeSpec(episode_id="preview", route_length_m=300, seed=3))
expert = PurePursuitPlanner()

frames = []
for step in range(300):
    control = expert.plan(state)
    result = simulator.step(control.throttle, control.steer, control.brake)
    state = result.state
    if step % 60 == 0:
        frames.append((step, state.image.copy(), state.command))
    if result.done:
        break
simulator.close()

figure, axes = plt.subplots(1, len(frames), figsize=(4 * len(frames), 3))
for axis, (step, image, command) in zip(axes if len(frames) > 1 else [axes], frames):
    axis.imshow(image)
    axis.set_title(f"step {step} · {command.name}")
    axis.axis("off")
plt.tight_layout()
plt.show()

print("frame shape:", frames[0][1].shape, frames[0][1].dtype)

# %% [markdown]
# ## Baseline: what the expert scores
#
# The expert is the ceiling the student is imitating. Establish it first — a
# student number is meaningless without it.

# %%
from pathfinder.metrics.driving_score import aggregate
from pathfinder.runner import run_episode

simulator = KinematicSimulator(render=False)  # no pixels needed for the expert
expert_scores = []
for index in range(12):
    spec = EpisodeSpec(
        episode_id=f"expert-{index}", route_length_m=300, max_steps=1200, seed=5000 + index
    )
    expert_scores.append(run_episode(simulator, spec, PurePursuitPlanner()))
simulator.close()

expert_summary = aggregate(expert_scores)
print("EXPERT (privileged, the imitation ceiling)")
for key, value in expert_summary.to_dict().items():
    print(f"  {key:22} {value}")

# %% [markdown]
# ## Train
#
# Settings below are sized for a T4. The knobs that matter most:
#
# * `pretrained=True` — ImageNet initialisation. This is the single biggest
#   difference from the CPU smoke test; from scratch, this much data is not
#   enough to learn useful visual features.
# * `episodes_per_iteration` and `iterations` — how much of the student's own
#   distribution gets aggregated. More iterations matter more than more epochs.
# * `epochs_per_iteration` — enough to actually fit the aggregate each round.

# %%
import time

from pathfinder.dagger import DAggerConfig, run_dagger
from planning.cil_model import CILModel

config = DAggerConfig(
    iterations=8,
    episodes_per_iteration=8,
    route_length_m=300.0,
    max_steps=1200,
    epochs_per_iteration=12,
    batch_size=64,
    learning_rate=3e-4,
    beta_decay=0.6,
    device=DEVICE,
    seed=7,
    max_dataset_size=120_000,
)

# ImageNet initialisation — the key difference from the CPU smoke test.
model = CILModel(pretrained=True, output_mode="control")

started = time.perf_counter()
model, report = run_dagger(config, model=model)
print(f"\ntotal wall time: {(time.perf_counter() - started) / 60:.1f} min")

# %% [markdown]
# ## Results
#
# The number the project claims is **error reduction**: how much
# expert-student disagreement fell from the first measured iteration to the last.
#
# Iteration 0 has no disagreement value by construction — there is no student yet.

# %%
import json

print(json.dumps(report.to_dict(), indent=2))

rows = [item for item in report.iterations if item.disagreement == item.disagreement]  # drop NaN
if len(rows) >= 2:
    first, last = rows[0].disagreement, rows[-1].disagreement
    print(f"\nfirst measured disagreement : {first:.4f}")
    print(f"last measured disagreement  : {last:.4f}")
    print(f"error reduction             : {(first - last) / first * 100:+.1f}%")
    if last >= first:
        print(
            "\nDisagreement did not fall. Before quoting anything, try:\n"
            "  - more iterations (aggregation is the mechanism, not epochs)\n"
            "  - more epochs_per_iteration (check train_loss is actually decreasing)\n"
            "  - a lower learning rate if train_loss is oscillating\n"
            "Report what you measure. A negative result is a result."
        )

# %%
figure, axes = plt.subplots(1, 3, figsize=(15, 4))
iterations = [item.iteration for item in report.iterations]

axes[0].plot(iterations, [item.train_loss for item in report.iterations], marker="o")
axes[0].set_title("train loss (L1 on active branch)")
axes[0].set_xlabel("DAgger iteration")

axes[1].plot(
    [item.iteration for item in rows], [item.disagreement for item in rows],
    marker="o", color="crimson",
)
axes[1].set_title("expert-student disagreement\n(on the student's own states)")
axes[1].set_xlabel("DAgger iteration")

axes[2].plot(
    iterations, [item.route_completion for item in report.iterations], marker="o", color="seagreen"
)
axes[2].axhline(expert_summary.route_completion, linestyle="--", color="gray", label="expert")
axes[2].set_title("route completion")
axes[2].set_xlabel("DAgger iteration")
axes[2].legend()

for axis in axes:
    axis.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Evaluate the student on held-out routes
#
# Scored on seeds the DAgger loop never trained on, so this is generalization
# rather than recall.

# %%
from pathfinder.dagger import CILPlanner

simulator = KinematicSimulator(render=True)
student_scores = []
for index in range(12):
    spec = EpisodeSpec(
        episode_id=f"student-{index}", route_length_m=300, max_steps=1200, seed=5000 + index
    )
    student_scores.append(
        run_episode(simulator, spec, CILPlanner(model, device=DEVICE), model_version="cil_dagger")
    )
simulator.close()

student_summary = aggregate(student_scores)
print(f"{'metric':24} {'expert':>12} {'student':>12}")
print("-" * 50)
for key in ("driving_score", "route_completion", "collisions_per_km", "failures"):
    print(
        f"{key:24} {expert_summary.to_dict()[key]:>12} {student_summary.to_dict()[key]:>12}"
    )

# %% [markdown]
# ## Save the checkpoint
#
# Saved with the architecture config and the full training report, so the
# artifact is self-describing and the repo can verify compatibility on load.

# %%
CHECKPOINT_PATH = "cil_dagger.pt"

torch.save(
    {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": {
            "output_mode": "control",
            "num_waypoints": model.num_waypoints,
            "input_height": 88,
            "input_width": 200,
            "branches": ["FOLLOW_LANE", "TURN_LEFT", "TURN_RIGHT", "GO_STRAIGHT"],
        },
        "metadata": {
            "dagger": report.to_dict(),
            "expert_summary": expert_summary.to_dict(),
            "student_summary": student_summary.to_dict(),
            "trained_on": DEVICE,
            "simulator_backend": "kinematic",
        },
    },
    CHECKPOINT_PATH,
)

print(f"saved {CHECKPOINT_PATH} ({os.path.getsize(CHECKPOINT_PATH) / 1e6:.1f} MB)")

# %%
# Verify it reloads exactly as the repo will load it.
reloaded = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
verify = CILModel(pretrained=False, output_mode=reloaded["config"]["output_mode"])
verify.load_state_dict(reloaded["state_dict"])
verify.eval()

with torch.inference_mode():
    branches = verify(torch.randn(1, 3, 88, 200))
print("checkpoint reloads cleanly")
print("branches:", len(branches), "each", tuple(branches[0].shape))

# %%
try:
    from google.colab import files  # type: ignore

    files.download(CHECKPOINT_PATH)
except ImportError:
    print(f"not in Colab — checkpoint is at ./{CHECKPOINT_PATH}")
