# Modular perception → planning → control, not end-to-end

PathFinder drives via a modular stack — the Detector feeds a planner, the
planner emits waypoints, a controller tracks them — rather than an end-to-end
network mapping pixels directly to steering. The decisive factor is that under
an end-to-end policy the KITTI-trained Detector plays no part in driving at all;
it becomes an artifact sitting beside the system instead of inside it. A modular
stack also fails legibly, which matters because simulation runs happen
unattended overnight and an opaque failure costs a full day.

## Considered options

- **End-to-end conditional imitation learning (CIL) + DAgger.** Already
  scaffolded in `pathfinder/planning/cil_model.py` and `pathfinder/dagger.py`.
  Rejected as the primary architecture: it orphans the Detector, offers almost
  no debugging surface, and has little interface structure to speak of.
- **Modular (chosen).** More rebuild work — the previous modular stack was
  deleted — and it exposes a KITTI→CARLA domain gap that end-to-end would have
  sidestepped, but both costs buy something back.

The CIL policy is **not** abandoned. It returns as a compared baseline once the
modular stack produces a driving score, which is what turns the planned
ground-truth-vs-Detector ablation into a measurement of how much perception
quality is actually worth.

## Consequences

The Detector now runs on synthetic CARLA imagery while trained on real-world
KITTI photographs. That domain gap is real, currently unmeasured, and must be
addressed deliberately rather than discovered late.
