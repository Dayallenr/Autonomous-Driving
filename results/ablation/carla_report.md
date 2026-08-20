# Perception ablation — carla backend

Generated 2026-08-20T23:32:11+00:00 from [`results\ablation\carla_report.json`](carla_report.json), which records every number below along with the full episode specifications needed to re-run it. This file is rendered from that artifact by `pathfinder/ablation_writeup.py`; edit the generator, not this file.

> **Scope: driving-quality.** Generated on the CARLA backend: scores measure driving quality under the stated perception boundary (perceived obstacles, privileged localization and traffic lights).

## What differs between the arms

Both arms drive identical seeded episodes — same routes, traffic, weather, controller (`pure_pursuit`), and scoring — and differ only in perception. The perception boundary is: **perceived obstacles, privileged localization and traffic lights**. The Detector arm earns obstacle ranges from camera pixels; localization and traffic lights stay privileged in *both* arms because the Detector has no traffic-light class and localization is a map problem. Claiming more than this boundary would be false.

## Suite

- Episodes per arm: 10
- Towns: Town01, Town03, Town05
- Weathers: ClearNoon, ClearSunset, HardRainNoon, WetNoon
- Seeds: 1000–1009

## Results

| Metric | Baseline (privileged) | Candidate (detector) | Difference (B − C) |
|---|---|---|---|
| Driving score | 25.2 | 8.86 | 16.34 |
| Route completion | 0.3961 | 0.4812 | -0.0851 |
| Infraction penalty | 0.7448 | 0.1413 | 0.6035 |
| Collisions per km | 2.525 | 26.498 | — |
| Failures | 0 | 0 | — |
| Mean perception latency (ms) | 0.004 | 8.351 | — |

A positive difference means the candidate's perception cost driving performance.

## Infraction breakdown

| Infraction | Baseline | Candidate |
|---|---|---|
| agent_blocked | 2 | 1 |
| collision_static | 4 | 4 |
| collision_vehicle | 0 | 47 |
| lane_invasion | 16 | 42 |
| red_light | 8 | 8 |

## Per-episode driving score

| Episode | Town | Weather | Seed | Baseline | Candidate | Delta | Flag |
|---|---|---|---|---|---|---|---|
| ep-0000 | Town01 | ClearNoon | 1000 | 42.91 | 3.38 | 39.53 |  |
| ep-0001 | Town03 | ClearSunset | 1001 | 34.38 | 8.63 | 25.76 |  |
| ep-0002 | Town05 | HardRainNoon | 1002 | 47.13 | 8.04 | 39.08 |  |
| ep-0003 | Town01 | WetNoon | 1003 | 9.63 | 0.36 | 9.27 |  |
| ep-0004 | Town03 | ClearNoon | 1004 | 12.44 | 12.42 | 0.01 |  |
| ep-0005 | Town05 | ClearSunset | 1005 | 28.26 | 1.35 | 26.92 |  |
| ep-0006 | Town01 | HardRainNoon | 1006 | 5.2 | 0.51 | 4.68 |  |
| ep-0007 | Town03 | WetNoon | 1007 | 21.69 | 52.66 | -30.97 |  |
| ep-0008 | Town05 | ClearNoon | 1008 | 14.28 | 0.97 | 13.31 |  |
| ep-0009 | Town01 | ClearSunset | 1009 | 36.04 | 0.22 | 35.81 |  |

The Detector was trained only on real KITTI imagery, so every CARLA town and weather above is unseen by it — synthetic renders are a genuine domain shift, and a large gap is the expected outcome. The measurement is the deliverable either way.

## Known measurement divergences

Both arms measure the same range convention (forward-frustum, camera-origin, ground-plane distance to the obstacle's nearest visible surface; issue #10), so the gap between them is perception — with one floor worth naming: below the camera's `min_measurable_range_m` the obstacle's ground contact leaves the frame, the Detector's range saturates and over-reads, while the privileged arm still reads the true range. That divergence *is* a perception limitation and, alongside detection quality itself, is part of the measured cost.

## Is perception the binding constraint?

Rule, fixed before the run: perception is the **binding constraint** on driving score when the score it costs (baseline minus candidate) exceeds the baseline's own shortfall from a perfect 100 — that is, when imperfect perception loses more score than everything else in the stack combined. The rule is conservative under a weak baseline: the lower the baseline's own score, the more perception must cost before it counts as binding. Episodes flagged as failed measure a mid-run failure, not perception, and are excluded from this verdict. The verdict is computed from the report artifact at render time, so revising the rule only means regenerating this file — never re-running the simulation.

Over the 10 clean episode(s): perception cost 16.34 driving-score points; the baseline's own shortfall is 74.80 points.

**Perception is not the binding constraint on driving score in this run** — the controller and the rest of the stack cost more than perception did. The deferred Detector fine-tuning decision is not triggered by this run.
