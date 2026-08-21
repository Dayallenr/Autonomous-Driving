# Perception ablation — kinematic backend

Generated 2026-08-21T19:46:06+00:00 from [`results/ablation/kinematic_report.json`](kinematic_report.json), which records every number below along with the full episode specifications needed to re-run it. This file is rendered from that artifact by `pathfinder/ablation_writeup.py`; edit the generator, not this file.

> **Scope: pipeline-only.** Generated on the kinematic backend, which neither simulates real physics nor renders real scenes. These numbers verify the ablation pipeline end to end; they are not driving quality and must never be quoted as such. The real measurement comes from the CARLA backend.

## What differs between the arms

Both arms drive identical seeded episodes — same routes, traffic, weather, controller (`pure_pursuit`), and scoring — and differ only in perception. The perception boundary is: **perceived obstacles, privileged localization and traffic lights**. The Detector arm earns obstacle ranges from camera pixels; localization and traffic lights stay privileged in *both* arms because the Detector has no traffic-light class and localization is a map problem. Claiming more than this boundary would be false.

## Suite

- Episodes per arm: 3
- Towns: Town01, Town03, Town05
- Weathers: ClearNoon, ClearSunset, HardRainNoon
- Seeds: 1000–1002

## Results

| Metric | Baseline (privileged) | Candidate (detector) | Difference (B − C) |
|---|---|---|---|
| Driving score | 92.01 | 92.0 | 0.01 |
| Route completion | 0.9201 | 0.92 | 0.0001 |
| Infraction penalty | 1.0 | 1.0 | 0.0 |
| Collisions per km | 0.0 | 0.0 | — |
| Failures | 0 | 0 | — |
| Mean perception latency (ms) | 0.0 | 102.357 | — |

A positive difference means the candidate's perception cost driving performance.

## Infraction breakdown

Neither arm committed a scoreable or tracked infraction.

## Per-episode driving score

| Episode | Town | Weather | Seed | Baseline | Candidate | Delta | Flag |
|---|---|---|---|---|---|---|---|
| ep-0000 | Town01 | ClearNoon | 1000 | 100.0 | 100.0 | 0.0 |  |
| ep-0001 | Town03 | ClearSunset | 1001 | 83.04 | 83.02 | 0.03 |  |
| ep-0002 | Town05 | HardRainNoon | 1002 | 92.98 | 92.98 | 0.0 |  |
