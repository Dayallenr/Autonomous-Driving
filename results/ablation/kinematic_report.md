# Perception ablation — kinematic backend

Generated 2026-08-20T08:12:06+00:00 from [`results/ablation/kinematic_report.json`](kinematic_report.json), which records every number below along with the full episode specifications needed to re-run it. This file is rendered from that artifact by `pathfinder/ablation_writeup.py`; edit the generator, not this file.

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
| Driving score | 48.3 | 48.26 | 0.04 |
| Route completion | 0.483 | 0.4826 | 0.0004 |
| Infraction penalty | 1.0 | 1.0 | 0.0 |
| Collisions per km | 0.0 | 0.0 | — |
| Failures | 0 | 0 | — |
| Mean perception latency (ms) | 0.0 | 97.433 | — |

A positive difference means the candidate's perception cost driving performance.

## Infraction breakdown

Neither arm committed a scoreable or tracked infraction.

## Per-episode driving score

| Episode | Town | Weather | Seed | Baseline | Candidate | Delta | Flag |
|---|---|---|---|---|---|---|---|
| ep-0000 | Town01 | ClearNoon | 1000 | 52.21 | 52.21 | 0.0 |  |
| ep-0001 | Town03 | ClearSunset | 1001 | 42.01 | 41.89 | 0.12 |  |
| ep-0002 | Town05 | HardRainNoon | 1002 | 50.69 | 50.69 | 0.0 |  |
