# BehaviorAgent reference baseline — carla backend

Generated 2026-08-21T21:50:48+00:00 from [`results\reference\carla_report.json`](carla_report.json), which records every number below along with the full episode specifications needed to re-run it. This file is rendered from that artifact by `pathfinder/reference_writeup.py`; edit the generator, not this file.

> **Scope: driving-quality.** Generated on the CARLA backend: the score measures the behaviour agent's driving quality over the recorded suite. It is a reference ceiling, not project work.
>
> CARLA's own behaviour agent — **not project work**. It reads the simulator's world state directly and appears only as a reference upper bound produced by the same routes, traffic, and scoring; nothing it does may be presented as this project's driving.

## What this measures

`carla_builtin_behavior_agent` (behaviour preset `normal`) driving the perception ablation's seeded suite. Its score is the reference ceiling for the Phase 3 comparison table and the teacher-quality measurement the DAgger training design depends on — an upper bound to cite alongside project results, never as one.

## Suite

- Episodes: 10
- Towns: Town01, Town03, Town05
- Weathers: ClearNoon, ClearSunset, HardRainNoon, WetNoon
- Seeds: 1000–1009

## Results

| Metric | `carla_builtin_behavior_agent` |
|---|---|
| Driving score | 31.33 |
| Route completion | 0.368 |
| Infraction penalty | 0.896 |
| Collisions per km | 2.038 |
| Failures | 0 |
| Mean policy latency (ms) | 1.907 |

## Infraction breakdown

| Infraction | Count |
|---|---|
| agent_blocked | 6 |
| collision_vehicle | 3 |

## Per-episode driving score

| Episode | Town | Weather | Seed | Driving score | Flag |
|---|---|---|---|---|---|
| ep-0000 | Town01 | ClearNoon | 1000 | 78.32 |  |
| ep-0001 | Town03 | ClearSunset | 1001 | 23.35 |  |
| ep-0002 | Town05 | HardRainNoon | 1002 | 21.98 |  |
| ep-0003 | Town01 | WetNoon | 1003 | 45.07 |  |
| ep-0004 | Town03 | ClearNoon | 1004 | 9.58 |  |
| ep-0005 | Town05 | ClearSunset | 1005 | 26.85 |  |
| ep-0006 | Town01 | HardRainNoon | 1006 | 47.36 |  |
| ep-0007 | Town03 | WetNoon | 1007 | 16.69 |  |
| ep-0008 | Town05 | ClearNoon | 1008 | 8.65 |  |
| ep-0009 | Town01 | ClearSunset | 1009 | 35.47 |  |

## Floor gate — go / stop-and-reassess (issue #16)

Issue #16 gates the training plan on the behaviour agent's live score being 'meaningfully above' the recorded privileged-PurePursuit floor, without defining 'meaningfully'. This project operationalises it, pre-registered before any live run, as a margin of at least 10.0 driving-score points over the identical seeded suite; anything less is 'stop-and-reassess' — the teacher would be too close to the floor for imitating it to be worth a training run.

- Floor: 25.2 — pure_pursuit under privileged perception, from `results\ablation\carla_report.json`
- Reference: 31.33
- Margin: 6.13 (required: 10.0)

**Verdict: stop-and-reassess**
