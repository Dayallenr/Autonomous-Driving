# Distributed benchmark run `localstack-rehearsal` — kinematic backend

Generated 2026-08-21T22:50:53+00:00 from [`results/distributed/localstack_rehearsal.json`](localstack_rehearsal.json), which records every number below along with the full episode specifications needed to re-run it. This file is rendered from that artifact by `pathfinder/distributed_writeup.py`; edit the generator, not this file.

> **Scope: pipeline-only.** Generated on the kinematic backend, which neither simulates real physics nor renders real scenes. These numbers verify the distributed benchmark pipeline — queue, workers, coordinator, telemetry warehouse end to end; they are not driving quality and must never be quoted as such. The real measurement comes from the CARLA backend.

## What this measures

One benchmark run through the distributed pipeline: episodes seeded onto a work queue, pulled by workers that register and heartbeat over gRPC, scored, submitted to the coordinator, and their telemetry archived to partitioned Parquet. The mechanics — redelivery after a worker death, dead-lettering, provenance on every row — are the subject; the driving aggregate means only what the scope banner above allows.

## Suite

- Episodes: 8
- Towns: Town01, Town03, Town05
- Weathers: ClearNoon, ClearSunset, HardRainNoon, WetNoon
- Seeds: 1000–1007
- Suite source: suite manifest results/distributed/suite.json

## Coordinator

- Run id: `localstack-rehearsal`
- Episodes: 8 completed of 8
- Duplicate submissions: 0
- Elapsed: 101.33 s

| Worker | Episodes completed | Healthy at collection |
|---|---|---|
| chaos-victim | 0 | no |
| kinematic-1 | 8 | yes |

A worker unhealthy at collection stopped heartbeating — either it finished and exited, or it died. Its completed episodes are counted above either way; an episode it held when it died reappears in the redelivery count below, which is the queue doing its one real job.

## Queue

- Redeliveries: 1 (delivery is at-least-once; a count above zero means an episode outlived the worker that first received it and was completed by another)
- Dead-lettered: 0
- Approximate depth after the run: 0
  - `ep-0000` was delivered 2 times and completed by `kinematic-1`

## Telemetry warehouse

- Rows written: 11691
- Parquet files: 1
- Partitioning: hive-style telemetry/frames/event_date=YYYY-MM-DD/part-*.parquet, one file per date partition per drain

## Aggregate

| Metric | Value |
|---|---|
| Episodes | 8 |
| Driving score | 94.11 |
| Route completion | 0.9411 |
| Infraction penalty | 1.0 |
| Collisions per km | 0.0 |
| Failures | 0 |

## Infraction breakdown

No scoreable or tracked infraction was committed.

## Per-episode results

| Episode | Worker | Backend | Policy | Deliveries | Driving score | Status |
|---|---|---|---|---|---|---|
| ep-0000 | kinematic-1 | kinematic | pure_pursuit | 2 | 100.0 | completed |
| ep-0001 | kinematic-1 | kinematic | pure_pursuit | 1 | 83.04 | completed |
| ep-0002 | kinematic-1 | kinematic | pure_pursuit | 1 | 92.98 | completed |
| ep-0003 | kinematic-1 | kinematic | pure_pursuit | 1 | 100.0 | completed |
| ep-0004 | kinematic-1 | kinematic | pure_pursuit | 1 | 94.18 | completed |
| ep-0005 | kinematic-1 | kinematic | pure_pursuit | 1 | 99.16 | completed |
| ep-0006 | kinematic-1 | kinematic | pure_pursuit | 1 | 100.0 | completed |
| ep-0007 | kinematic-1 | kinematic | pure_pursuit | 1 | 83.54 | completed |

## Suite cross-check

Every enqueued episode has exactly one result, and every result belongs to the recorded suite.
