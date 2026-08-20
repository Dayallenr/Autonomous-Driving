# The observation-to-control interface is named Policy, not Planner

`pathfinder/runner.py` defines the interface every Policy satisfies:
`plan(state) -> ControlOutput`. It was called `Planner`. It is now called
`Policy`, and `pathfinder/policies.py` — `build_policy`, `POLICY_NAMES`,
`PurePursuitPolicy`, `CILPolicy`, `CarlaBehaviorAgentPolicy` — follows.

The decisive factor is that this project already uses "planner" for something
else. ADR-0001 describes the chosen architecture as "the Detector feeds a
planner, the planner emits waypoints, a controller tracks them" — three stages,
of which the planner is one. What `runner.py` names is not that stage. It is the
whole mapping from observation to throttle, steer, and brake: `PurePursuitPolicy`
plans *and* controls, and `CILPolicy` does both in a single network. Calling it
`Planner` claimed a name that ADR-0001 had already given to a part of it.

`CONTEXT.md` had said as much since it was written — it defines **Policy** as
"the model that maps an observation to a steering, throttle, and brake command"
and lists "the planner" among the terms to avoid. The code contradicted the
glossary in every file. That drift was flagged as an open question on the
perception-seam epic with the instruction to settle it deliberately rather than
let it persist; this is that decision.

## Considered options

- **Amend the glossary to bless `Planner`.** Cheapest, and it would have made
  the repo self-consistent in an afternoon. Rejected because it resolves the
  contradiction in favour of the wrong term: it would leave ADR-0001's pipeline
  "planner" and the interface `Planner` sharing a name while meaning different
  things, which is the ambiguity the glossary exists to prevent.
- **Rename in code (chosen).** A wide mechanical change — the protocol, three
  implementations, the registry, the worker CLI flag, the telemetry column
  `planner_latency_ms` → `policy_latency_ms`, and the prose in every module that
  mentioned it. Done before the perception seam is built on top of the interface,
  because every later ticket makes the rename larger.

## Consequences

`--planner` on `scripts/run_worker.py` is now `--policy`, and the Parquet/Redshift
column `planner_latency_ms` is now `policy_latency_ms`. Neither is a breaking
change in practice: no telemetry has been written to a real warehouse and no
deployment passes the old flag. The DDL is generated from `FRAME_COLUMNS`, so
`redshift_ddl()` and `athena_ddl()` follow automatically.

`pathfinder/planning/cil_model.py` keeps its package name. It holds a network
architecture rather than the interface, and moving it would churn the DAgger
module and the Colab notebook for no gain in the vocabulary this ADR is about.
Renaming that package is a loose end, not a contradiction.

"Planner" remains correct in exactly two places, both of them CARLA's:
`GlobalRoutePlanner`, which really does plan routes, and CARLA's `LocalPlanner`,
referenced in a comment in `pathfinder/policies.py`.
