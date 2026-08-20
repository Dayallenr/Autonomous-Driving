# PathFinder

An autonomous driving system developed and evaluated in the CARLA simulator,
built so that its results can be reproduced by anyone who clones the repo.

## Language

### Models

**Detector**:
The object-detection model trained on KITTI that finds vehicles, pedestrians,
and cyclists in a single camera image.
_Avoid_: the model, the network, perception model

**Policy**:
The model that maps an observation to a steering, throttle, and brake command.
_Avoid_: the model, the planner, the agent, the driver

Both are models, so **"the model" is never a valid term on its own** — say which
one. A claim about one is not a claim about the other.

### Simulation

**Probe**:
A check that the simulator is reachable and behaves deterministically, run with
no Policy in the loop. Establishes that the environment works, never that the
system drives.
_Avoid_: test, smoke test, CARLA test

**Episode**:
One seeded route driven from start to finish by a Policy, producing a route
completion figure and an infraction record.
_Avoid_: run, trial, simulation, test

"Tested in CARLA" is ambiguous between these two and should not be used. A Probe
is not an Episode; only Episodes say anything about driving quality.
