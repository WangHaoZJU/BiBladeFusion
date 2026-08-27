# HoloRobot robot-stack migration

Status: active implementation plan

## Decision

BiBladeFusion owns the thin-blade reconstruction task: bilateral proxy construction,
front/back partitioning, coverage evidence, view generation, view scoring, depth
comparison, and future thermal fusion.

HoloRobot is the source of truth for the robot stack:

- Elite CS68 hardware lifecycle and motion commands;
- robot-domain joint and pose contracts;
- CS68 kinematics, URDF, STL, joint limits, and collision geometry;
- inverse kinematics, path planning, trajectory generation, collision checking, and
  execution preflight;
- guarded execution, approval, stop, and fail-closed behavior.

The integration boundary is:

```text
BiBladeFusion planned camera pose
  -> calibrated camera/tool transform
  -> HoloRobot MotionGoal
  -> reachability / collision / path / trajectory preflight
  -> explicit operator approval
  -> HoloRobot guarded Elite execution
```

BiBladeFusion must not add a second robot kinematics, collision, trajectory, or Elite
motion implementation after this migration begins. Existing implementations remain only
until their callers have moved to the HoloRobot-backed adapter.

## Coordinate conventions

The HoloRobot conventions are authoritative for this integration, including its current
Elite TCP pose interpretation. Internal orientations use normalized quaternions in
`(x, y, z, w)` order. Conversion at the Elite backend boundary follows HoloRobot rather
than independently interpreting the SDK documentation.

Canonical frames for the initial single-arm cell are:

- `elite_b_base`: CS68 base/world frame;
- `elite_b_flange`: physical flange frame;
- `elite_b_tcp`: configured controller TCP frame;
- `elite_b_wrist_d435i_optical`: D435i optical frame.

Calibration artifacts remain installation-specific. A HoloRobot artifact may be reused
only when its robot identity, camera identity, mounting state, coordinate frames, and
acceptance status all match the current installation.

## Source and dependency policy

The required HoloRobot robot-stack code and CS68 resources are copied into
BiBladeFusion and maintained in its package namespace. Each imported component records
the source repository and commit in a machine-readable provenance module. Copied code
is adapted only where needed to use BiBladeFusion configuration and artifact contracts;
the HoloRobot behavior and coordinate conventions remain authoritative.

Only the CS68 and single-arm capabilities required by this project are imported. The
unrelated HoloRobot agent, application, multi-arm, laser, and simulation subsystems are
not copied. The bundled Elite description resources retain their upstream Apache-2.0
license and attribution.

## Migration increments

1. **Complete:** import the required HoloRobot code/resources and add fail-closed
   provenance checks.
2. **Complete:** align robot state and command conversion with the HoloRobot Elite path.
3. **Complete:** resolve copied CS68 URDF/STL/limits and cross-check FK fixtures.
4. **Partially complete:** HoloRobot Pinocchio/FCL self-collision is implemented;
   workcell environment collision still uses the legacy conservative capsule prefilter.
5. Convert planned camera poses into the copied motion contracts and persist preflight
   evidence and cost beside the BiBladeFusion view plan.
6. **Library layer complete:** guarded execution is behind default-off configuration,
   exact hash confirmation, expiring one-shot permits, and live revalidation. No motion
   CLI is exposed before step 5 and supervised hardware acceptance are complete.
7. Remove superseded MDH/capsule code only after artifact compatibility and regression
   tests cover existing offline workflows.

## Safety gates

- Motion remains disabled by default.
- Import or model-resolution failure is blocking, never a fallback to approximate motion.
- Missing hand-eye calibration, robot identity, collision model, current joint state, or
  preflight evidence is blocking.
- A BiBladeFusion view plan never directly calls the Elite SDK.
- Hardware execution must pass HoloRobot planning/preflight and an explicit operator
  approval gate in the same run.
- The first hardware acceptance is a separately marked test with a known safe pose and
  an operator present; unit and dry-run tests never command motion.
