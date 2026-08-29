# HoloRobot robot-stack migration

Status: software safety and supervised coordination implemented; hardware acceptance
and real-system validation pending

## Decision

BiBladeFusion owns the thin-blade reconstruction task: bilateral proxy construction,
front/back partitioning, coverage evidence, view generation, view scoring, depth
comparison, and future thermal fusion.

HoloRobot is the behavioral source of truth for the ES68 robot stack:

- Elite ES68 hardware lifecycle and motion commands;
- robot-domain joint and pose contracts;
- ES68 calibrated kinematics, joint limits, generated URDF, accepted STL manifest, and
  collision geometry;
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
motion implementation after this migration begins. Its occupancy layer extends the
HoloRobot-style self/workcell checks for an unknown environment; it does not replace the
robot model, trajectory contract, or guarded execution boundary.

## Coordinate conventions

The HoloRobot conventions are authoritative for this integration, including its current
Elite TCP pose interpretation. Internal orientations use normalized quaternions in
`(x, y, z, w)` order. Conversion at the Elite backend boundary follows HoloRobot rather
than independently interpreting the SDK documentation.

Canonical frames for the initial single-arm cell are:

- `elite_b_base`: ES68 base/world frame;
- `elite_b_flange`: physical flange frame;
- `elite_b_tcp`: configured controller TCP frame;
- `elite_b_wrist_d435i_optical`: D435i optical frame.

Calibration artifacts remain installation-specific. A HoloRobot artifact may be reused
only when its robot identity, camera identity, mounting state, coordinate frames, and
acceptance status all match the current installation.

## Source and dependency policy

The required HoloRobot robot-stack code and structural resources are copied into
BiBladeFusion and maintained in its package namespace. Each imported component records
the source repository and commit in a machine-readable provenance module. Copied code
is adapted only where needed to use BiBladeFusion configuration and artifact contracts;
the HoloRobot behavior and coordinate conventions remain authoritative.

Some upstream resource paths and Python symbols retain `cs68` in their names because the
pinned HoloRobot structural skeleton originated there. That is provenance, not the
identity of the physical robot: the production system is ES68. Its generated URDF uses
the calibrated ES68 chain, and production collision resolution requires the explicit,
ready final ES68+D435i STL manifest. Missing or unready final geometry is blocking; the
upstream-labelled meshes are not an automatic production fallback.

Only the Elite single-arm capabilities required by this project are imported. The
unrelated HoloRobot agent, application, multi-arm, laser, and simulation subsystems are
not copied. The bundled Elite description resources retain their upstream Apache-2.0
license and attribution.

## Migration increments

1. **Complete:** import the required HoloRobot code/resources and add fail-closed
   provenance checks.
2. **Complete:** align robot state and command conversion with the HoloRobot Elite path.
3. **Complete at the software-contract level:** materialize the calibrated ES68 chain,
   bind joint offsets/limits and cross-check FK fixtures. Final hardware acceptance of
   the generated URDF axes and ES68+D435i mesh attachments remains mandatory.
4. **Implemented offline:** Pinocchio/FCL checks ES68+D435i self-collision and configured
   clearance-expanded AABB workcell obstacles. A separate FoundationStereo safety map
   represents the unknown blade/environment as `FREE`, `OCCUPIED`, or `UNKNOWN`; unknown
   and out-of-grid queries block.
5. **Software proof backends complete:** an adaptive interval proof uses FCL midpoint
   separation and conservative serial-chain displacement bounds for continuous
   ES68+D435i mesh clearance. A separate interval enclosure proves the complete robot
   sweep against the three-state voxel map. Either backend returns `UNKNOWN`, rather
   than passing, when its subdivision or numerical proof limit is reached. Both proofs
   bind the exact joint path, geometry, policy and map evidence.
6. **Guarded execution complete at software-contract level:** execution remains behind
   default-off configuration, exact per-segment hash confirmation, expiring one-shot
   permits, live-start revalidation, a frozen occupancy generation and asynchronous stop
   generations. Software completion is not physical clearance.
7. **Read-only replay and live observation complete:** immutable supervisory snapshots visualize the
   verified data actually present in their source chain. Exact robot/camera collision
   meshes appear only when the active final model reproduces the mapping geometry hash;
   planned TCP paths appear only from a canonically derived preflight. Stopped perception
   samples provide the recorded actual path; they are explicitly not described as
   high-rate ServoJ tracking. The GUI cannot approve or transmit motion. `--follow` is
   atomic filesystem observation, not an avoidance controller.
8. **Supervised runtime implemented, default off:** the stop-and-capture composition binds
   settled capture, FoundationStereo, fresh occupancy, unknown-blade coarse science,
   schema-5 handoff, fixed-reference fine science, next-view planning and one guarded
   short segment at a time. The first map views remain operator-guided because unknown
   space blocks motion. Every later segment still needs its own exact approval. No part
   of this status claims that the physical ES68/D435i/workcell has passed acceptance.
9. Remove superseded MDH/capsule code only after artifact compatibility and regression
   tests cover existing offline workflows.

## Safety gates

- Motion remains disabled by default.
- Import or model-resolution failure is blocking, never a fallback to approximate motion.
- Missing hand-eye calibration, robot identity, final ES68+D435i collision manifest,
  current joint state, fresh `MAP_READY` occupancy, or preflight evidence is blocking.
- FoundationStereo mapping retains surfaces clearly in front of rendered robot geometry,
  but masks matching or farther measurements and leaves those rays `UNKNOWN`; it never
  clears through the robot or its occluded background.
- Continuous mesh and occupancy proofs are conservative interval certificates, not dense
  sampling. Failure to establish a strictly positive bound is blocking. Self-masked
  robot voxels remain `UNKNOWN`; they can be exempted only inside an immutable,
  operator-recorded static-free AABB acceptance whose robot geometry, workspace and exact
  regions match the runtime. `OCCUPIED` always blocks, including inside such a region.
- An accepted static-free region must remain free of every external object throughout the
  experiment and must not overlap the possible blade, fixture or support envelope. The
  acceptance asset itself never authorizes a segment.
- A BiBladeFusion view plan never directly calls the Elite SDK.
- Hardware execution must pass HoloRobot planning/preflight and an explicit operator
  approval gate in the same run.
- Offline occupancy replay and supervisory snapshots never satisfy that gate, and the
  supervisory GUI contains no authorization path.
- The first hardware acceptance is a separately marked test with a known safe pose and
  an operator present; unit and dry-run tests never command motion.
