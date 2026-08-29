# HoloRobot robot-stack migration

Status: offline safety layers implemented; continuous swept-motion proofs, live
coordination, and hardware acceptance pending

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
5. **Artifact contract implemented; production path blocked:** ordered endpoint-feasible
   view sequences produce immutable, source-bound, re-derived tool goals and discrete
   mesh/occupancy diagnostics. Production mode stops at the missing continuous swept-mesh
   proof before generating a ServoJ trajectory; continuous robot-versus-voxel evidence is
   a separate unresolved requirement. Library-only diagnostic overrides are not approval
   evidence.
6. **Library layer complete:** guarded execution is behind default-off configuration,
   exact hash confirmation, expiring one-shot permits, and live revalidation. No motion
   CLI is exposed before supervised hardware acceptance is complete.
7. **Offline replay skeleton complete:** immutable supervisory snapshots visualize the
   verified data actually present in their source chain. Exact robot/camera collision
   meshes appear only when the active final model reproduces the mapping geometry hash;
   planned TCP endpoints appear only from a canonically re-derived preflight, and no
   actual continuous TCP trace is currently persisted. The GUI is read-only, remains
   `REPLAY/BLOCKED`, and cannot approve or transmit motion. `--follow` is only filesystem
   polling, not online obstacle avoidance.
8. **Implemented at library level; production integration pending:** the stop-and-capture
   coordinator now binds settled capture, FoundationStereo inference, fresh occupancy,
   optional fine-science assets, planning invalidation and guarded short-segment state.
   It is disabled by default, has no public composition-root or motion CLI, and has not
   been hardware-verified. `bbf occupancy build-replay` still deliberately seals every
   result `STALE`, so it can never supply live motion evidence.
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
- The present occupancy checker uses transformed local-AABB bounding spheres at discrete
  joint samples. The resulting self-`UNKNOWN` intersection, missing continuous
  mesh/FCL swept-volume proof, and independently missing continuous robot-versus-voxel
  swept-volume proof are physical motion-release blockers, even if other checks pass.
- A BiBladeFusion view plan never directly calls the Elite SDK.
- Hardware execution must pass HoloRobot planning/preflight and an explicit operator
  approval gate in the same run.
- Offline occupancy replay and supervisory snapshots never satisfy that gate, and the
  supervisory GUI contains no authorization path.
- The first hardware acceptance is a separately marked test with a known safe pose and
  an operator present; unit and dry-run tests never command motion.
