# Elite ES68 control safety contract

BiBladeFusion contains a scoped copy/adaptation of the Elite control stack from
HoloRobot commit `93216a428cb8004382e9e39e5da7cd7bc6cbfffd`. It does not import HoloRobot
at runtime and does not modify the HoloRobot source tree. The physical robot in this
project is ES68. Some copied source paths and internal symbols retain `cs68` solely to
preserve upstream provenance; they must not be interpreted as a CS68 production model.

## Implemented boundary

- `EliteArm` follows HoloRobot's Dashboard, RTSI, and EliteDriver lifecycle.
- The Elite boundary interprets TCP values as metres plus XYZ roll/pitch/yaw radians,
  matching the pinned HoloRobot implementation. Internal geometry remains `PoseSE3`.
- Point-to-point joint/TCP trajectories, `writeSpeedj`, zero-speed stopping, ServoJ
  priming, ServoJ hold, and fixed-rate ServoJ streaming are available as library APIs.
- The calibrated ES68 chain, joint offsets/limits, generated URDF, final D435i wrist
  mount, and Pinocchio/FCL collision pairs form one hash-bound motion-model contract.
  Production model resolution requires the explicit, ready final ES68+D435i STL
  manifest and fails closed when it is absent or inconsistent.
- Conservative linear-joint preflight evaluates bounded-step mesh and depth-derived
  occupancy samples. In production mode it currently blocks at the missing continuous
  swept-mesh proof before ServoJ generation; an independent continuous
  robot-versus-voxel proof is also mandatory. A library-only diagnostic override may
  generate a stream for inspection, but it is never approval-ready.
- Unknown-blade occupancy is built from accepted FoundationStereo observations in
  `base`. A rendered ES68+D435i depth image retains a measured surface clearly in front
  of the robot, but masks matching or farther measurements so a dropout cannot clear
  through known robot geometry; those rays remain `UNKNOWN`. The voxel states are
  `FREE`, `OCCUPIED`, and `UNKNOWN`, and unknown/out-of-grid space blocks.

## Gates before any motion

All of the following are mandatory:

1. `robot.motion_enabled` must be explicitly set to `true`; its committed default is
   `false`.
2. A preflight must be clear and have both the final-model mesh checker and an occupancy
   checker backed by a fresh `MAP_READY` snapshot. Clear preflight evidence still
   reports `motion_authorized: false`.
3. The operator must type the exact preflight-hash confirmation returned by
   `GuardedEliteExecutor.approval_prompt()` in the same process.
4. The resulting permit expires, is valid only once, and is bound to the exact start,
   goal, ES68+D435i model contract, occupancy sequence/hash/freshness horizon, and full
   ServoJ runtime/command stream.
5. Immediately before sending commands, the executor compares live joints with the
   preflight start, confirms the occupancy identity and remaining freshness, and repeats
   both Pinocchio/FCL and occupancy path validation.

There is intentionally no motion CLI yet. `bbf safety preflight-path` requires
`--occupancy`, persists and re-verifies the mesh/occupancy contracts and available
diagnostic evidence for an ordered view sequence, and does not connect to the robot. If a
valid fresh map is supplied by a lower-level integration, current production motion
evaluation stops at the missing continuous swept-mesh proof before a ServoJ stream exists.
Existing robot-status, acquisition, planning, validation, occupancy, supervision, and
preflight commands remain non-moving.

No current CLI publishes a fresh `MAP_READY` occupancy asset. The offline
`build-replay` path below is deliberately not a substitute for the missing native
coordinator.

`bbf occupancy build-replay` is an offline evidence-reconstruction command. Every map it
writes is deliberately `STALE`, so supplying it to preflight remains blocked. Likewise,
`bbf supervise build-replay` and `bbf supervise replay` only build or display immutable
`REPLAY/BLOCKED` snapshots. The GUI is read-only: it has no permit, approval, stop, or
motion-transmission interface. Its `--follow` mode only observes newly published files
and is not a real-time control or dynamic-avoidance loop.

Hardware enablement can be considered only after a native coordinator atomically binds
settled capture, FoundationStereo inference, self-masking, fresh map publication, plan
invalidation, preflight, and execution; that coordinator has not yet been implemented or
verified on the ES68/D435i system.

## Known limitations

- Mesh/FCL and robot-versus-voxel occupancy checks currently sample the joint path at
  bounded discrete states; the latter approximates each collision mesh by the bounding
  sphere of its transformed local AABB. Neither supplies its required continuous swept-
  volume proof. These are independent release blockers, not interchangeable checks.
- Because robot pixels are correctly removed without ray-clearing, the robot's own
  volume remains `UNKNOWN`; the current bounding-sphere query can therefore block the
  capture pose itself. A conservative, model-hash-bound self-volume treatment with an
  observed-free shell (or stronger equivalent) is required before physical release.
- The map is stop-and-capture evidence for a static segment, not certified continuous
  dynamic-obstacle avoidance. Map changes invalidate existing preflights and permits.
- Elite's copied public joint-limit profile has velocity limits but marks acceleration
  limits unavailable. Preflight therefore records `acceleration_limits_unavailable`.
- Unit tests use SDK doubles and never connect to hardware, power on a controller,
  release brakes, or transmit a motion command.
