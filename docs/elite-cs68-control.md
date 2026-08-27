# Elite CS68 control safety contract

BiBladeFusion contains a scoped copy/adaptation of the Elite CS68 control stack from
HoloRobot commit `93216a428cb8004382e9e39e5da7cd7bc6cbfffd`. It does not import HoloRobot
at runtime and does not modify the HoloRobot source tree.

## Implemented boundary

- `EliteArm` follows HoloRobot's Dashboard, RTSI, and EliteDriver lifecycle.
- The Elite boundary interprets TCP values as metres plus XYZ roll/pitch/yaw radians,
  matching the pinned HoloRobot implementation. Internal geometry remains `PoseSE3`.
- Point-to-point joint/TCP trajectories, `writeSpeedj`, zero-speed stopping, ServoJ
  priming, ServoJ hold, and fixed-rate ServoJ streaming are available as library APIs.
- The copied CS68 URDF, STL meshes, D435i wrist mount, joint limits, and Pinocchio/FCL
  collision pairs are package-owned resources with pinned provenance.
- Conservative linear-joint preflight samples collision geometry and generates a
  velocity-limited ServoJ sequence from the copied CS68 limits.

## Gates before any motion

All of the following are mandatory:

1. `robot.motion_enabled` must be explicitly set to `true`; its committed default is
   `false`.
2. A preflight must be clear and have an available collision checker. Clear preflight
   evidence still reports `motion_authorized: false`.
3. The operator must type the exact preflight-hash confirmation returned by
   `GuardedEliteExecutor.approval_prompt()` in the same process.
4. The resulting permit expires, is valid only once, and is bound to the exact start,
   goal, collision evidence, and ServoJ command stream.
5. Immediately before sending commands, the executor compares live joints with the
   preflight start and repeats Pinocchio/FCL path validation.

There is intentionally no motion CLI yet. Existing `bbf robot status`, acquisition,
planning, and path-validation commands remain non-moving. Hardware enablement will be
added only with persisted preflight artifacts, an interactive approval command, and a
separately supervised known-safe-pose acceptance procedure.

## Known limitations

- The current Pinocchio/FCL backend covers CS68 and D435i-mount self-collision. Existing
  capsule/workcell validation remains the environment-collision prefilter until the
  workcell is migrated to HoloRobot occupancy or primitive/hybrid geometry.
- Elite's copied public joint-limit profile has velocity limits but marks acceleration
  limits unavailable. Preflight therefore records `acceleration_limits_unavailable`.
- Unit tests use SDK doubles and never connect to hardware, power on a controller,
  release brakes, or transmit a motion command.
