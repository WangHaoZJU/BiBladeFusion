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
- Conservative linear-joint preflight proves the full mesh/workcell interval using FCL
  separation and serial-chain displacement bounds, and independently proves the full
  robot-versus-voxel interval with displacement-expanded geometry spheres. Both
  certificates are bound to the exact segment; inconclusive subdivision, numerical,
  evidence or freshness states return `UNKNOWN` and block.
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

`bbf safety preflight-path` requires `--occupancy`, persists and re-verifies the
mesh/occupancy contracts for an ordered view sequence, and never connects to the robot.
Robot-status, ordinary acquisition, planning, validation, occupancy replay, supervision
and standalone preflight commands remain non-moving.

The only public physical-motion composition is the default-off, interactive
`bbf scan run-unknown` runtime. It first requires `scan doctor --mode unknown`, opens one
ES68 and one D435i, and accepts explicit operator-positioned `c` captures until a live
`MAP_READY` generation exists. It then prepares one bounded segment at a time. The
operator must paste the exact current preflight token; the one-shot permit is consumed
before the private capability may perform any power/brake preparation. That preparation
does not clear the stop latch or transmit a trajectory. Joint state, occupancy identity,
freshness, both continuous proofs, stop generation and permit expiry are checked again
before the compare-and-clear resume and ServoJ transport. Every successful segment ends
with an acknowledged stop and one capture.

`bbf occupancy build-replay` is an offline evidence-reconstruction command. Every map it
writes is deliberately `STALE`, so supplying it to preflight remains blocked. Likewise,
`bbf supervise build-replay` and `bbf supervise replay` only build or display immutable
`REPLAY/BLOCKED` snapshots. The GUI is read-only: it has no permit, approval, stop, or
motion-transmission interface. Its `--follow` mode only observes newly published files
and is not a real-time control or dynamic-avoidance loop.

The coordinator atomically binds settled capture, FoundationStereo inference,
self-masking, fresh-map publication, coarse/fine scientific staging, plan invalidation,
preflight, one-shot approval and guarded execution. This closes a software contract, not
hardware acceptance: the committed configuration is off, and the complete path is not
yet verified on the final ES68/D435i/blade workcell. Hardware release still requires
final dimensions/attachments, measured timing and clearance thresholds, static-free
workcell declarations, known-collision fixtures, controller stop tests and controlled
operator acceptance.

## Known limitations

- Both interval proofs are conservative and can reject safe motion. The occupancy proof
  approximates each collision mesh with a sphere enclosing its transformed local AABB,
  then further expands it by the interval motion bound. It must not be relaxed to point
  sampling merely to obtain a feasible route.
- Because robot pixels are correctly removed without ray-clearing, the robot's own
  volume remains `UNKNOWN`. Only whole UNKNOWN voxels inside an immutable, physically
  accepted static-free AABB may use the narrow external-object-free exception;
  `OCCUPIED` always blocks. Such an AABB may not overlap the blade, fixture, support or
  any other external-object envelope.
- The map is stop-and-capture evidence for a static segment, not certified continuous
  dynamic-obstacle avoidance. Map changes invalidate existing preflights and permits.
- Elite's copied public joint-limit profile has velocity limits but marks acceleration
  limits unavailable. Preflight therefore records `acceleration_limits_unavailable`.
- Unit tests use SDK doubles and never connect to hardware, power on a controller,
  release brakes, or transmit a motion command. Continuous-proof regression tests are
  software geometry tests, not physical ES68 safety certification.
