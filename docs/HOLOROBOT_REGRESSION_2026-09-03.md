# HoloRobot reference regression — 2026-09-03

Status: offline implementation and regression complete; physical verification pending

Implementation commit: `47fc472 fix: align ES68 motion lifecycle with HoloRobot`

Reference repository: `/home/vale/Documents/HoloRobot`

Reference commit: `93216a428cb8004382e9e39e5da7cd7bc6cbfffd`

The reference worktree was clean during this review. BiBladeFusion's thermal feature
worktree was not modified or included.

## 1. Result

The latest failures were not caused by NBV scoring or by all candidates being
unreachable. The hardware run had already produced an approval-ready segment. The
failure was in the transition from a stopped capture state into one ServoJ transaction
and back to a stopped state.

Two control-lifecycle differences from HoloRobot directly explained the observed log:

1. BiBladeFusion started an external-control reverse session during guarded enable while
   its software stop latch was intentionally still set. Post-enable revalidation then
   left the session without ServoJ traffic long enough for it to close. Guarded resume
   opened it again. This matches the three reverse ports opening, closing, then opening
   and closing a second time in the reported run.
2. After a successful stream, BiBladeFusion immediately sent `writeIdle` and stopped the
   Dashboard task before checking that the physical joints had converged to the final
   ServoJ setpoint. HoloRobot repeatedly holds the final setpoint and reads the persistent
   robot feedback first.

The implementation now establishes the reverse session once, when the fresh approved
stop latch is released, and ports HoloRobot's endpoint-hold feedback loop before the
strict stop-and-capture boundary.

## 2. Module-by-module comparison

| Area | HoloRobot behavior | BiBladeFusion decision |
|---|---|---|
| Elite object lifetime | One arm owns Dashboard, RTSI, and EliteDriver connections | Retain one `EliteArm`; its persistent RTSI remains the stationarity authority |
| Read-only bootstrap | Normal HoloRobot enable commonly creates the motion driver immediately | Keep BiBladeFusion's read-only initial connection so an unapproved bootstrap cannot start motion infrastructure |
| Enable/recovery | Clear the local stopped state and establish reverse control as one transition | Guarded enable now performs only driver-object creation, power, brake release, and speed cap; guarded resume atomically releases the approved latch and establishes reverse control once |
| ServoJ endpoint | Repeat the final approved setpoint, sample actual joints, require three consecutive samples within `0.005 rad`, and time out after `2 s` | Port the same `2 s / 0.005 rad / 20 ms / 3 samples` policy behind the existing guarded-motion capability |
| Segment stop | HoloRobot's generic arm stop uses `writeIdle` | Retain BiBladeFusion's additional Dashboard task stop because captures are required to have a real stop-and-capture boundary; endpoint convergence now happens before that stop |
| Robot feedback | Read actual joints from the arm's persistent feedback connection | Reuse the persistent Elite RTSI connection for streaming, endpoint settling, and post-stop evidence; the separate perception sampler is not used as the endpoint authority |
| IK | Pure MDH forward kinematics, finite-difference world Jacobian, damped least squares, joint-limit clamp, and multiple seeds | Port this solver as the default candidate reachability path. Keep vendor KDL only as an explicitly injected historical/reproduction path |
| Collision | Robot-model geometry remains separate from environment representations | Retain BiBladeFusion's ES68+D435i URDF/STL Pinocchio/HPP-FCL model and robust swept proof; do not return to link circumspheres |
| Occupancy | HoloRobot provides an incremental sparse occupancy backend suitable for its direct scene-depth workflows | Deliberately retain BiBladeFusion's full-scene, three-state, multi-view FoundationStereo occupancy and CUDA DDA. Its UNKNOWN/static-free evidence semantics are blade-experiment requirements absent from the generic HoloRobot mapper |
| Active view science | Generic frontier/raycast utilities exist | Retain BiBladeFusion's blade-side, fin-face, projected-support, and evolving-proxy gain because this is the paper-specific contribution |

## 3. Code changes

### 3.1 One reverse-control session per approved segment

`EliteArm._guarded_enable_for_servoj_control` no longer launches the external-control
script while the approved stop latch is held. `EliteArm._guarded_resume_servoj_control`
now:

1. validates the capability, safety state, enabled state, and exact stop generation;
2. atomically clears that approved stop latch;
3. establishes the reverse session once;
4. rechecks that no concurrent stop changed the generation;
5. relatches locally on recovery failure without overwriting a newer concurrent stop.

No ServoJ command is sent by this transition. The existing current-position preparation
write remains the first ServoJ write.

### 3.2 Endpoint convergence before controller stop

After the exact approved stream succeeds, the same final command is held while the
persistent RTSI joint feedback is sampled. Success requires three consecutive samples at
or below `0.005 rad` maximum joint error. The result records duration, sample count,
maximum error, and final error. A timeout stops the segment and reports the numerical
error rather than allowing a capture from an unconfirmed endpoint.

Only after this succeeds does the existing `writeIdle` plus Dashboard stop request run.
The coordinator then proves the ordinary stopped/stationary sample window.

### 3.3 HoloRobot MDH IK replaces normal KDL candidate probing

Normal `EliteCs68IkChecker` construction no longer imports or invokes the vendor KDL
plugin. It uses the controller-specific calibrated MDH chain, HoloRobot's damped
least-squares update, ES68 joint limits, the live joint seed, and HoloRobot's three preset
seeds. Consequently an unreachable candidate becomes one bounded result rather than a
burst of SDK warnings.

The injected KDL interface remains for historical artifact reproduction. Final endpoint
FK, collision, and continuous path checks remain separate gates; an IK success alone does
not authorize motion.

### 3.4 Actionable stationarity timeout

Every ordinary stationarity timeout now includes the last accepted feedback sample:

```text
robot_mode, runtime_state, controller_stopped, goal_error, stable_samples,
actual/target joint speed, actual/target TCP linear/angular speed
```

This distinguishes an asynchronous Dashboard stop, endpoint error, residual velocity,
and missing settled-window coverage in one run. The old phrase “before the next
robot-state sample” did not prove an RTSI disconnection; it could also mean the settle
deadline elapsed between polling instants.

## 4. Timing findings

The locally synchronized `blade-placement-20260901-01-attempt-11` timing artifacts show:

```text
manual hard-ROI wait / foreground preflight: 226.127 s exclusive
CPU depth-ray integration:                220.510 s exclusive over 7 calls
FoundationStereo backend:                   3.344 s
candidate reachability checks:              0.016 s for 12 calls
```

Thus the historical multi-minute first cycle was not primarily FoundationStereo or IK.
Most time was operator annotation plus CPU ray integration and artifact handling. The
current eiai route already selects deterministic CUDA DDA, which addresses the dominant
compute term. The next run must preserve its new timing artifacts before any further
optimization is chosen; no unmeasured timeout increase is part of this fix.

## 5. Offline verification

The following scopes were exercised using the main source tree and the available local
test environment:

```text
planning/guarded execution/Elite arm: 107 passed
stationarity and planning/workflow integration: 230 passed
Elite control, guarded execution, stationarity: 143 passed
motion-envelope commissioning module: 11 passed
```

The full-suite result is recorded in `docs/CURRENT_STATE.md`. No physical command was
issued during this regression.

## 6. Single physical acceptance criterion

The next eiai run is one validation attempt, not another discovery loop. After the exact
approval token:

- one group of reverse ports should connect; there must be no pre-motion
  connect/close/reconnect cycle;
- the arm should execute exactly the displayed single segment;
- endpoint settling should finish before `writeIdle`/Dashboard stop;
- the controller should reach the stopped stationary window and advance to the next
  capture;
- if it fails, the terminal line must contain the last controller state and numerical
  goal/velocity context.

Passing offline tests does not claim that these physical postconditions have already
been observed.
