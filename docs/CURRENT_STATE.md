# BiBladeFusion current state

Checkpoint date: 2026-09-04

Authoritative branch for this work: `main`

Local main worktree: `/home/vale/Documents/Proj1/biblade-fusion-main`

eiai deployment checkout: `/home/eiai/Documents/wh/BiBladeFusion`

This file is an operational checkpoint, not a claim that the unresolved hardware flow is
working. Update it after every material hardware result or fix.

## 1. Repository checkpoint

The commit containing this checkpoint is the deployment's minimum starting point; use
`git log -1 --oneline` after pulling `origin/main` to record its immutable identifier.
The preceding HoloRobot-aligned control/IK regression is documented in
`docs/HOLOROBOT_REGRESSION_2026-09-03.md`.

Relevant preceding fixes are:

```text
8e88624 fix: reuse bound path proof during execution
3240475 fix: bound guarded path revalidation time
621ac87 fix: bind bootstrap readiness after phase transition
a0a69ca fix: allow supervised bootstrap mapping motion
274e40c fix: accept X-AnyLabeling bootstrap polygons
2ba9542 fix: construct strict ES68 collision checker
eede1ec fix: tighten swept mesh collision proof
```

The separate `/home/vale/Documents/Proj1/biblade-fusion` worktree is on
`feat-thermal-usb-hcusb` and contains thermal-camera work. Do not copy its dirty changes
into `main` while addressing planning/runtime failures.

## 2. eiai environment and accepted configuration

The following values were reported by the operator and successfully parsed on eiai. They
are deployment state, not committed `configs/default.yaml` values.

### 2.1 Workcell and proxy

```yaml
view_filter:
  workspace:
    minimum_m: [-1.00, -0.55, 0.00]
    maximum_m: [1.00, 0.55, 1.10]

occupancy:
  workspace_bounds_min_m: [-1.00, -0.55, 0.00]
  workspace_bounds_max_m: [1.00, 0.55, 1.10]
  maximum_map_age_s: null
  unknown_policy: block
  minimum_source_views: 3
  maximum_source_views: 3
  minimum_free_observations: 3

proxy_model:
  voxel_size_m: 0.002
  minimum_points: 100
  estimated_planar_extents_m: [0.30, 0.13]
  estimated_thickness_m: 0.006
  blade_envelope_min_m: [0.454, 0.001, 0.031]
  blade_envelope_max_m: [0.639, 0.164, 0.368]
  minimum_envelope_retained_fraction: 0.95
```

The blade envelope above is the last reported configured value. It must be treated as a
science envelope and revalidated if later data shows blade/fin points being clipped.

### 2.2 Accepted static-free evidence

```text
path: data/acceptance/es68_d435i_static_free_002
id:   b87a40387b6bcbb9e802c6a984edc8a883766d9c64a433ebe908d274b7965bae
```

The target exclusion column used by that accepted declaration is approximately:

```text
x: [0.45, 0.64] m
y: [-0.10, 0.17] m
z: [0.00, 0.37] m
```

Five accepted-static-free AABBs cover the workspace before/after the target in `x`, on
the negative/positive `y` sides of the target column, and above the target envelope.
They do not override OCCUPIED voxels.

### 2.3 Current motion envelope

```text
path: data/acceptance/es68_d435i_motion_envelope_004
id:   29624b08242d2c8ef7544cb958bf2a64335f895b719b421f792ff8c750719f9b
metadata SHA-256: ce807010bc11bdf50bcfb804214f85705ad155bf3c7af16d3a214a2354722766
```

The operator recorded `_004` after D036-D038 physically passed forward, reverse and
intentional tracking-stop trials for the current `8 ms / 0.03 s` control contract.
`configs/local.yaml` binds this path and ID. `_003` and earlier assets remain historical
evidence and must not be rebound as current.

### 2.4 Runtime configuration facts last confirmed

```text
adaptive IK search: enabled
maximum ranked online preflight candidates: 3
legacy maximum segment joint delta: parsed but ignored
single-view bootstrap motion: enabled
projected ROI: dilation 12 px, minimum 100 reference points,
               minimum 500 reference pixels, minimum match fraction 0.50
occupancy ray integration: deterministic CUDA DDA
```

The online contract now treats the declared workspace as a hard outer boundary, searches
at most 32 IK poses / 1.5 s per candidate family, collision-checks every distinct IK
branch before choosing an endpoint, and preflights the selector's complete bounded
science-ranked queue before moving to one selected viewpoint.
`motion_preflight.maximum_joint_step_rad` is only an internal HoloRobot interpolation and
sampled-collision interval.
ServoJ timing uses both the packaged joint velocity vector and the configured ES68
acceleration vector `[4, 4, 4, 4, 4, 4] rad/s^2`.

Run commands on eiai must use `/usr/bin/env -u PYTHONPATH` because ROS Humble's Python
3.10 Pinocchio/HPP-FCL packages otherwise shadow the Python 3.12 virtual environment.
Do not run a plain `uv sync --frozen` as a routine pull step: it previously removed the
GPU/FoundationStereo and private Elite SDK packages from the environment.

## 3. Current physical placement and reusable evidence

The operator reported that the blade/fixture placement has not moved, so the active
placement remains:

```text
blade-placement-20260901-01
```

Each software retry must still use a new `run_id` and output directory.

A previously accepted first-view polygon was read successfully by
`_read_hard_roi_seed(...)` as:

```text
mode: hard_roi
kind: polygon
vertices: 11
```

The known source is under the `planning-test-20260903-04` first-view annotation tree on
eiai. Before reusing it, pass the JSON file path—not the JSON file contents—to
`--bootstrap-polygon`, and verify that the camera, robot, blade, and fixture have not
moved relative to that captured first view.

An offline continuous collision replay at the stopped pose from
`planning-test-20260903-03` reported:

```text
status: clear
samples: 7
termination: all_intervals_certified
deepest subdivision: 2
minimum margin: 0.000968118750610798 m
blocking reasons: ()
```

This is useful diagnostic evidence, not proof that the later driver/recovery sequence is
correct.

## 4. Latest end-to-end progress

The runtime has demonstrated the following transitions on real hardware:

```text
operator_bootstrap
  -> first formal frame and hard ROI
  -> coarse candidate generation
  -> bootstrap_motion_ready
  -> waiting_approval
```

The operator received and entered an exact approval token. Thus first-view perception,
adaptive candidate production, at least one approved preflight, and the supervisory
approval path were reached.

Earlier blockers that have already received code changes include:

- giant link-sphere false collision behavior;
- missing fully hash-bound collision checker;
- bootstrap `MAPPING` state rejected by live supervision;
- X-AnyLabeling polygon schema handling;
- expensive duplicated mesh/occupancy path proofs.

Do not assume they are physically closed merely because a unit test or commit exists;
regressions must be distinguished from the latest blocker.

## 5. Historical hardware failures and implemented corrections

The 2026-09-04 09:56–09:59 physical attempt
`planning-test-20260904-095629` captured and reconstructed a new first view, then stopped
before motion because the planner required a back-side opposing fin pair but had searched
only symmetric fin-axis azimuths. The message's `tested 4 endpoints` counted four semantic
families rather than their internal attempts, and `1 seed` meant one composite checker;
both descriptions hid the actual search trace.

The preceding physical attempt (2026-09-03 22:25–22:27 local time) reached exact approval but
then showed two reverse-port sessions and ended in a stationarity timeout. Earlier attempts
expired an already approved permit during guarded enable/recovery and rejected a sampled
`RUNNING/PLAYING` state despite the motion command having been idled. Those observations
are historical; none has been re-run on hardware since this wider correction set.

The 2026-09-04 regression corrected the common causes rather than adding more delay:

1. a permit expires only before exact consumption; power/brake recovery cannot invalidate
   an already consumed permit;
2. reverse control starts once at guarded resume, and unchanged path evidence is not
   recomputed a second time after recovery;
3. final ServoJ feedback converges before stop; normal stop matches HoloRobot's
   `writeIdle(0)` and does not call Dashboard `stopProgram`;
4. post-motion stationarity requires an unchanged stop latch plus sampled joint/TCP pose;
   `runtime_state=PLAYING` and instantaneous velocity noise are not treated as motion by
   themselves. Bootstrap remains strict Dashboard-STOPPED;
5. online IK reuses the already loaded Pinocchio/URDF collision model and HoloRobot's
   neighboring-seed sweep; analytic MDH remains the offline fallback and KDL is not the
   normal path;
6. one NBV is one complete viewpoint path and one capture; the legacy 0.02 rad setting is
   HoloRobot-style joint interpolation, not an intermediate reconstruction trigger;
7. each online cycle checks the selector's complete already-bounded science-ranked queue;
   the legacy `maximum_ranked_preflight_candidates` value is parse-only and cannot discard
   a safe fourth or later candidate;
8. incremental occupancy reuses an unchanged verified prefix and the live writer does not
   immediately replay identical rays. Sliding/replacement windows and cross-process reads
   remain strict full rebuild/replay;
9. automatic ROI transfer uses the projected proxy's per-pixel depth band plus the blade
   envelope, and fine NBV receives bounded adaptive distance/incidence fallback;
10. measurement completion is based on acquired coverage; downstream mesh/watertight QA
    is nonblocking by default and remains available as a strict opt-in;
11. synchronized vale/eiai assets with stale absolute roots relocate only after exact
    content-hash verification.
12. opposing fin candidates retain the required sign but may share a tangential bias;
    incidence is information-ranked, wrist rolls are interleaved, and a symmetric camera
    pair is no longer required.
13. online path validation no longer runs the recursive six-dimensional continuous
    interval certificate. It reuses HoloRobot's fixed-step, fail-fast collision contract.
    HoloRobot's deployed ES68 resolution is `0.1 rad / (5 - 1) = 0.025 rad`; the already
    interpolated BiBladeFusion path is checked at its finer `0.02 rad` waypoints rather
    than being accidentally subdivided by five again. Exact URDF/STL robot geometry,
    UNKNOWN-as-blocked occupancy, map/hash binding, approval, ServoJ tracking stop, and
    endpoint settling remain active. Adjacent same-state occupancy voxels are represented
    as exact X-axis run boxes to remove per-voxel FCL calls without using link spheres.
14. every distinct solution returned by HoloRobot's bounded Pinocchio seed sweep is now
    checked at the endpoint with the same hash-bound URDF/STL model. The nearest clear
    branch is retained; a nearer colliding branch cannot hide a farther clear branch.
15. motion planning now follows HoloRobot's composite order: validate endpoints, try the
    straight conservative route, and invoke one bounded RRTConnect search only for a true
    interior `PATH_BLOCKED` result. UNKNOWN evidence and endpoint collisions fail fast and
    never consume an OMPL timeout. Any detour is resampled and completely rechecked before
    a permit can be prepared.
16. the concrete Pinocchio seed sweep no longer stops at its first converged branch; all
    distinct bounded branches reach the endpoint collision gate.
17. coarse ranked camera poses are re-solved from the latest stopped joint state before
    path preflight, so a first-view IK endpoint is not reused after robot motion. Fine NBV
    also receives the shared runtime Pinocchio/URDF checker rather than a separate MDH
    fallback.
18. ServoJ timing now applies HoloRobot's joint velocity and acceleration duration bounds;
    production stationarity sampling shares the persistent EliteArm RTSI connection and
    ends immediately after exposure instead of spanning inference, mapping, and writes.

The previous KDL `-5` flood and the multi-minute source replay are therefore not expected
on the new normal path. This is an offline conclusion, not a claim that the physical arm
has already demonstrated it.

## 6. Measured offline performance and remaining boundary

Historical first-cycle evidence showed `220.510 s` of CPU depth-ray integration over
seven integration calls, versus `3.344 s` for FoundationStereo. The new live initial
three-view path requires one integration per new source (three total), rather than
rebuild/write/read replay. eiai's selected CUDA DDA handles those three calls. Exact GPU
wall time must be read from the next eiai timing asset; it is not inferred here.

The synchronized real placement's coarse view planning completed offline with:

```text
candidates: 2
geometry feasible: 2
IK endpoint feasible: 2
wall time: 0.78 s
```

It selected independently feasible front/back adaptive views. Online preflight walks the
selector's complete bounded science-ranked queue. For each endpoint it first checks the
straight joint-space route with HoloRobot's sampled-segment contract. When—and only
when—both endpoints are clear and the route is blocked in its interior, one RRTConnect
solve receives a 1.0 s default budget. A returned detour is resampled at no more than
0.02 rad and rechecked in full against exact robot geometry and one bound occupancy map.
The former recursive interval proof remains available for offline acceptance/diagnostics
but is no longer called by the active NBV loop.

One offline replay of an old occupancy artifact took about 64 s and correctly failed
because the artifact was rendered with Open3D while the current local environment selected
the NumPy renderer. This is a cross-process historical replay, not the new live cached
path. Renderer identity remains a deliberate hard evidence boundary.

The 2026-09-04 fin-discovery regression was separately reproduced with the synchronized
attempt-09 real proxy. The previous symmetric policy produced zero complete back pairs.
The signed common-bias policy plus production Pinocchio checker produced one complete back
pair in about 1.17 s; all IK, workspace and later motion gates remain unchanged.

A local occupancy regression probe used the operator-reported first-view joint vector,
the accepted 2.0 x 1.1 x 1.1 m workspace partition, a conservative UNKNOWN map, and the
production ES68+D435i STL model. One pose classified 179,154 broad-phase voxels in 0.0143 s
with no dangerous run requiring an FCL distance call. This demonstrates removal of the
per-voxel Python/FCL loop for accepted-free space; it is not a substitute for timing the
next real eiai occupancy map and complete candidate path.

The 2026-09-04 11:25 physical attempt reached `bootstrap_motion_ready`, remained inside
planning/preflight, and terminated after about five minutes with the outer runtime hiding
the concrete inner blocker as `active supervised runner entered a blocked state`. Code
inspection found two deterministic multipliers in the online occupancy path: every sampled
pose recomputed the complete immutable occupancy SHA-256 before and after its query, and a
path already interpolated at `0.02 rad` was subdivided into five samples per small segment.
The current correction binds/hashes one immutable snapshot once at path entry and once at
path exit, checks all poses directly against that bound object, uses the `0.02 rad`
waypoints without redundant subdivision, and preserves the inner candidate/preflight
reason in the terminal.

The exact 11:25 occupancy snapshot (`200 x 110 x 110`, 154,867 accumulated FREE votes,
1,694 occupied voxels) and the three attempted joint endpoints were replayed locally with
the production ES68+D435i STL model. The corrected preflights took `1.84 s`, `1.67 s`, and
`3.19 s` instead of about `10.27 s`, `9.45 s`, and `86.29 s`. More importantly, the event
record showed seven ranked candidates while the legacy configured limit stopped after
three. Replaying the four untried front-fin paths found two clear paths in `2.05 s` and
`4.85 s`. The legacy `maximum_ranked_preflight_candidates: 3` field is therefore parse-only;
the already-bounded selector queue is no longer truncated. This correction is not yet
physically verified.

The 2026-09-04 15:44 physical attempt
`planning-test-20260904-154403` ran main commit `c9f141e`. No robot motion occurred. It
reused the unchanged first-view ROI, completed perception, and reached the HoloRobot
single-arm planning path. Exact persisted timing evidence separates the 38.41-second
operator wait as follows:

```text
next-view selection and current-stop IK rebind: 10.658 s
candidate 1 mesh path check:                 4.081 s
candidate 1 occupancy path check:           20.938 s
candidate 2 failure before collision:        0.032 s
RRTConnect calls:                                 0
```

Candidate 1 was rejected at its occupancy goal with UNKNOWN/occupied robot-link results,
but the goal query evaluated all remaining STLs after the first blocker. Candidate 2 then
hit `ValueError: motion planning waypoints must preserve exact start and goal`. Replaying
the recorded current state and candidate vector reproduced the cause: interpolation
changed two goal components by `1.11e-16 rad`. D026 preserves exact endpoint tuples at
every interpolation boundary and makes bound online occupancy queries stop on their first
blocking geometry. This is code-verified; the timing improvement and first physical motion
remain unverified.

## 7. Required next action and regression result

The next action is one fresh eiai physical validation after pulling the D026 endpoint and
fail-fast correction. The existing `_003` motion envelope remains the current binding;
D026 changes neither ServoJ limits nor collision thresholds. Use the runbook and a new
`run_id`; because the 15:44 attempt caused no motion and the operator reported no scene
change, its first-view ROI may be reused only while that remains physically true. After
the approval token, verify one reverse connection, one complete viewpoint motion,
endpoint settle, `writeIdle`, sampled stationary pose, and transition to the next capture.

Current status is:

```text
offline/code regression: complete for D026
real-data view planning: complete (0.78 s)
full physical single-view-to-motion workflow: hardware verification pending
```

Using the main source tree with the available local test environment:

```text
full suite: 1248 passed, 3 skipped in 101.41 s
D026 endpoint/occupancy focus: 51 passed
real OMPL binding probe: clear detour, maximum resampled step below 0.02 rad
ruff: all checks passed
```

The skipped tests require optional local PyTorch/Open3D packages absent from the vale test
environment. CUDA availability and FoundationStereo must still pass `scan doctor` on eiai.

## 8. Update template

Append or replace the relevant sections after the next event:

```text
UTC/local time:
main commit tested:
eiai git status:
placement_id:
run_id/output:
last successful phase:
exact blocking line:
controller robot_mode/runtime_state:
relevant snapshot/event paths:
offline reproduction:
tests added and results:
next single action:
```

## 9. D027 control-boundary regression

The next optimization target remains candidate quality only after one normal physical
motion succeeds. A fresh HoloRobot comparison found that the active planner, multi-branch
Pinocchio IK, straight-first/bounded-RRT route order, velocity/acceleration timing, ServoJ
stream loop, endpoint feedback hold, and `writeIdle` segment boundary are aligned.

Two remaining runtime deviations were corrected:

- guarded execution now uses HoloRobot's `0.001 rad` plan-start tolerance instead of
  `0.01 rad`, preventing a collision-validated but non-streamed start bridge from becoming
  the first ServoJ command jump;
- automatic post-motion capture reuses the already completed segment-boundary stationary
  evidence and unchanged stop generation, rather than issuing a second `writeIdle` and
  repeating the same settled interval.

The outer runtime preserves the runner's exact execution blocker and the console reports
progress every five seconds across guarded enable, ServoJ, settle, capture/inference and
the next bounded plan. No candidate-generation, IK, collision, UNKNOWN, clearance,
tracking, or acceptance threshold changed. The eiai `_003` motion-envelope asset therefore
remains applicable. Code regression is complete; the first physical motion on this change
is still pending. The final local result is `1252 passed, 3 skipped`; the skips are the
same optional vale PyTorch/Open3D tests, and the focused motion/state-machine suite reports
`276 passed`. Ruff and `git diff --check` pass.

## 10. D028 full-chain audit and current result

The HoloRobot-first audit was split across motion execution, candidate/IK planning, and
coarse-to-fine/state-machine persistence. It found that the active route planner itself
was already straight-first with bounded RRTConnect, but several surrounding layers could
still multiply latency or turn a recoverable condition into a complete failed run.

The following corrections are now code-verified:

1. ServoJ preparation and the first unchanged stream command receive one bounded reverse-
   connection recovery, matching HoloRobot's persistent external-control boundary. Later
   stream commands are never retried because their execution state is ambiguous.
2. IK solutions are yielded in HoloRobot seed order and endpoint collision is checked
   immediately. A clear first branch stops the solve; a colliding branch cannot hide a
   later clear branch.
3. Every bounded adaptive candidate prefix now covers tilt, roll, azimuth and inward/
   outward distance before expanding the Cartesian product. The former first-32 prefix
   could falsely test only one narrow part of the configured family.
4. Fin discovery is no longer frozen at the first-view joint posture. After each accepted
   stop, a new immutable discovery revision is evaluated from the latest joints. The next
   generation binds the exact revision used, while initialization, proxy and view plan
   remain append-only.
5. Absence of a complete initial opposing fin pair no longer suppresses an otherwise
   informative, endpoint-feasible normal view. Actual bilateral/two-face fin evidence is
   still required at schema-5 promotion.
6. Read-only live timeline/GUI failures disable that observer with a warning and cannot
   abort science, occupancy or motion. Authoritative checkpoint callbacks remain strict.
7. A path-blocked coarse run now requests an operator-positioned `SAFETY_REFRESH` instead
   of stopping the whole experiment. The stale proposal is discarded, the refresh updates
   occupancy only, and IK/NBV/path selection restarts from the new stopped posture.
8. One NBV selection plus its selector-bounded path queue has a 30-second responsiveness
   budget. D029 makes it a shared cooperative deadline and records the indivisible native-
   call boundary. Experimental motion also has a finite duration derived from the approved
   ServoJ stream and existing controller/settle limits when release timing is bypassed.
9. `scan doctor` now parses the ES68 kinematics and stereo calibration, checks calibration
   resolution, probes the RealSense Python API, and treats a configured-but-missing OMPL
   fallback as failure. Malformed assets fail before output reservation or hardware open.

No IK, collision, UNKNOWN, clearance, tracking, operator approval, or final bilateral-fin
gate was relaxed. Original URDF/STL collision geometry and the accepted static-free/
occupancy contract remain authoritative.

Current offline result:

```text
ruff: all checks passed
full suite: 1271 passed, 3 skipped in 98.95 s
focused planning/runtime/storage suite: passed
physical single-view -> first motion -> automatic next capture: still pending on eiai
```

The three skips are local-environment probes requiring PyTorch/Open3D on vale. eiai must
still pass `scan doctor` and one guarded physical segment; an offline suite cannot prove
camera transport, controller state, workcell placement, or physical collision clearance.

## 11. D029-D031 integrated regression, visualization, and acceptance checkpoint

The current uncommitted main-worktree integration is based on `a7c7286`. Three independent
audits covered motion/NBV planning, coarse-to-fine state composition, and read-only
visualization before main-line cross-review.

Planning now separates stable camera-pose/science semantics from state-dependent robot
feasibility. Both fin-discovery and ordinary surface candidates are re-solved from every
accepted stopped posture; a candidate rejected at the initial view may re-enter the ranked
queue later. Fin pose families are enumerated breadth-first across side/axis, and bounded
search termination distinguishes complete physical exhaustion from candidate, IK, feasible
count, or duration truncation. Historical fin evidence is verified against the immutable
discovery revision that actually selected it.

The 30-second planning/preflight value is now one cooperative absolute deadline across
candidate generation, IK, endpoint collision, straight-path checks, occupancy queries and
bounded RRTConnect. Expiry moves the coordinator to recoverable `MOTION_BLOCKED`. It is not
a hard-real-time upper bound: one indivisible Pinocchio/FCL/KDL/hash/NumPy/state-read call may
finish after expiry before the next check can stop work. No safety or information-gain
threshold was relaxed.

The composed offline runtime now verifies the intended experiment path:

```text
one hard ROI
  -> automatic coarse candidate/approval/motion/stop/capture cycles
  -> schema-5 coarse reference
  -> fine bootstrap and active views
  -> final reconstruction COMPLETE
```

Recovery continues from immutable coarse/fine evidence with a fresh cycle identity and one
occupancy-only `SAFETY_REFRESH`; it restores no old motion/permit/map authority and does not
request the first ROI again. A broken read-only observer cannot abort the experiment.

The existing PySide supervisor now consumes a lightweight atomic
`live_timeline/live_planning.json`. It displays the science-ranked candidate queue, active
and selected candidate, IK/endpoint/straight/RRT status, recorded timings, exact blockers,
selected path, and grey/blue/green candidate camera frusta beside the yellow current camera.
It remains command-incapable. SSH X11 performance is still a deployment limitation; the
next visualization phase is a browser/SSE observer extracted from HoloRobot's read-only
subset, with all control routes excluded.

Code verification after integration:

```text
focused planning/runtime/storage/supervision regression: 334 passed
ruff: all checks passed
git diff --check: passed
full repository regression: 1294 passed, 3 skipped in 99.18 s
physical single-view -> first motion -> automatic next capture: pending on eiai
```

The skipped tests require the optional PyTorch/Open3D packages absent from the vale test
environment; eiai still owns CUDA/FoundationStereo and physical-device validation.

The project memory and root `接受文档.md` now explicitly record the remaining gap between
the desired one-initial-view automatic system and the commissioned implementation: exact
per-segment approval is still required, online path safety is the reviewed HoloRobot
fixed-step sampled contract rather than the former recursive continuous certificate, the
remote browser/SSE observer is pending, and the integrated revision still needs its first
physical NBV motion plus automatic post-motion capture. These are acceptance boundaries,
not completed capabilities.

Do not describe the full physical workflow as complete until eiai has produced one guarded
motion and automatic post-motion capture with this exact integrated revision.

## 12. 2026-09-04 19:04 eiai planning deadline and D032 correction

The physical attempt `planning-test-20260904-190447` used the unchanged
`blade-placement-20260901-01` placement and reached `bootstrap_motion_ready`. FoundationStereo
and the first-view proxy completed, but no approval token was issued and the robot did not
move. The shared planning deadline expired at `30.000836 s` while candidate 4 was inside an
occupancy STL-to-voxel query. Persisted diagnostics separate the consumed budget as follows:

```text
next-view selection/current-stop rebind:  8.634 s
candidate 1 preflight:                    4.384 s
candidate 2 preflight:                    5.003 s
candidate 3 preflight:                    4.870 s
candidate 4 before deadline:              7.075 s
```

The first three candidates were all vetoed at their goal occupancy state by
`environment_occupancy_unknown:forearm_link_0`. Each nevertheless spent `3.40-3.75 s` on
the complete mesh path before occupancy was evaluated. Candidate 4 spent `5.40 s` on its
mesh path before the deadline was observed during occupancy. The final exception label is
the cooperative check point where expiry was detected; it is not evidence that one FCL
call consumed the complete 30 seconds.

D032 restores HoloRobot's goal-state-first order for the separated BiBladeFusion mesh and
occupancy checkers. Online sampled preflight now checks the goal mesh, builds the prospective
ServoJ duration needed for map-freshness authority, checks conservative occupancy goal-first,
and only then evaluates the complete mesh path. A non-clear occupancy path is already a hard
veto, so skipping the still-unchecked mesh path cannot authorize motion. Clear candidates
still require every mesh and occupancy sample before approval. Online occupancy also returns
from inside one robot STL query at the first blocking voxel; standalone diagnostic queries
retain exhaustive link/voxel counts and mark whether each query was complete.

The attempt also exposed a state-composition regression. The coordinator correctly emitted
recoverable `MOTION_BLOCKED`, but `SupervisedExperimentRunner.step()` added a terminal runner
block, after which the outer runtime requested stop and persisted `ABORTED`. D032 preserves
`MOTION_BLOCKED` as `NEEDS_CAPTURE`; the outer coarse runtime remains active and the console
can request the documented occupancy-only `SAFETY_REFRESH`. Corrupt evidence, failed
persistence, unconfirmed stop and explicit operator rejection remain terminal.

Code verification on eiai after D032:

```text
focused planning/runtime/supervision regression: 223 passed
full repository regression: 1301 passed, 1 skipped in 171.62 s
ruff: all checks passed
git diff --check: passed
physical first NBV motion and automatic next capture: still pending
```

The single skip is the CUDA-only unit case because the test process did not expose CUDA. The
real D435i/FoundationStereo/controller path was not opened by these tests. The next physical
action remains one fresh run with a new `run_id` and output after `scan doctor`; do not use
the aborted attempt as proof that the corrected timing reaches approval.

A diagnostic-only replay then loaded the exact hash-verified occupancy snapshot, stopped
joint vector and first four persisted target joints from the failed run. Their corrected
preflights took `0.279 s`, `0.277 s`, `0.272 s` and `0.264 s`; each retained the original
`environment_occupancy_unknown:forearm_link_0` blocker and evaluated one goal mesh sample plus
one goal occupancy sample. The production `read_occupancy_mapping` semantic replay could not
run because this test process reported `torch.cuda.is_available() == false` while the artifact
requires CUDA DDA. Therefore these timings demonstrate the corrected call path only; they are
not motion-eligible occupancy evidence and do not replace the next CUDA/physical validation.

## 13. 2026-09-04 19:39 second physical planning result and D033 exact-cache fix

The fresh attempt `planning-fix-d032-20260904-193942` retained the same unchanged physical
placement and confirmed the D032 state fix: planning expiry produced
`phase=coarse_scan runner=motion_blocked disposition=needs_capture`, so it did not falsely
abort the coordinator. The operator then entered `q`, after which the expected explicit-stop
path persisted `ABORTED`. No approval token was issued and the arm never moved.

The first-view selector completed in `9.284594 s`. Goal-first rejection reduced candidates
1-4 and 6-10 to approximately `0.25-1.47 s`, but candidate 5 had a late straight-path
occupancy block and bounded RRT fallback, and candidate 11 was still being fully checked
when the common deadline was observed at `30.023722 s` after sampled mesh pose 13. This was
therefore a remaining repeated-query cost, not a controller or motion failure.

D033 keeps all existing vetoes and the 30-second deadline. Within one checker and immutable
snapshot content hash it now caches only (a) the conservative voxel classification/X-run
layout for an integer broadphase AABB and (b) the corresponding immutable HPP-FCL voxel-run
box/transform. Every sampled robot pose still uses the original URDF collision STL and calls
HPP-FCL exact distance against every relevant dangerous run. Cache entries are discarded on
snapshot-content change; UNKNOWN, occupied, accepted-static-free and out-of-grid categories
remain distinct.

After the goal mesh veto, online straight-path preflight evaluates the complete goal-first
occupancy path before the complete mesh path. This is a deliberate order-only divergence
from HoloRobot's per-waypoint mesh-then-environment loop: the accepted path is the same
intersection of both complete sampled gates, while an occupancy blocker avoids mesh work
that cannot reverse the veto. RRT state validity still checks mesh before occupancy, and a
clear RRT result is fully resampled through both gates.

A diagnostic-only replay used this attempt's exact snapshot, stopped joint tuple, 13-item
ranked queue and deployed configuration. Candidate 11 remained the first CLEAR candidate.
Its occupancy-only straight-path check took `2.935 s` (previously approximately `5.998 s`).
The complete candidate preflight queue through rank 11 took `15.857475 s`; adding the
recorded selector time gives `25.142069 s`, leaving `4.857931 s` below the unchanged deadline.
Candidate 5 took `5.365757 s` including its unchanged bounded RRT failure. The replay process
did not expose CUDA, so this is timing/path evidence only and is not a motion-eligible
semantic attestation.

Code verification after D033:

```text
focused motion/occupancy/runtime regression: 132 passed
full repository regression: 1303 passed, 1 skipped in 172.25 s
ruff: all checks passed
git diff --check: passed
physical first NBV motion and automatic next capture: still pending
```

The next physical attempt must use a new `run_id` and output directory. It must reach
`waiting_approval` before any token is entered; the offline margin is evidence for retrying,
not proof of controller, camera, CUDA or real-motion acceptance.

## 14. 2026-09-04 20:20 first approved motion and D034 control correction

Run `planning-fix-d033-20260904-201710` physically validated the D033 planning fix. It reached
`waiting_approval` in `20.01 s`, accepted the exact token and consumed the bound permit. The
stored straight path had 1545 commands at 4 ms (`6.176 s`) and was collision/occupancy CLEAR.
Execution then aborted on `tracking_error_exceeded`; no automatic capture occurred.

A read-only RTSI sample after shutdown reported `safety_status=NORMAL`,
`runtime_state=STOPPED`, zero actual joint velocities, and joints
`[3.730767488, -1.963301583, 2.025609657, -2.172934919, -2.392959527,
-0.003615902] rad`. Relative to the approved start, this is only about `0.187 s` along the
stored linear 4 ms path. The planned J6 target was advancing at `0.256894 rad/s` immediately
from the first tick. This establishes a command/controller progression mismatch rather than
an IK or collision failure.

D034 makes two bounded corrections:

1. ServoJ paths now use a true rest-to-rest triangular/trapezoidal profile. For the same
   start/goal under the replacement eiai settings, offline generation produces 1362 commands
   at 8 ms (`10.888 s`), J6 peak velocity `0.160518 rad/s`, peak acceleration
   `0.159896 rad/s^2`, and first/last interval speed about `0.000640 rad/s`.
2. `configs/local.yaml` now matches the HoloRobot physical Elite-A baseline: ServoJ/stream
   period `0.008 s`, lookahead `0.03 s`, warmup `0.2 s`, Dashboard speed scaling `0.05`,
   and trajectory speed scaling `0.05`.

The existing tracking threshold remains `0.03 rad`. A future abort will persist structured
stream and last-feedback diagnostics instead of only `tracking_error_exceeded`. The focused
planning/execution/coordinator suite reports `141 passed`; the full repository regression
reports `1306 passed, 1 skipped` in `184.78 s` (CUDA-only test skipped because CUDA was not
available in the test process).

This changes the motion-control hash to
`c7a4662c3cc17e1b1ba7ae0a9dcabffa62246a1a230d425a3e2fa79148acacb5`.
The old `_003` acceptance is retained unchanged but is no longer active; the local
motion-envelope path and ID are intentionally null. `scan doctor` therefore blocks unknown-
blade motion until a new bounded commissioning sequence is recorded. Do not rerun the long
NBV command before that replacement acceptance exists.

For that replacement commissioning interval, the local deployment is explicitly in
commissioning mode: `robot.motion_enabled=false` and `stop_and_capture.enabled=false`.
The dedicated commissioning executor alone creates a temporary motion-enabled robot config
after consuming its exact output-bound token. The failed dry-run before this switch did not
connect to or move the robot.

The first subsequent execution attempt also made no motion: EliteDriver creation failed
before `before_state` or any ServoJ stream existed because `xray` held the established
loopback source port `50002`. The immutable failed output is
`data/acceptance/d034_20260904-204620_trial_01_forward`. D035 moves the eiai-local SDK ports
from the ephemeral-range values `50001–50004` to the currently free non-ephemeral values
`29001–29004`; the next attempt must use a new output directory and newly printed token.

## 15. 2026-09-04 21:00 D034 retry result and D036 controller/endpoint correction

The physical retry
`data/acceptance/d034_20260904-204620_trial_01_forward_retry01` validates the D035 port
correction and the 8 ms host stream timing. EliteDriver connected and sent 215 commands in
`1.722458 s`; average/max/p95 tick periods were `8.048/8.163/8.126 ms`, no loop-body overrun
was recorded, and the maximum tracking error was `0.018453 rad`, below the unchanged
`0.03 rad` abort threshold. This trial nevertheless failed correctly because it could not
establish the required stationary goal window.

The 215 commands comprised the 90-command rest-to-rest segment plus a fixed 125-command
endpoint tail. Immediately before `writeIdle`, goal error was `0.001463 rad`, but feedback
still reported up to `0.009050 rad/s`. After `writeIdle`, the arm crossed the goal and came
to rest `0.013553 rad` outside it; maximum measured stop drift was `0.015016 rad`. This is
not accepted endpoint or stop behavior, so production motion remains unauthorized.

D034 had incorrectly treated HoloRobot's trajectory `speed_scaling=0.05` as also being the
Elite Dashboard setting. The source configuration for the same Elite-A at `192.168.6.60`
actually uses `default_speed_scaling=1.0`, while its trajectory stream independently uses
`speed_scaling=0.05`. D036 restores that separation in `configs/local.yaml`. Commissioning
now sends only the preflighted rest-to-rest stream, then invokes the same guarded feedback-
verified endpoint hold used by production execution before requesting `writeIdle`. The
existing post-stop `0.002 rad`, velocity, one-second stationary-window and five-second
timeout gates are unchanged.

The resulting motion-control hash is
`3a85600d873cd05eb7738a96a832c0a216c08d5da03b851f284b1aa10016db30`.
The old candidate cannot be retried in either direction: the stopped joints are
`0.033553 rad` from its old start and `0.013553 rad` from its old goal, both outside the
`0.001 rad` nominal live-start gate. A fresh stopped stereo/occupancy snapshot and a new
candidate/output-bound token are required. The failed artifact remains immutable.

Code verification after D036 currently records:

```text
commissioning/device/guarded-execution focused tests: 119 passed
full repository regression: 1305 passed, 1 skipped in 171.56 s
ruff and git diff --check: passed
physical D036 forward commissioning trial: pending
```

## 16. 2026-09-04 21:27 D036 forward commissioning PASS

The attended D036 forward trial
`data/acceptance/d036_20260904-211858_trial_01_forward` passed physically with candidate
`aed51cffd459e8e988d4618329eb9eadc54fccfc9c5ace95f391f7761f29909b`.
The current-pose start matched exactly, the 90-command rest-to-rest stream completed in
`0.718340 s`, and maximum tracking error was `0.003168 rad`. Average/max/p95 host tick
periods were `8.070/8.186/8.130 ms`, with no loop-body overrun or watchdog error.

Before `writeIdle`, the reused guarded endpoint loop held the approved endpoint for three
consecutive feedback samples. It settled in `0.040423 s`; maximum/final endpoint tracking
errors were `0.000247/0.00003185 rad`. After `writeIdle`, maximum six-axis stop drift was
`0.00006060 rad`, stop acknowledgement was `0.00002671 s`, and the required continuous
stationary window lasted `1.009667 s`. Its final goal error was `0.00003223 rad`, with zero
joint and TCP speed under `RUNNING/NORMAL` feedback.

This physically validates the D036 Dashboard/trajectory separation and feedback endpoint
boundary for one bounded forward segment. It does not authorize production motion. The arm
is within the unchanged `0.001 rad` reverse live-start tolerance of the sealed forward goal,
so the next action is a separately output-bound reverse dry-run, followed by one attended
reverse trial only after its exact token is reviewed. Intentional tracking-fault and final
motion-envelope acceptance remain pending.

## 17. 2026-09-04 21:41 reverse pre-stream rejection and D037 live-bound rebind

The first D036 reverse attempt
`data/acceptance/d036_20260904-211858_trial_02_reverse` failed before any ServoJ write.
Its `stream_result`, endpoint-settle and stop-request fields are null; the before/after
joint vectors are identical and stationary under `RUNNING/NORMAL`. The arm therefore
remains at the successful forward endpoint, and the failed output is immutable evidence
rather than a motion trial.

The rejected live state was only `0.00003223 rad` from the sealed reverse start, well inside
the unchanged `0.001 rad` start tolerance. Because the sealed reverse segment was already
exactly `0.02 rad` on J4 and the residual lay in the adverse direction, measuring from the
live state to the sealed goal yielded `0.02003223 rad`. The implementation simultaneously
allowed a nonzero live-start tolerance and required the live-to-sealed-goal delta to remain
at most `0.02 rad`; those requirements were not closed under ordinary endpoint residual.

D037 does not relax either bound. For every nominal or intentional-fault commissioning
execution, it deterministically projects from the measured live start toward the sealed
goal and clips only when needed to the candidate-specific trial bound (never above
`0.02 rad`, or `0.01 rad` for the fault trial). It then rebuilds the rest-to-rest stream and
reruns the complete original-mesh preflight from the exact live start to the rebound goal.
The sealed goal, measured start, scale, requested delta, executed delta and actual goal are
persisted separately.

Exact offline replay of the failed state produced scale `0.9983909949`, executed delta
`0.0200000000 rad`, goal displacement from the sealed return point `0.00003223 rad`, 90
commands over `0.712 s`, CLEAR status, valid continuous swept-volume evidence and minimum
certificate margin `0.004175745 m`. Focused execution/storage/device tests report
`125 passed`; the full repository reports `1306 passed, 1 skipped in 172.59 s`; ruff and
`git diff --check` pass. The skip is the existing CUDA-only test.

The HoloRobot reference keeps a sealed stream and accepts a start mismatch up to its start
tolerance; it has no commissioning-specific `0.02 rad` physical-segment promise. D037 is a
deliberate bounded-commissioning adaptation: it preserves that start tolerance while adding
an exact live segment cap and fresh mesh proof. The configuration-derived motion-control
hash remains `3a85600d873cd05eb7738a96a832c0a216c08d5da03b851f284b1aa10016db30`.
A retry needs only a new reverse output path and token; no new capture/candidate is required
because the failed attempt did not move the arm.

## 18. 2026-09-04 21:52 D037 reverse PASS and D038 fault-window correction

The attended reverse retry
`data/acceptance/d037_20260904-211858_trial_02_reverse_retry01` passed. Its measured live
start residual was `0.00003223 rad`; D037 rebound the requested `0.02003223 rad` return to
exactly `0.02000000 rad` with scale `0.9983909949` and reran the complete mesh proof. The
90-command stream finished in `0.714369 s` with `0.003236 rad` maximum tracking error.
Average/max/p95 tick periods were `8.026/8.136/8.124 ms`, with no loop overrun or watchdog
error.

The guarded endpoint hold completed in `0.040300 s` with `0.00002164 rad` final error.
Maximum post-idle stop drift was `0.00005387 rad`; stop acknowledgement was
`0.00002548 s`; the `1.009718 s` stationary window ended with zero joint/TCP speed. Final
feedback equals the candidate's original sealed start under `RUNNING/NORMAL`. Forward and
reverse nominal commissioning are therefore physically complete, but production motion
remains unauthorized until the intentional-fault trial and replacement acceptance exist.

Inspection before that final trial found a deterministic incompatibility in its old setup.
The intentional mode retained a real `0.001 rad` tracking threshold but exposed only the
first ten commands of the new rest-to-rest profile. Both nominal physical trials measured no
more than `0.00040904 rad` error in that window and first crossed `0.001 rad` at index 15, so
the old fault trial would predictably fail without testing its stop path.

D038 keeps the real feedback comparator and `0.001 rad` intentional threshold. It checks
every command and caps the trial at 24 commands. The exact 0.01-rad fault trajectory has 64
commands and its first-24 commanded excursion is only `0.00266566 rad`. Execution now rejects
the fault trial before any write unless that truncated excursion lies inside
`[0.002, 0.003] rad`, and stores the measured excursion in the immutable result. Production
tracking remains `0.03 rad`; collision, start tolerance, live-segment bounds, stop and
stationarity thresholds are unchanged. Focused commissioning/device/acceptance regression
reports `69 passed`; the full repository reports `1306 passed, 1 skipped in 171.72 s`.
Ruff and `git diff --check` pass; the skip is the existing CUDA-only test. The next hardware
action is one new-output forward `--intentional-tracking-fault` dry-run; it does not connect
to the robot or attempt motion.

## 19. 2026-09-04 22:02 D038 fault PASS and motion-envelope acceptance `_004`

The final bounded commissioning trial
`data/acceptance/d038_20260904-215214_trial_03_tracking_fault` passed. The intentionally
strict real-feedback comparator crossed `0.001 rad` at command 16 with maximum error
`0.00101258 rad`; the inner stream returned `tracking_error_exceeded`, which is the expected
condition for this trial. The outer trial then acknowledged stop in `0.00003152 s`, observed
maximum six-axis drift `0.00043768 rad`, and completed a `1.010317 s` continuous stationary
window with zero joint/TCP speed. The command window remained at the pre-reviewed
`0.00266566 rad`, below its `0.003 rad` cap, with no watchdog error.

This completes the three replacement-contract physical trials: D036 forward nominal, D037
reverse nominal and D038 intentional tracking-stop. No further commissioning run is required.
The new immutable acceptance is:

```text
path: data/acceptance/es68_d435i_motion_envelope_004
acceptance_id: 29624b08242d2c8ef7544cb958bf2a64335f895b719b421f792ff8c750719f9b
metadata_sha256: ce807010bc11bdf50bcfb804214f85705ad155bf3c7af16d3a214a2354722766
motion_control_contract_sha256: 3a85600d873cd05eb7738a96a832c0a216c08d5da03b851f284b1aa10016db30
```

The declaration does not shrink the envelope to the latest short-segment measurements. It
retains larger per-axis tracking values from the previous representative-workspace physical
evidence and takes the per-axis maximum stop drift across the old and three new trials, with
the unchanged `1.5` safety factor. The resulting accepted joint uncertainties are
`[0.02392165, 0.00315935, 0.00863424, 0.00551003, 0.02877571, 0.02429119] rad`.
Unchanged collision-assembly, bootstrap and emergency-stop checks are inherited; the changed
ServoJ contract, nominal boundaries and tracking-stop path are verified by the new trials.

`configs/local.yaml` now binds `_004` and restores both `robot.motion_enabled=true` and
`stop_and_capture.enabled=true`. Strict readback confirms matching acceptance ID, robot
geometry, collision contract and control hash. A sandboxed `scan doctor` passed every
structural and acceptance check but could not see host CUDA and therefore reported only
`scan_foundation_stereo_device=FAIL`; this is not a host readiness result. The next action is
a host `systemd-run` CUDA doctor followed, on PASS only, by a fresh real unknown-blade run.

## 20. 2026-09-05 D038 first NBV segment, exact no-path diagnosis and resumable D039 code

The experimental run
`data/experiments/blade-placement-20260901-01-real-nbv-d038-20260904-220200`
completed the first physical NBV leg. The 1,342-command ServoJ stream took `13.236086 s`,
then one `writeIdle`/settle boundary obtained 21 stationary samples over `1.039144 s`.
Final goal error was `0.00002409 rad`, maximum stop-window joint drift was
`0.00005387 rad`, and the next FoundationStereo capture, ROI propagation and generation
`coarse_science/generations/000001` were committed without another hard-ROI annotation.

The next planning attempt did not hide a reachable segment. Exact offline replay preserved
all 11 science-ranked candidates and the same first target. Candidates 1-10 are blocked by
UNKNOWN occupancy in the two-source `MAPPING` prefix. Candidate 11 was allowed to finish
outside the production deadline for diagnosis: straight and bounded-RRT occupancy paths were
CLEAR, but both final sampled mesh paths were BLOCKED by
`self_clearance:forearm_link_0:d435i_collision_link_0`. Its full diagnostic preflight took
`74.767 s`; it is not executable. The recorded two-source D038 state therefore contains no
safe automatic path, and no collision, UNKNOWN, queue or 30-second gate was relaxed.

D039 code removes the misleading and duplicated work around that result:

1. one strict-read transaction memoizes immutable coarse generations while rechecking all
   declared view, reconstruction, stereo and occupancy bytes before typed reuse;
2. an already full-semantic occupancy result carries a private storage authority, so the
   final consume boundary streams SHA-256/size/header checks instead of reconstructing eight
   large snapshots or replaying CUDA rays; D038's two-view check fell from about `15.56 s`
   to `0.3668 s`;
3. exact discovery policy and stopped-joint bindings reuse the already completed endpoint
   IK/collision result; any policy, posture or authority mismatch falls back to the original
   full rebind. The D038 selector fell from `45.89 s` to `5.45-5.59 s`, with identical queue,
   TCP targets and only `2.56e-8 rad` solver-level joint variation against the old rebind;
4. a planning deadline is now the terminal typed blocker `planning_restart_required`, not a
   false claim that occupancy alone blocked motion. Occupancy/path vetoes still expose the
   separate operator-positioned safety-refresh route;
5. old authorityless experimental chains such as D038 may resume only with
   `--experimental`; production/experimental mismatch fails before hardware. Resume restores
   science checkpoints but never restores a proposal, permit, occupancy freshness or motion
   authority.

The D038 handoff was re-read on host CUDA in `65.401 s`: phase `coarse`, generation
`000001`, original placement ID, `experimental_unaccepted=true`, and no science/timing
production authority. Code validation reports `552 passed` for the combined affected set and
`1337 passed, 1 skipped in 176.22 s` for the repository; Ruff and `git diff --check` pass.
The skip is the existing CUDA-only unit test in a non-CUDA pytest process. No robot/camera
connection or experiment write occurred during these replays.

The next physical action is exactly one D038 resume. Keep the blade, fixture, camera mount and
robot base unchanged, and require the robot to be stopped. After resume measures and verifies
that stopped pose, do not reposition it before entering `c` once at the recovery prompt to
capture the third independent occupancy-only safety source. That refresh does not count as
blade science. Plan again from the resulting current map; paste an approval token only if the
new proposal reaches `waiting_approval` and the displayed path agrees with the scene. If any
physical reference moved, do not resume D038; start a new placement/run.
