# BiBladeFusion decision log

This log records method and workflow decisions that must survive chat compaction and new
tasks. Add new entries; do not rewrite old rationale merely because implementation details
change. Corrections should reference the superseded decision explicitly.

## D001 — Paper contribution is the complete active measurement system

Date: 2026-09-03

Status: accepted

The work will be presented as one complete robotic reconstruction system rather than as a
large suite of isolated algorithm improvements requiring extensive ablation. The central
novelty is blade-specific, robot-realizable active viewpoint planning. Reconstruction and
thermal mapping are downstream components.

## D002 — Target workflow begins from one initial view

Date: 2026-09-03

Status: accepted

The desired end state is not a permanently manual three-view initialization. The operator
selects one safe initial camera view and draws one blade polygon. Subsequent acquisition,
coarse discovery, side crossing, and fine NBV should proceed automatically apart from
temporary supervised approvals used during experimental commissioning.

## D003 — Distance and incidence angle are adaptive variables

Date: 2026-09-03

Status: accepted

The original `+-15 degree` fin observation poses remain useful seeds, not rigid
constraints. Candidate distance, tilt, and roll may vary. Effective new information is
the positive objective; IK, workspace, collision, and path safety are vetoes. A failed IK
candidate does not invalidate the method when another useful reachable candidate exists.

## D004 — Keep three geometry domains separate

Date: 2026-09-03

Status: accepted

Robot self/environment checking uses URDF and original collision STL meshes. The safety
occupancy map integrates the full scene after robot self-masking. The blade scientific
model uses blade ROI/support only. No blade ROI may crop safety evidence, and no full-scene
occupancy cloud may be treated as the blade reconstruction.

The previous large per-link circumsphere behavior caused false collisions and is not an
acceptable final geometry approximation. Broad-phase AABBs remain allowed only for
acceleration.

## D005 — Current outer workspace and static-free interpretation

Date: 2026-09-03

Status: accepted for the current workcell

The current workspace is `x=[-1.00,1.00] m`, `y=[-0.55,0.55] m`, and
`z=[0.00,1.10] m`. The robot base `z=0` is slightly above the table, and table-height error
is intentionally ignored for this experiment.

Accepted-static-free AABBs state that the relevant space outside the target envelope is
physically empty despite missing ray evidence. They do not turn observed OCCUPIED voxels
into FREE, do not authorize motion by themselves, and must be regenerated if the workcell
or target exclusion envelope changes.

## D006 — First ROI is manual; later ROIs are identity-preserving projections

Date: 2026-09-03

Status: accepted

The first rectified left image receives exactly one operator polygon labelled `blade`.
Later masks are derived by projecting the accepted blade proxy, applying a bounded image
dilation, checking foreground/depth consistency, and intersecting with the blade science
envelope. Generic automatic component selection must not silently substitute an unrelated
mask.

## D007 — Experimental safety gates must protect concrete invariants

Date: 2026-09-03

Status: accepted

This remains a supervised robot experiment with exact geometry, collision/path checking,
controller stop monitoring, and an accessible physical emergency stop. However, arbitrary
tight timing gates, duplicated expensive proofs, and production-release ceremony must not
consume the experiment or repeatedly block valid stopped data. Every gate must name the
physical or evidence-integrity failure it prevents and use a measured, sufficient budget.

## D008 — Main planning work and thermal work remain isolated

Date: 2026-09-03

Status: accepted

Unknown-blade planning/runtime fixes are developed and committed in the clean main
worktree. Thermal-camera feature changes remain in their separate worktree/branch until an
explicit integration decision. A push to `main` must not accidentally include thermal
work.

## D009 — Repository documentation is the authoritative long-term memory

Date: 2026-09-03

Status: accepted

Codex local memories and chat compaction help continuity but are not authoritative enough
for a long-running physical experiment. `AGENTS.md` requires each new task to read the
stable project memory, current state, decision log, and hardware runbook. Material
decisions, results, and operational commands are updated in Git so that vale, eiai, new
tasks, and future context compactions recover the same state.

## D010 — HoloRobot-first regression and reuse

Date: 2026-09-03

Status: accepted

The current sequence of hardware failures is treated as evidence of an architectural
control-stack regression, not as a queue of independent symptoms. Before making more
local fixes, compare BiBladeFusion end to end with the operator-owned reference project
at `~/Documents/HoloRobot`.

For general mechanisms already present there—robot connection, continuous state and
visual feedback, ServoJ motion, enable/recovery/stop, stationarity, synchronization, and
occupancy-map computation—reuse or minimally adapt the proven HoloRobot implementation.
Only design from scratch where HoloRobot lacks the blade-specific requirement. The
operator explicitly authorizes reuse between these projects.

The regression must also address experiment turnaround time. Measure every phase from the
first capture to the first motion, aggregate expected candidate IK failures instead of
printing unbounded warning floods, and eliminate duplicated work whose inputs have not
changed. No further physical run should be requested until the HoloRobot comparison,
offline reproduction, and focused control-lifecycle tests are complete.

## D011 — One reverse-control session begins at approved resume

Date: 2026-09-03

Status: partially superseded by D021; complete-viewpoint semantics retained

Guarded enable prepares power, brakes, speed scaling, and the driver object but does not
launch an external-control script while the software stop latch is still held. The fresh
one-shot permit authorizes guarded resume to atomically clear that exact latch and open
the reverse session once. This removes the observed unused first session and retains the
rule that no motion command precedes approval.

## D012 — ServoJ endpoint settles before the stop-and-capture boundary

Date: 2026-09-03

Status: superseded in part by D014; offline verified, physical verification pending

Reuse HoloRobot's endpoint-hold feedback policy: repeat only the already approved final
setpoint, sample the persistent RTSI joint state, and require three consecutive samples
within `0.005 rad` inside `2 s`. Only then send `writeIdle`. Normal segment completion no
longer requests Dashboard STOPPED; see D014. This is convergence of the commanded segment,
not an arbitrary post-motion delay.

## D013 — Default candidate IK is HoloRobot MDH/DLS

Date: 2026-09-03

Status: superseded in part by D019; offline verified

Candidate reachability uses the calibrated controller MDH chain, finite-difference world
Jacobian, damped least-squares update, ES68 limits, and multi-seed strategy adapted from
HoloRobot. The vendor KDL solver remains injectable for historical reproduction but is
not the normal online candidate path. Expected unreachable candidates are returned as
data and no longer flood the hardware console with KDL warnings.

## D014 — Normal segment stop is the HoloRobot `writeIdle` boundary

Date: 2026-09-04

Status: accepted; offline verified, physical verification pending

After endpoint feedback convergence, normal execution sends `writeIdle(0)` and latches a
new immutable stop generation. It does not call Dashboard `stopProgram`, so the reverse
session remains available and does not reconnect solely to stop. The following stationary
window accepts a sampled `RUNNING/PLAYING` controller only while the stop generation is
unchanged and joint/TCP pose remains within threshold. Bootstrap and independent deadline
fault stops retain the stronger Dashboard path.

## D015 — One NBV is one complete motion and one capture

Date: 2026-09-04

Status: accepted; offline verified, physical verification pending

The coordinator no longer converts `maximum_segment_joint_delta_rad=0.02` into artificial
intermediate captures. It executes one complete joint-linear path to a viewpoint;
`motion_preflight.maximum_joint_step_rad` remains internal interpolation. D021 supersedes
the continuous-proof and complete-queue clauses: online validation now follows HoloRobot's
sampled segment contract and observes the configured ranked-candidate work bound.

## D016 — Live occupancy reuses verified work without weakening disk replay

Date: 2026-09-04

Status: accepted; offline verified

An appended independent source reuses the committed update prefix and integrates only the
new frame. Replacement, sliding, discontinuous, or mismatched windows still rebuild from
scratch. A file-identity-bound in-process cache lets the live writer hand the already fully
validated mapping to the coordinator without immediately raycasting it again; fresh
processes and changed files retain full semantic and ray replay.

## D017 — Measurement coverage and reconstruction QA are separate gates

Date: 2026-09-04

Status: accepted; offline verified

Active measurement completes when required blade-side, boundary, and fin regions satisfy
the coverage contract. Downstream mesh/watertight reconstruction QA is recorded but does
not block further measurement by default. Deployments may opt into strict reconstruction
blocking. Fine NBV applies the same bounded adaptive distance/incidence fallback used in
coarse discovery when a nominal pose is unreachable.

## D018 — Synchronized absolute paths relocate only by content identity

Date: 2026-09-04

Status: accepted; offline verified

Initialization and occupancy artifacts copied between the vale and eiai checkouts may
resolve a missing historical absolute path under the active project only when the candidate
file has the exact recorded size and SHA-256. Existing-but-different files and ambiguous
content continue to fail closed.

## D019 — Candidate IK uses an analytic fixed-MDH Jacobian

Date: 2026-09-04

Status: accepted; offline verified

The D013 solver contract remains, but its finite-difference Jacobian is replaced by the
analytic world geometric Jacobian derived from the calibrated fixed-MDH chain. This removes
six FK perturbations per iteration while retaining DLS, joint limits, multiple seeds,
nearest equivalent joint wrapping, FK replay, and bounded search time.

## D020 — Runtime candidate IK and fin discovery use the shared Pinocchio model

Date: 2026-09-04

Status: accepted; offline real-data replay verified, physical verification pending

The 2026-09-04 09:56 physical attempt completed the first capture and FoundationStereo
stage but rejected every symmetric back-side fin pair before motion. Its diagnostic also
mislabelled one composite IK checker as one seed. The runtime now reuses the already loaded
collision URDF's Pinocchio kinematic model, following HoloRobot's current preferred
URDF/Pinocchio IK backend and its bounded near-pose seed perturbations. The analytic MDH
solver remains an offline fallback when a shared Pinocchio model is unavailable.

An opposing fin pair constrains the sign of the fin-axis component; it does not require
two camera positions symmetric about that axis. Candidate azimuth now searches the signed
open half-plane, including common tangential bias, while distance, incidence and wrist
roll remain variables. The first bounded pass ranks the 45-degree information optimum
before lower-value tilts and interleaves wrist rolls. No IK, workspace, endpoint-collision,
or complete sampled-path gate is relaxed.

Replay of the synchronized attempt-09 proxy changed back-side pair availability from zero
to one complete pair. With the production Pinocchio checker, fin-discovery planning took
about 1.17 seconds, excluding the Pinocchio model load already paid by runtime collision
construction.

## D021 — Online motion reuses HoloRobot's sampled single-arm contract

Date: 2026-09-04

Status: partially superseded by D022/D023; recursive online proof removal remains accepted

The active NBV loop no longer runs BiBladeFusion's recursive continuous-interval mesh and
occupancy certificates. It now mirrors HoloRobot's `ConservativeJointPlanner`: interpolate
the straight joint path at `motion_preflight.maximum_joint_step_rad`, check five evenly
spaced configurations per adjacent segment, and reject immediately on the first non-clear
result. At most `stop_and_capture.maximum_ranked_preflight_candidates` endpoints are checked
in one operator cycle. That final limit is superseded by D023 because it truncated an
already-bounded selector queue.

This change does not restore the former large link spheres. Self-collision still uses the
hash-bound ES68+D435i URDF and original collision STL meshes. Environment checks still use
the immutable occupancy snapshot, accepted static-free contract, conservative UNKNOWN
policy, obstacle/uncertainty clearance, and original robot STL. For speed, adjacent
same-state voxels along X are merged into an exactly equivalent union box before FCL
distance queries. Operator approval, map/model hashes, one-shot permit consumption, ServoJ
tracking supervision, endpoint convergence, and stop/stationarity checks remain required.

The recursive continuous proof remains an offline acceptance and diagnostic facility. A
sampled online result carries its own integrity hash and cannot be presented as a continuous
certificate. D027 supersedes the former live-start bridge: execution now requires the live
state to remain inside HoloRobot's 0.001 rad start tolerance and never executes an unplanned
bridge.

## D022 — Bind occupancy once per path and use HoloRobot's effective resolution

Date: 2026-09-04

Status: accepted; offline verified, physical verification pending

The 11:25 eiai attempt proved that D021 had replaced the recursive certificate but had
still composed two independently reasonable layers incorrectly. BiBladeFusion called the
single-pose occupancy API at every sampled path state; that API deliberately recomputed
and rechecked the entire immutable occupancy hash for an independent query. It therefore
hashed the same map twice per pose. It also interpolated the path at `0.02 rad` and then
applied HoloRobot's five samples to each already-small segment.

One online path is now one immutable occupancy transaction: recompute and bind the map at
entry, query all sampled robot poses against that exact frozen object, and recompute once
at exit (or before returning the first blocker). No map identity, UNKNOWN policy,
clearance, original-STL query, or fail-fast behavior is removed.

HoloRobot's deployed ES68 profile uses `max_joint_step_rad=0.1` and five endpoint-inclusive
samples, whose worst effective spacing is `0.025 rad`. BiBladeFusion's existing
`motion_preflight.maximum_joint_step_rad=0.02` is already finer. Online validation now
checks those pre-interpolated waypoints directly and subdivides only if a supplied path is
coarser than `0.025 rad`. The evidence schema is `holorobot_sampled_joint_v2` so old and
new semantics cannot be confused. The recursive certificate remains offline-only.

The outer unknown-blade runtime must preserve the inner runner's typed blocking reasons;
generic `active supervised runner entered a blocked state` output is no longer acceptable.
Segment timing diagnostics separately record mesh collision, ServoJ generation, and
occupancy collision spans for the next hardware result.

## D023 — Do not truncate the selector's bounded fallback queue

Date: 2026-09-04

Status: accepted; exact eiai artifact replay verified, physical verification pending

The 11:25 eiai event ledger contained seven science-ranked endpoint-feasible candidates,
but the coordinator tried only three because
`stop_and_capture.maximum_ranked_preflight_candidates` was set to three. The first two
straight paths had exact STL self-clearance blockers and the third crossed UNKNOWN with
`wrist_2_link_0`; the coordinator then declared the cycle blocked without inspecting four
already-generated alternatives.

Replay against the exact immutable occupancy snapshot found two collision-free paths among
those four alternatives. Therefore the NBV selector, not a second coordinator limit, owns
candidate-family boundedness. The coordinator now checks the complete returned queue in
unchanged science-gain order and accepts the first safe route. It remains fail-fast within
each route, never relaxes a collision threshold, and never generates new unbounded
candidates. The legacy configuration key stays parseable so reviewed deployment files do
not break, but it no longer discards valid fallback paths.

## D024 — Filter every IK branch, then use HoloRobot composite path planning

Date: 2026-09-04

Status: accepted; offline verified, physical verification pending

HoloRobot's Pinocchio solver already explores a bounded neighboring/preset seed set, but
the old candidate interface selected the nearest IK solution before collision checking.
Consequently a self-colliding wrist branch could reject a camera pose even when another
solution for the same pose was clear. Adaptive coarse and fine planning now expose every
distinct solution, check each endpoint with the already loaded hash-bound ES68+D435i
URDF/STL backend, and select the nearest collision-clear branch. The artifact records each
branch verdict; endpoint feasibility remains a hard gate and contributes no science gain.

After endpoints are valid, online route planning now follows HoloRobot's
`CompositeMotionPlanner` order: conservative straight joint path first, RRTConnect only
for an interior `PATH_BLOCKED` result. UNKNOWN collision state, stale or mismatched map
evidence, and collisions at start or goal fail fast and never enter OMPL. The blade
experiment uses one 1.0 s solve by default instead of HoloRobot's general five attempts of
five seconds. This is a deliberate bounded-latency specialization, not a new planner.

OMPL is optional and pinned as the `motion-ompl` extra. If it is absent, the straight
planner and complete bounded NBV candidate queue remain functional and `scan doctor`
reports the missing fallback as a warning. A successful detour is resampled to the same
joint-step contract, checked again in full against exact robot geometry and one immutable
occupancy transaction, converted into the existing velocity-limited ServoJ stream, and
bound into path evidence by a waypoint SHA-256. No collision threshold, UNKNOWN policy,
operator approval, tracking stop, or stationarity gate is relaxed.

## D025 — Rebind IK at each stop and reuse HoloRobot motion/sampling semantics

Date: 2026-09-04

Status: accepted; offline verified, physical verification pending

D024 described multi-branch endpoint filtering, but the concrete Pinocchio adapter still
stopped after its first converged seed. It now retains every distinct branch from the
bounded HoloRobot seed family, normalizes branches around the current posture, and passes
all of them to the exact endpoint collision gate. In addition, coarse science-ranked
camera poses are solved again from the most recent accepted stopped-state trace before
path preflight. A joint endpoint cached at the initial view is no longer treated as valid
after the arm has moved. Fine NBV already rebuilt its checker from the current stopped
trace; production now injects the same already-loaded Pinocchio/URDF model there as well,
instead of silently falling back to the separate MDH implementation.

ServoJ command generation now copies HoloRobot's velocity-and-acceleration segment timing
rule. The reviewed ES68 acceleration vector is `[4, 4, 4, 4, 4, 4] rad/s^2` and is part of
the immutable motion-control contract. Redundant collinear collision samples are removed
only from the timing polyline; all geometric corners and the separately verified collision
path remain unchanged. Execution reconstructs and compares the exact stream before a
permit can be used.

The production perception cycle no longer creates one process-isolated RTSI connection
per frame or keeps that sampler alive through FoundationStereo and occupancy work. A short
read-only sampler shares EliteArm's persistent, locked RTSI state source and is closed
immediately after the camera bracket. Stationarity validation remains strict over exposure,
while later inference, ray integration, and disk persistence are outside that physical
window. This removes a non-physical failure mode without weakening capture stability.

## D026 — Preserve authoritative endpoints and fail fast within online occupancy queries

Date: 2026-09-04

Status: accepted; exact eiai failure reproduced, code verified, physical verification pending

The `planning-test-20260904-154403` run used `c9f141e`. Its first rejected candidate
spent 4.08 seconds in mesh checking and 20.94 seconds in occupancy checking even though
the prechecked goal pose was already unsafe. The occupancy result continued evaluating
all remaining robot STLs after the first blocking link, producing six diagnostic reasons.
HoloRobot's online collision contract stops on the first unsafe geometry because one is
already a complete motion veto. Bound online state/path queries now do the same; standalone
diagnostic pose checks still inspect every geometry. No occupancy classification,
clearance, UNKNOWN policy, map binding, or robot STL is changed.

The following candidate then failed before collision checking because path interpolation
recomputed its exact goal as `start + 1.0 * (goal - start)`. For the recorded eiai joint
vectors, two components changed by one ULP (`1.11e-16 rad`). That is physically zero but
violates BiBladeFusion's exact waypoint/hash identity contract. Both straight and OMPL
resampling now keep the authoritative start, intermediate OMPL knots, and goal tuples
verbatim; arithmetic interpolation is used only for interior samples. Sampled collision
poses and ServoJ segment endpoints follow the same rule. This is the necessary adaptation
around HoloRobot's interpolation helper because HoloRobot does not impose BiBladeFusion's
strict tuple/hash boundary.

## D027 — Use HoloRobot's exact motion-start boundary and one stop per executed leg

Date: 2026-09-04

Status: accepted; code regression complete, physical verification pending

The experiment must first execute a normal first viewpoint motion before candidate-family
optimization resumes. A control-chain comparison found two BiBladeFusion-only deviations
after planning had already succeeded.

First, HoloRobot rejects a plan when the current joints differ from its stored start by
more than `1e-3 rad`; the guarded BiBladeFusion executor allowed `1e-2 rad`. Although the
larger live-start bridge was collision-checked, it was not inserted into the
time-parameterized ServoJ stream, so the first planned command could still introduce the
unexecuted jump. The production default is now the same `1e-3 rad` as HoloRobot. The
existing measured collision uncertainty remains unchanged and still covers this smaller
state mismatch.

Second, successful execution already performed endpoint feedback settling, one
`writeIdle(0)`, and one full sampled stationary interval. The immediately following
automatic capture issued another `writeIdle(0)` and repeated the same one-second settled
window. The automatic capture now reuses the exact preceding evidence only when its view
ID, capture purpose, immutable stop generation, and stop latch still match. Bootstrap and
operator-repositioned captures continue to establish their own stop and settled window;
any changed stop generation blocks capture.

This removes no IK, collision, occupancy UNKNOWN, tracking, endpoint, approval, or camera
stationarity gate. It changes no reviewed numeric configuration or motion-envelope hash,
so the eiai `_003` acceptance remains the current binding. The operator console now names
the complete approved-cycle scope and reports its live runner phase every five seconds;
typed execution blockers are preserved instead of being replaced by a generic outer
message.

## D028 — Re-evaluate state-dependent NBV and keep recoverable failures recoverable

Date: 2026-09-04

Status: accepted; full offline regression complete, physical verification pending

The experiment objective is not to execute a fixed first-view plan. It is to select new
blade/fin information whose IK is valid from the current stopped robot posture and whose
complete route is safe. Therefore an IK verdict computed at the first operator view cannot
remain the authority after the arm moves. Every stopped posture now creates or reuses one
immutable fin-discovery revision. Candidate pose semantics remain stable, but the chosen IK
branch and endpoint-collision result are re-evaluated. The accepting coarse generation
records the exact revision; schema-5 promotion may not change it without a new view.

Candidate-family enumeration is breadth-first across configured tilt, wrist roll, azimuth,
and optical distance. IK branches are consumed lazily in HoloRobot seed order and stop at
the first endpoint-clear solution. These changes remove false `UNREACHABLE` results and
unnecessary solves; they do not treat IK failure as gain and do not bypass endpoint or path
collision.

`MOTION_BLOCKED` means automatic motion cannot proceed with the current stopped posture and
map, not that the experiment evidence is corrupt. It is now exposed as a manual
`SAFETY_REFRESH` attention point. The operator may reposition the stopped arm, capture one
occupancy-only frame, and let the system discard/recompute the stale proposal. Corrupt
assets, failed event persistence, unconfirmed emergency stop, and invalid science evidence
remain terminal.

One planning/preflight turn has a 30-second responsiveness budget and experimental segment
execution always receives a finite watchdog. D029 supersedes the original post-return-only
planning check with a cooperative deadline and records the remaining native-call boundary.
Read-only supervision is best effort. No motion-safety threshold or declared selector family
is silently relaxed or truncated.

## D029 — Share one cooperative deadline across NBV, IK, collision and RRT

Date: 2026-09-04

Status: accepted; full offline regression complete, physical verification pending

A timeout checked only after `select_next` or path preflight returns does not limit the
operator's wait. One absolute monotonic deadline now starts at the coordinator's planning
transition and is inherited through the same-thread planning call chain. Candidate filters,
coarse and fine candidate loops, adaptive pose families, HoloRobot MDH/Pinocchio seed and
iteration loops, IK-branch endpoint checks, mesh/occupancy path samples, occupancy robot-STL
and voxel-run queries, and OMPL state-validity callbacks poll that same deadline. OMPL solve
and simplify receive the lesser of their local limit and the remaining total budget.

Deadline exhaustion is a recoverable `MOTION_BLOCKED` result. It must not be reported as
physical IK unreachability, low information gain, or a collision-free result. The complete
selector-bounded queue and every safety threshold remain unchanged.

This is cooperative cancellation, not asynchronous thread termination or a hard-real-time
guarantee. A single Pinocchio operation, FCL collision/distance call, vendor KDL call,
snapshot hash, large NumPy operation or robot-state read cannot be safely interrupted in
the middle. Expiry is observed at the next surrounding check; therefore the practical upper
bound is the configured 30 seconds plus at most one indivisible native call and its return
latency. This boundary must remain visible in papers and operator documentation.

## D030 — Resume from immutable science, and keep live planning visibility read-only

Date: 2026-09-04

Status: accepted; full offline regression complete, physical verification pending

An interrupted experimental chain may resume only after its append-only outer handoff and
coarse/fine authorities replay successfully. The new coordinator starts beyond every
persisted cycle identity. It restores no proposal, permit, source window, stop latch or
occupancy freshness. After a real stopped-state proof, one operator-positioned
`SAFETY_REFRESH` rebuilds current occupancy and planning restarts from the current joints.
Persisted blade science and the first hard ROI are retained; the safety refresh does not
count as a blade measurement.

The runtime publishes a small atomically replaced `live_planning.json` sidecar beside the
immutable timeline. It contains only already-computed candidate ranking, gate status,
available durations, blocking reasons, selected path and camera poses. The PySide viewer
uses it to show candidate rows and current/queued/active/selected camera frusta. Missing
evidence stays `UNKNOWN/PENDING`; the viewer never recomputes IK, collision or RRT and has
no command route. Sidecar/viewer failure is isolated from science and motion. A later
browser/SSE transport may reuse HoloRobot's read-only display subset, but no HoloRobot
control endpoint may be copied into the experiment observer.

## D031 — The online path-safety claim is the reviewed HoloRobot sampled contract

Date: 2026-09-04

Status: accepted; supersedes remaining project-memory wording that implied an online
recursive continuous-sweep certificate

The operator explicitly required the active motion path to reuse HoloRobot's single-arm
logic instead of BiBladeFusion's independently built online continuous-certificate stack.
The scientific requirement remains a safe complete route, but the implementation and paper
claim must be precise: endpoints are checked with original URDF/STL geometry, the complete
straight or RRTConnect waypoint path is sampled at the reviewed joint-step contract, every
sample is checked against original robot meshes and bound conservative occupancy, and a
non-clear sample vetoes motion. RRTConnect is used only for an interior straight-path block.

The recursive interval/swept-volume implementation remains available for offline diagnosis
and acceptance studies. It is not executed by the normal online NBV loop and must not be
used to describe online evidence as a formal continuous certificate. This wording change
does not relax UNKNOWN, clearance, endpoint, approval, tracking or stop requirements.

## D032 — Apply the complete goal-state veto before path work and preserve recoverable blocks

Date: 2026-09-04

Status: accepted; full offline regression complete, physical verification pending

The eiai run `planning-test-20260904-190447` showed that a candidate whose goal occupancy
was already UNKNOWN could spend several seconds on a complete mesh-path scan before that
goal veto was reported. HoloRobot's motion cell applies its combined goal-state collision
gate before entering the straight or fallback planner. BiBladeFusion separates original-
mesh and occupancy backends, so the equivalent order is now: goal mesh, ServoJ duration for
the required occupancy freshness horizon, goal-first occupancy path, then complete mesh
path. Occupancy is evaluated goal-first and remains bound/hash-checked once at each path
transaction boundary. A clear result still requires both complete sampled paths; a blocker
only stops work that cannot change the hard-veto result.

Online occupancy queries also stop inside the first blocking robot STL at the first exact
voxel distance that is within the unchanged margin. That mode records `query_complete=false`
because its counts are intentionally diagnostic prefixes. Standalone pose diagnostics keep
the exhaustive default and record `query_complete=true`. Both modes use the original URDF/STL
geometry, the exact voxel/run boxes, and conservative UNKNOWN-as-blocked semantics.

The same run revealed that `SupervisedExperimentRunner.step()` converted the coordinator's
recoverable planning-deadline `MOTION_BLOCKED` into a terminal runner block. The outer runtime
then called stop and persisted `ABORTED`. A coordinator-owned `MOTION_BLOCKED` now passes
through as `NEEDS_CAPTURE`, leaving coarse/fine runtime ownership intact so the operator can
perform the existing occupancy-only `SAFETY_REFRESH`. Only this explicitly recoverable phase
is preserved; other exceptions and evidence-integrity failures retain the terminal path.

This decision changes evaluation order and state propagation, not safety thresholds,
candidate ranking, the 30-second cooperative deadline, RRT conditions, approval authority,
ServoJ tracking, stop behavior, or immutable evidence requirements.

## D033 — Cache immutable occupancy query structure and short-circuit path gates by veto

Date: 2026-09-04

Status: accepted in code and offline replay; physical verification pending

The eiai run `planning-fix-d032-20260904-193942` proved that D032 made planning expiry
recoverable but did not provide enough timing margin. The first safe candidate was rank 11;
reconstructing the same accepted-static-free voxel layout and identical HPP-FCL voxel-run
boxes at hundreds of adjacent path samples consumed the remaining budget. Increasing the
deadline, reducing sample density, shrinking the UNKNOWN region or skipping a collision gate
would weaken or obscure the existing contract and was rejected.

The occupancy checker may now reuse an integer-AABB classification/run layout and immutable
voxel-run `Box`/`Transform3f` only while the recomputed snapshot content hash is unchanged.
The cache does not store a robot-pose collision answer. Each pose still places the original
URDF/STL geometry and executes exact HPP-FCL distance queries with the same clearance,
uncertainty and UNKNOWN-as-blocked rules. Snapshot changes invalidate the structural caches.

For online straight-path preflight the goal mesh remains the first veto, after which the
complete occupancy path is evaluated before the complete mesh path. This intentionally
differs from HoloRobot's per-waypoint mesh-then-environment order solely as a fail-fast
optimization for the bootstrap occupancy distribution. A candidate is CLEAR only after both
complete sampled gates pass, so their conjunction and sample density are unchanged. An
occupancy-blocked path cannot become executable through later mesh work. OMPL state validity
retains HoloRobot's mesh-then-occupancy order, and any returned detour is fully rechecked.

Exact-attempt diagnostic replay found the same first CLEAR rank-11 candidate in `15.857475 s`
of preflight; with the recorded `9.284594 s` selector time the planning transaction was
`25.142069 s`, below the unchanged 30-second cooperative budget. This authorizes one new
guarded physical attempt only; it does not itself authorize motion or replace CUDA semantic
verification.

## D034 — Use the validated Elite-A cycle and a true rest-to-rest ServoJ profile

Date: 2026-09-04

Status: code-verified; replacement motion-envelope acceptance required

The first approved physical NBV in `planning-fix-d033-20260904-201710` passed planning,
occupancy, collision, exact approval and permit consumption, but stopped on
`tracking_error_exceeded`. A post-failure read-only RTSI sample found the arm stationary and
safe at a point only about `0.187 s` along the stored 4 ms command path. The software had
already advanced much farther before the abort. The deployed configuration used a 4 ms
ServoJ command period and 0.1 s lookahead, whereas the same Elite-A controller at
`192.168.6.60` has HoloRobot physical evidence for an 8 ms period, 0.03 s lookahead, 0.2 s
warmup and 0.05 stream scaling. The `_003` acceptance trials used only a clipped 0.02 rad
command followed by a long endpoint hold, so they did not validate sustained large-travel
tracking and masked this mismatch.

The eiai deployment therefore moves to the HoloRobot-validated `8 ms / 0.03 s / 0.05`
settings. The `0.05` value applies both to the Elite Dashboard speed scaling used by the
driver and to trajectory time parameterization; leaving the robot-level default at `0.3`
would commission and execute a different controller contract. In addition, the time
parameterizer no longer merely lengthens a linearly sampled
segment. Each non-collinear joint-path segment now uses a synchronized triangular or
trapezoidal scalar profile, starts and ends at zero velocity, and respects the same scaled
per-joint velocity and acceleration limits. A path corner is a rest point. This deliberately
corrects HoloRobot's historical duration-only acceleration approximation while retaining its
driver lifecycle, warmup, tracking abort and endpoint-hold semantics.

The tracking threshold remains `0.03 rad`; no collision, UNKNOWN, workspace, IK or approval
gate changes. Execution now records the command count, last command index, maximum tracking
error, threshold and final command/feedback sample in the immutable failure event. Because
the ServoJ period, lookahead and stream scaling are part of the motion-control hash,
`es68_d435i_motion_envelope_003` remains immutable history and must not be rebound. The local
acceptance path/ID are cleared until a new bounded commissioning sequence measures the
replacement contract.

## D035 — Keep EliteDriver servers outside the host ephemeral-port range

Date: 2026-09-04

Status: deployed configuration corrected; physical retry pending

The first D034 nominal commissioning execution did not connect to or move the robot.
EliteDriver creation failed because `xray` held an established loopback connection whose
local source port was `50002`. The host ephemeral range is `32768–60999`, so the historical
HoloRobot ports `50001–50004` can be allocated to unrelated outgoing connections even when
no process is listening on them. Checking only listening sockets is therefore insufficient.

The eiai deployment moves all four SDK-configurable local EliteDriver endpoints to the free,
non-ephemeral range `29001–29004`, preserving their existing role order. The SDK injects the
configured endpoints into its external-control script, so this is a transport binding change,
not a ServoJ timing, gain, tracking, stop, path, collision or scientific-policy change. It is
a deliberate local divergence from HoloRobot's historical port numbers because this host runs
the long-lived proxy. The failed trial output remains immutable, and every retry uses a new
output-bound approval token.

## D036 — Separate Dashboard scaling from trajectory scaling and settle before writeIdle

Date: 2026-09-04

Status: forward physical trial passed; reverse, fault trial and acceptance pending

The D034 forward retry connected through the D035 ports and maintained the requested 8 ms
host cadence without an overrun. Its ServoJ stream remained below the `0.03 rad` tracking
abort threshold, but the arm was still moving at up to `0.009050 rad/s` after 90 trajectory
commands plus a fixed one-second endpoint tail. Calling `writeIdle` at that point allowed
the observed state to cross the requested endpoint and stop `0.013553 rad` away. The strict
post-stop goal window rejected the trial as designed.

The best-supported cause is a configuration-semantic mismatch in D034. HoloRobot's physical Elite-A
registry sets the controller Dashboard `default_speed_scaling` to `1.0`; the `0.05` recorded
in its execution artifact is the independent trajectory time-parameterization scaling.
Applying `0.05` to both slowed controller consumption relative to the already slowed host
stream and left queued endpoint progression at the control boundary. D036 sets only the
Dashboard value back to `1.0`; the rest-to-rest trajectory scaling remains `0.05`, and the
8 ms ServoJ period, 0.03 s lookahead and all tracking/collision limits remain unchanged.

Commissioning also removes its special fixed-duration endpoint commands. After the exact
preflighted stream succeeds, it now calls the same capability-gated, feedback-driven
`_guarded_settle_servoj_endpoint` used by production execution. That loop repeatedly sends
only the already approved endpoint and requires consecutive in-tolerance feedback before
`writeIdle`. The separate post-stop commissioning gate still requires at most `0.002 rad`
goal error and one continuous second below both joint/TCP velocity thresholds. No endpoint,
tracking, stop or safety threshold was relaxed.

Changing Dashboard scaling produces motion-control hash
`3a85600d873cd05eb7738a96a832c0a216c08d5da03b851f284b1aa10016db30`.
Because the failed trial stopped outside both sealed endpoints, neither direction of the old
candidate is eligible for reuse. Recovery starts with a new stopped capture, occupancy and
candidate; old candidate and trial artifacts remain immutable evidence.

The first D036 forward trial then passed with 90 commands, `0.003168 rad` maximum stream
tracking error, a feedback-settled endpoint error of `0.00003185 rad`, and maximum post-idle
stop drift of `0.00006060 rad`. A continuous `1.009667 s` stationary window ended at
`0.00003223 rad` goal error with zero joint/TCP speed. This is physical evidence for the
corrected one-way control boundary, not a production authorization; reverse and intentional
tracking-fault trials remain required before recording a replacement acceptance.

## D037 — Rebind a commissioning segment to its tolerated live start without exceeding its bound

Date: 2026-09-04

Status: reverse physical trial passed; intentional-fault trial and acceptance pending

The first reverse attempt after the D036 forward PASS was rejected before streaming. Its
live start differed from the sealed reverse start by only `0.00003223 rad`, but J4 was on
the adverse side of an already exact `0.02 rad` sealed segment. The live-to-sealed-goal
distance was consequently `0.02003223 rad`. Allowing `0.001 rad` start variation while
requiring an unchanged maximum live-to-goal distance of `0.02 rad` made the commissioning
contract internally inconsistent.

Increasing the trial bound, consuming the stale sealed stream, or ignoring the residual was
rejected. D037 instead computes a deterministic goal from the measured, stationary live
start toward the sealed goal and clips that direction to the smaller of the candidate's
sealed maximum and the global `0.02 rad` limit. Intentional tracking-fault trials retain
their separate `0.01 rad` limit. A `1e-12` numerical comparison avoids classifying the stored
`0.020000000000000018` representation as a physical excess; it does not enlarge the motion
bound.

The rebound goal is not trusted from the old proof. The full rest-to-rest time
parameterization and original ES68/D435i mesh continuous preflight are rerun from the exact
measured start to that goal before driver preparation. Trial evidence records both sealed
and actual goals plus live-start error, requested delta, executed delta and scale. HoloRobot
uses the same `1e-3 rad` start-tolerance concept but has no commissioning-specific maximum
physical-segment promise, so this live-bound rebind is a deliberate minimal adaptation.

Replay using the exact failed reverse state returned CLEAR with a `0.02 rad` executed delta,
90 commands, valid continuous swept evidence and `0.004175745 m` minimum certificate margin.
The failed attempt wrote no ServoJ command and did not move the robot, so its candidate may
be reused only with a new output-bound token. No threshold, collision rule, controller
setting or production authorization changed.

The D037 reverse retry
`data/acceptance/d037_20260904-211858_trial_02_reverse_retry01` then passed physically. The
live rebind converted a requested `0.02003223 rad` return into exactly `0.02000000 rad` with
scale `0.9983909949`. The 90-command stream reported `0.003236 rad` maximum tracking error,
the endpoint hold ended at `0.00002164 rad`, maximum post-idle drift was `0.00005387 rad`,
and a `1.009718 s` stationary window completed with zero joint/TCP speed. The final feedback
equals the original sealed candidate start. This validates D037 physically without granting
production motion authority.

## D038 — Make the intentional tracking-fault window reachable under rest-to-rest motion

Date: 2026-09-04

Status: physical fault trial passed; replacement acceptance recorded and bound

The original intentional-fault commissioning mode retained a `0.001 rad` tracking threshold
but truncated the rest-to-rest stream to ten commands. Both accepted nominal directions show
the same early response: maximum command/feedback error through index 9 is only
`0.00040904 rad`; the first measured samples above `0.001 rad` occur at index 15. The old
fault mode would therefore have completed its ten-command stream normally and failed its
expected-abort assertion. Repeating that unchanged on hardware would provide no new evidence.

D038 preserves the real RTSI-derived tracking error and the existing `0.001 rad` intentional
threshold; it does not inject a synthetic error or change the production `0.03 rad` guard.
Only the attended fault trial checks every command and permits at most 24 commands. For the
current `0.01 rad` continuously mesh-proven fault segment, the 64-command rest-to-rest stream
reaches only `0.00266566 rad` by command 24. Before any ServoJ write, execution now requires
the truncated command window to remain inside a reviewed `[0.002, 0.003] rad` excursion band;
the measured window excursion and all bounds are stored in `trial.json`. This prevents a
future trajectory change from silently turning the detector test into a larger move.

The HoloRobot comparison confirms that the runtime abort remains the same real-feedback
`tracking_error_exceeded` path and that `writeIdle`/stationarity semantics are unchanged.
D038 only adapts the commissioning window to the newer rest-to-rest trajectory. Focused
commissioning, EliteArm and acceptance tests report `69 passed`; the full repository reports
`1306 passed, 1 skipped in 171.72 s`. Ruff and `git diff --check` pass; the skip is the
existing CUDA-only test.

The D038 physical trial
`data/acceptance/d038_20260904-215214_trial_03_tracking_fault` passed as designed. Real RTSI
feedback crossed `0.001 rad` at command 16, producing `tracking_error_exceeded`; the outer
trial acknowledged stop in `0.00003152 s`, measured at most `0.00043768 rad` post-request
drift and completed a `1.010317 s` zero-speed stationary window. The reviewed command window
was `0.00266566 rad`; no synthetic fault or production-threshold change was used.

The replacement asset is `data/acceptance/es68_d435i_motion_envelope_004`, acceptance ID
`29624b08242d2c8ef7544cb958bf2a64335f895b719b421f792ff8c750719f9b`, metadata SHA-256
`ce807010bc11bdf50bcfb804214f85705ad155bf3c7af16d3a214a2354722766`. Its bounds retain the
larger per-axis tracking measurements from the prior representative-workspace evidence and
take the per-axis maximum stop drift across that evidence and the three replacement-contract
trials; the safety factor remains `1.5`. Unchanged collision assembly, bootstrap and emergency
stop checks are inherited rather than repeated. The local deployment now binds this exact
asset and restores production motion and stop-and-capture. No further commissioning trial is
required for this control contract.

## D039 — Reuse verified typed evidence within one planning transaction and resume D038 for a fresh safety source

Date: 2026-09-05

Status: code/offline verified; third physical occupancy source and later NBV legs pending

D038 proved one complete physical NBV motion/stop/capture leg, then exceeded the cooperative
planning deadline on its two-source occupancy prefix. The reported checkpoint near an IK or
STL-distance call was not itself the root cause. Exact profiling found repeated strict coarse
and occupancy replay before selection, followed by an actually path-expensive last candidate.

The accepted design keeps full semantic verification at the first read/publication boundary,
then carries private typed generation and occupancy storage authorities only inside the same
planning transaction. Before consumption, every declared metadata, array, snapshot and source
authority is rechecked by exact path, SHA-256 and size; NPY dtype/shape headers and stable file
identity before/after streaming are also checked. Different paths are never deduplicated merely
because their content hashes match, conflicting authorities for one path fail, and no authority
survives the transaction. This is byte-preserving reuse, not a replacement for CUDA ray replay,
FK, robot rendering or semantic verification in a new process.

Discovery may reuse endpoint IK/collision only when its policy hash, exact stopped joints and
typed generation authority match the selection transaction and an endpoint validator was
active. Any mismatch takes the old full rebind path. Exact D038 comparison preserved the
11-candidate order and TCP matrices; maximum solver-level joint difference was
`2.56e-8 rad`. The selector changed from `45.89 s` to approximately `5.5 s` without changing
science scores or vetoes.

The remaining D038 deadline is a real bounded-completeness outcome, not justification for a
larger timeout. Candidates 1-10 are rejected by UNKNOWN occupancy. Candidate 11's straight and
RRT occupancy checks are CLEAR, but both sampled original-mesh paths end BLOCKED by forearm to
D435i self-clearance. The full queue has no executable segment on the saved two-source map.
Three-dimensional cuboid merging and a Python AABB clear-only prefilter were tested against the
exact snapshot and rejected because they provided no net speedup; neither remains in production
code.

A cooperative planning deadline now persists `planning_restart_required` and terminates the
run cleanly. It is not relabelled as `SAFETY_REFRESH`, and the console must not invite repeated
`c front/back` input. Experimental resume revalidates the append-only chain, drops all old
motion authority and asks for one fresh occupancy-only source while the stopped pose and entire
physical placement remain unchanged. The robot may be recovered from whatever stopped pose is
measured at resume, but it must not be moved between that verification and the prompted
capture. That third independent source is the next evidence needed to make the map `MAP_READY`
and replan; it is not fabricated from an old stereo frame and does not count as a science view.
