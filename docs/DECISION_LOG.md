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
before lower-value tilts and interleaves wrist rolls. No IK, workspace, collision, or
continuous-path gate is relaxed.

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
certificate. Guarded execution accepts either explicit contract and revalidates only a new
live-start bridge with the same mode bound into the permit.

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
