# BiBladeFusion project memory

Status: authoritative research contract

Last reviewed: 2026-09-04

## 1. Research goal

Build a complete, paper-ready robotic measurement system for a thin-walled,
double-sided, finned blade. The sensing platform is an Elite Robots ES68 carrying an
Intel RealSense D435i; geometry is obtained from synchronized stereo imagery with
FoundationStereo and fused across actively selected views.

The paper presents one coherent reconstruction system. It does not need to be framed as
a large collection of independent algorithmic improvements. The central method and
engineering contribution is the way the robot chooses and safely reaches useful views.

The governing problem statement is:

> Starting from the measurement needs of a thin-walled, double-sided, finned blade,
> actively search for robot viewpoints that provide effective new information, admit an
> IK solution, and have a collision-safe complete robot route under the reviewed
> HoloRobot single-arm sampling contract.

Fine multi-view fusion and thermal-field mapping are downstream consumers. They must not
drive the design away from the viewpoint-planning problem.

## 2. Intended end-to-end workflow

The target workflow starts from one operator-selected initial view:

1. The stopped robot is manually placed at a safe first view that contains the visible
   blade and fins.
2. The system captures a synchronized formal stereo/robot frame.
3. The operator draws exactly one polygon labelled `blade` on the rectified left image.
4. FoundationStereo depth plus the polygon initializes a blade-only coarse proxy.
5. Full-scene valid depth, after robot self-masking, initializes the safety occupancy
   evidence independently of the blade proxy.
6. Coarse next-best-view selection ranks adaptive candidates by expected new blade/fin
   information. IK, workspace, endpoint collision, and complete-path checks under the
   reviewed HoloRobot sampled contract veto invalid candidates. One selected NBV produces
   one complete viewpoint motion and one capture; trajectory interpolation is not a reason
   to reconstruct intermediate views.
7. During the single-view prefix, motion may use only UNKNOWN voxels wholly covered by
   an immutable accepted-static-free region. This does not claim that the map is ready.
8. New stopped views are captured. From the second view onward, the blade ROI is produced
   from projection of the accepted blade proxy plus foreground/depth consistency; the
   operator does not redraw it under normal conditions.
9. After the required independent evidence exists, a map generation becomes `MAP_READY`
   and ordinary online NBV continues.
10. The system crosses sides as needed, observes both blade faces and opposing fin faces,
    and completes acquisition from its coverage ledger. Final mesh/watertight QA and
    thermal mapping are downstream results, not default motion gates.

The present implementation may still expose intermediate operator approvals. Those are
experimental control points, not the desired scientific workflow.

### 2.1 Current implementation gaps relative to the desired end state

The following are accepted commissioning boundaries, not completed research goals:

- every automatically planned segment still requires an exact operator approval token;
- online route safety uses HoloRobot's fixed-step sampled STL/occupancy checks, not the
  earlier recursive mathematical continuous-sweep certificate; the recursive proof remains
  offline diagnostic/acceptance evidence only;
- the live supervisor is currently a local PySide read-only window; the remote-friendly
  HoloRobot-style browser/SSE observer is not yet implemented;
- the complete software chain is offline-regressed, but the current revision has not yet
  completed a physical first NBV motion and automatic post-motion capture on eiai.

These gaps must remain visible in acceptance reports and the paper. They may not be silently
relabelled as finished capabilities.

## 3. Candidate viewpoint method

### 3.1 Variables, not rigid poses

Camera distance, surface incidence angle, wrist roll, target position, blade side, and
fin-side tilt are search variables. The original `+-15 degree` fin probes are useful seed
directions, not hard requirements. A candidate can move farther, nearer, or to another
incidence angle when that improves reachability or measurement value.

Opposing fin evidence constrains the sign of the view component across a fin; it does not
require a geometrically symmetric camera pair. The candidate family may add a shared
tangential bias so the arm can observe both fin faces from asymmetric but informative
poses. Runtime IK reuses the same Pinocchio/URDF kinematic model already loaded for the
ES68 collision assembly and the bounded neighboring-seed pattern proven in HoloRobot.
Every distinct IK branch is endpoint-collision checked before choosing the nearest clear
solution; the nearest mathematical solution is not privileged when it collides.
The selector's stored joint endpoint is provisional: immediately before path
preflight, every science-ranked camera pose is solved again using the latest stopped
robot posture. A solution generated from the first-view posture must never be carried
unchanged across later accepted motions.

This applies to both ordinary surface views and fin-discovery views. Camera-pose and
scientific semantics may remain stable, but every accepted stopped posture owns a new
immutable feasibility revision containing its IK branch and endpoint-collision verdict.
Budget-truncated search is not evidence that the untested pose family is unreachable.

After endpoint filtering, motion follows HoloRobot's single-arm composite order: try the
conservative straight joint route first, then invoke one tightly bounded RRTConnect search
only for a true interior path obstruction. UNKNOWN evidence and invalid endpoints are not
planning problems and must fail fast. Every detour is resampled and fully rechecked before
operator approval.

ServoJ timing follows HoloRobot's per-segment dynamic rule: scaled joint velocity and
acceleration limits both contribute a minimum duration. Collision sampling waypoints may
be dense, but redundant collinear samples are removed from the timing polyline so they do
not create artificial stop/start acceleration penalties. Path corners remain intact.

There is no scientific reason to prescribe one narrow standoff interval when the actual
requirements are stereo depth validity, sufficient projected support, collision-free
geometry, and a solvable robot posture.

### 3.2 Positive scientific gain

The ranking score should represent expected new measurement information, including:

- visible proxy surface whose confidence or coverage is still insufficient;
- a completely unseen or weakly observed blade side;
- exposure of fin side faces and completion of opposing fin-face evidence;
- projected blade area and stereo matchability;
- incidence quality without treating one angle as mandatory;
- diversity relative to already accepted camera poses.

Existing code expresses the coarse idea as measurement quality multiplied by a weighted
combination of surface, side, and fin deficits. The exact weights are experimental
parameters and must be recorded with results.

IK feasibility, joint limits, workspace membership, robot/environment collision,
continuous swept-path proof, and evidence integrity are hard gates. They must not add
positive information gain. Among science-ranked candidates, an invalid candidate may be
skipped in favor of the next ranked candidate.

### 3.3 Online NBV role

Online NBV is sufficient as the organizing strategy. It does not need a complete object
model in advance: the proxy, coverage ledger, side/fin evidence, and newly fused views
provide the evolving state. The contribution is the blade-specific gain and the coupling
to robot feasibility and safe path execution, not the mere use of the term NBV.

## 4. Three distinct geometry products

These products must never be conflated:

| Product | Input | Purpose |
|---|---|---|
| Robot collision geometry | ES68/D435i URDF and original collision STL meshes | Self-collision and robot shape used in environment/path checking |
| Safety occupancy geometry | Full-scene valid depth after robot self-mask | Table, fixture, blade, and any other observed external obstacle |
| Blade science geometry | First hard ROI or later projected/verified ROI, intersected with the blade envelope | Proxy, coverage, NBV gain, fusion, and reconstruction |

Transformed AABBs may accelerate broad-phase collision queries, but final decisions use
the original mesh geometry against relevant occupancy voxel boxes. A single large link
bounding sphere is not an acceptable final collision model.

## 5. Workcell assumptions fixed for the current experiment

The current accepted outer workspace in the robot base frame is:

```text
x: [-1.00, 1.00] m
y: [-0.55, 0.55] m
z: [ 0.00, 1.10] m
```

The physical agreement for this experiment is:

- the robot base `z=0` is slightly above the table surface;
- error at this boundary is intentionally ignored for this experiment;
- apart from the table, blade, and fixture, the stated workspace is available for robot
  motion;
- the blade and fixture stay fixed within one `placement_id`;
- the safety occupancy map still integrates the full observed scene;
- accepted-static-free volumes only resolve self-occluded UNKNOWN space outside the
  target envelope; any OCCUPIED observation still wins.

If the blade, fixture, camera mount, robot base, or relevant workcell geometry moves, the
old placement-dependent map/proxy evidence cannot be reused.

## 6. ROI policy

The first formal view uses one operator polygon on the rectified left image. It should
contain the complete visible blade and fins while excluding fixture and table as far as
practical.

For subsequent views:

1. transform the accepted proxy into the candidate camera frame;
2. project visible proxy points into the rectified image;
3. dilate the projected support to tolerate calibration and coarse-proxy error;
4. require foreground/depth consistency with the projected reference;
5. intersect accepted pixels with the blade base-frame envelope;
6. reject rather than silently choose an unrelated mask when projection support or match
   fraction is insufficient.

An automatic mask is evidence derived from the initial blade identity; it is not generic
largest-component selection.

## 7. Experimental safety philosophy

This is a supervised research experiment, not an initial production release. Essential
physical protections remain mandatory: exact robot geometry, collision checking,
complete route validation under the reviewed online contract, controller state, stop
behavior, immutable map identity, and operator access to the physical emergency stop.

At the same time, a safety gate must correspond to a concrete physical or evidence
invariant. Camera/robot timestamp jitter, small stopped-pose variation, computation time,
or a repeated internal validation step must not block merely because an arbitrary tight
threshold was chosen. Time limits used to detect a real hang must cover measured normal
execution with margin and must not invalidate a static map solely because computation
took longer.

One online NBV turn shares one absolute cooperative planning deadline across candidate
generation, IK, endpoint checks, straight-path validation, occupancy queries and bounded
RRT. Deadline expiry is a recoverable motion block, not an IK/collision/science verdict.
It is not a hard-real-time guarantee: a single native Pinocchio/FCL/KDL/OMPL operation is
observed at its next safe return boundary rather than being killed asynchronously.

The strict robot-stationarity evidence window surrounds camera exposure only. The
FoundationStereo forward pass, occupancy ray integration, and artifact persistence happen
after that window and cannot retroactively invalidate an already stopped image. Production
sampling shares EliteArm's persistent, serialized RTSI state source; it must not open a new
robot connection for every perception cycle.

## 8. Paper narrative

The paper should be organized around the complete system:

- one-view operator bootstrap;
- blade-specific evolving proxy and coverage representation;
- adaptive, information-seeking double-sided/fin viewpoint generation;
- IK-aware and path-safe robot realization;
- multi-view geometry reconstruction and quantitative evaluation;
- optional thermal mapping as a downstream extension.

Evaluation should demonstrate that the system measures the intended blade regions and
runs reliably. Large method-by-method ablation campaigns are optional, not the default
scope. Focus limited experimental effort on end-to-end success, coverage, geometric
accuracy, number of views, planning/runtime cost, and representative failure analysis.

## 9. HoloRobot is the control-stack reference

The operator developed and owns HoloRobot, available at `~/Documents/HoloRobot`. For the
parts of this project that are general robot-system infrastructure rather than
blade-specific science, HoloRobot is the first implementation reference.

Before changing BiBladeFusion, inspect HoloRobot for existing implementations of:

- Elite robot connection and lifetime management;
- continuous robot-state and visual-feedback acquisition;
- ServoJ command generation, streaming, enable, stop, and recovery;
- controller/runtime-state interpretation and stationarity detection;
- camera/robot synchronization and coordinate transforms;
- occupancy-map construction, ray integration, and update policy;
- collision and motion-planning integration where present.

When an applicable, proven implementation exists, transplant it into BiBladeFusion or
adapt it with the smallest necessary interface changes. Do not independently rebuild the
same mechanism and then repair it one symptom at a time. New design work should focus on
requirements HoloRobot does not already solve: thin bilateral finned-blade representation,
blade identity propagation, active information gain, adaptive candidate search, and the
coupling between these scientific objectives and robot feasibility.

A complete regression must therefore begin with a module-by-module comparison against
HoloRobot, followed by an end-to-end timing and state-transition audit. The goal is not
only correctness: the first-view-to-first-segment latency must be measured by phase, and
duplicated reconstruction, IK, collision proof, connection setup, or state validation
must be removed or cached where its bound inputs have not changed.

Experiment visualization follows the same reuse rule. Prefer HoloRobot's browser/SSE and
Three.js observation patterns for remote-friendly read-only display, but never transplant
its command routes into the measurement supervisor. Visualization is diagnostic: failure
must not stop science or motion, and missing evidence must stay unknown rather than being
inferred.

## 10. Related authoritative project documents

- `docs/CURRENT_STATE.md`: latest operational checkpoint and unresolved blocker.
- `docs/EXPERIMENT_RUNBOOK.md`: commands and evidence collection for eiai runs.
- `docs/DECISION_LOG.md`: dated decisions and rationale.
- `docs/eiai-five-repairs-tomorrow-2026-09-03.md`: detailed five-repair design record.
- `docs/adaptive-ik-view-search.md`: implemented adaptive search behavior.
- `docs/coverage-next-view-selector.md`: fine coverage and NBV semantics.
- `docs/occupancy-motion-safety.md`: formal occupancy/path-safety contract.
