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

### 2.3 Accepted motion envelope

```text
path: data/acceptance/es68_d435i_motion_envelope_002
id:   39b675eca06390f8b99a1a18b0b3743e084c42df2c821ed67d0901d4c231a240
metadata SHA-256: 8655a35a4c450be96ae563e3b832063a8a7468b62aab087aee6c6c887580af2f
```

The eiai configuration parser confirmed both path and ID. `scan doctor --mode unknown
--experimental --ray-integration-backend cuda` subsequently passed
`unknown_blade_coarse_to_fine`; optional xFormers and FlashAttention remained warnings.

### 2.4 Runtime configuration facts last confirmed

```text
adaptive IK search: enabled
legacy maximum ranked preflight candidates: parsed but ignored
legacy maximum segment joint delta: parsed but ignored
single-view bootstrap motion: enabled
projected ROI: dilation 12 px, minimum 100 reference points,
               minimum 500 reference pixels, minimum match fraction 0.50
occupancy ray integration: deterministic CUDA DDA
```

The online contract now treats the declared workspace as a hard outer boundary, searches
at most 32 IK poses / 1.5 s per candidate family, preflights the complete bounded science
queue, and moves directly to one selected viewpoint. `motion_preflight.maximum_joint_step_rad`
is only an internal interpolation/collision-proof interval.

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

The last physical attempt (2026-09-03 22:25–22:27 local time) reached exact approval but
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
5. online IK uses a bounded analytic-MDH multi-seed solver instead of the noisy KDL probe;
6. one NBV is one complete viewpoint path and one capture; the legacy 0.02 rad setting no
   longer triggers intermediate reconstruction cycles;
7. all bounded science-ranked endpoints receive path preflight, so an unsafe top result
   does not hide a safe lower-ranked view;
8. incremental occupancy reuses an unchanged verified prefix and the live writer does not
   immediately replay identical rays. Sliding/replacement windows and cross-process reads
   remain strict full rebuild/replay;
9. automatic ROI transfer uses the projected proxy's per-pixel depth band plus the blade
   envelope, and fine NBV receives bounded adaptive distance/incidence fallback;
10. measurement completion is based on acquired coverage; downstream mesh/watertight QA
    is nonblocking by default and remains available as a strict opt-in;
11. synchronized vale/eiai assets with stale absolute roots relocate only after exact
    content-hash verification.

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

It selected independently feasible front/back adaptive views. That closes the known
planning-time regression. Current path planning proves straight joint-space motion and
tries alternative science endpoints; it does not yet synthesize a curved RRT/OMPL route
around an obstacle to the same endpoint. HoloRobot's optional OMPL implementation was
inspected, but direct transplantation would not preserve BiBladeFusion's continuous
uncertainty-bound per-leg evidence and OMPL is absent from the validated dependency set.

One offline replay of an old occupancy artifact took about 64 s and correctly failed
because the artifact was rendered with Open3D while the current local environment selected
the NumPy renderer. This is a cross-process historical replay, not the new live cached
path. Renderer identity remains a deliberate hard evidence boundary.

## 7. Required next action and regression result

The next action is exactly one fresh eiai physical validation using the runbook and a new
`run_id`. Do not change placement, acceptance assets, or safety geometry before this test.
After the approval token, verify one reverse connection, one complete viewpoint motion,
endpoint settle, `writeIdle`, sampled stationary pose, and transition to the next capture.

Current status is:

```text
offline/code regression: complete
real-data view planning: complete (0.78 s)
full physical single-view-to-motion workflow: hardware verification pending
```

Using the main source tree with the available local test environment:

```text
full suite: 1219 passed, 3 skipped in 98.71 s
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
