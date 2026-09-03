# BiBladeFusion current state

Checkpoint date: 2026-09-03

Authoritative branch for this work: `main`

Local main worktree: `/home/vale/Documents/Proj1/biblade-fusion-main`

eiai deployment checkout: `/home/eiai/Documents/wh/BiBladeFusion`

This file is an operational checkpoint, not a claim that the unresolved hardware flow is
working. Update it after every material hardware result or fix.

## 1. Repository checkpoint

At the time of this checkpoint, local `main` and `origin/main` both point to:

```text
aa279b1 perf: reuse robust proof inside motion envelope
```

The HoloRobot-aligned control/IK regression described in
`docs/HOLOROBOT_REGRESSION_2026-09-03.md` is committed as:

```text
47fc472 fix: align ES68 motion lifecycle with HoloRobot
```

`aa279b1` is its parent and the deployment's minimum starting checkpoint.

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
maximum ranked preflight candidates: 3
single-view bootstrap motion: enabled
projected ROI: dilation 12 px, minimum 100 reference points,
               minimum 500 reference pixels, minimum match fraction 0.50
occupancy ray integration: deterministic CUDA DDA
```

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

## 5. Latest unresolved hardware blocker

### 5.1 Newest run reported at 2026-09-03 22:25–22:27 local time

The newest physical run again reached:

```text
coarse_scan/bootstrap_motion_ready
  -> coarse_scan/waiting_approval
  -> exact EXECUTE token accepted
```

SCHED_FIFO priority 99 succeeded. The Elite driver then opened its three reverse TCP
ports, closed them after about one second, reopened all three ports, and closed them again
less than one second later. Approximately 19 seconds later the run terminated with:

```text
external_execution_failed:
StationarityTimeoutError: stationarity timed out before the next robot-state sample
```

The reported total service runtime was `6 min 47.767 s`, with `8 min 35.841 s` CPU time.
Whether the arm physically moved during this attempt was not reported and must not be
inferred from the connection log.

The repeated KDL `-5` messages preceded a successfully prepared segment and are therefore
individual candidate IK failures, not this run's terminal cause. They should eventually
be summarized rather than flooding the operator log.

### 5.2 Closely related preceding failures

The preceding physical run failed after the exact approval token was entered:

```text
external_execution_failed:
RobotCommandError: motion execution permit expired during post-recovery revalidation
```

The same run's cleanup then failed to obtain a new robot-state sample:

```text
stop_failed:
StationarityTimeoutError: stationarity timed out before the next robot-state sample
```

In the preceding run, the closely related failure was:

```text
motion execution permit expired during guarded enable
```

followed by cleanup observing `robot_mode=RUNNING`, `runtime_state=PLAYING` instead of a
confirmed stopped state.

Taken together, these failures indicated a control-stack lifecycle problem rather than a
candidate-generation failure. The HoloRobot comparison found that an unused reverse
session was started during guarded enable while the stop latch remained set and then
timed out before resume, explaining the double port sequence. It also found that
BiBladeFusion stopped the controller task before confirming final ServoJ convergence.

Both paths now have offline-tested fixes: reverse control begins once at approved resume,
and the final approved setpoint is held and checked against persistent RTSI feedback
before the Dashboard stop. The physical blocker is therefore **code-fixed but not yet
hardware-verified**.

## 6. Required next action

The required offline work is complete:

1. HoloRobot commit `93216a428cb8004382e9e39e5da7cd7bc6cbfffd` was reviewed for Elite
   lifecycle, feedback, ServoJ, IK, collision, and occupancy behavior.
2. The reverse-port double connection was traced and removed from the guarded transition.
3. HoloRobot endpoint settling was adapted behind the existing guarded capability.
4. HoloRobot's MDH/DLS multi-seed IK became the default online reachability path.
5. Stationarity timeout messages now expose last controller/goal/velocity state.
6. Existing bound robust mesh/occupancy proofs remain reused only under their hashes.
7. Historical timing artifacts established that manual annotation and CPU DDA—not
   FoundationStereo or IK—dominated the old first cycle. eiai already selects CUDA DDA.
8. The focused and full offline regression results passed as recorded below.

The next action is exactly one fresh eiai physical validation using the runbook and a new
`run_id`. Do not change placement, acceptance assets, or safety geometry before this
test. After the approval token, verify one reverse connection, one segment, endpoint
settle, controller stop, and transition to the next capture.

Current status is:

```text
offline/code regression: complete
full physical single-view-to-motion workflow: not yet re-verified
latest physical blocker: code-fixed; hardware acceptance pending
```

## 7. Offline regression result

Using the main source tree with the available local test environment:

```text
full suite: 1209 passed, 3 skipped
```

The skipped tests require optional local PyTorch/Open3D packages absent from this vale
test environment. CUDA availability and the FoundationStereo stack must still pass
`scan doctor` on eiai before the physical run. Targeted results and the subsystem audit
are in `docs/HOLOROBOT_REGRESSION_2026-09-03.md`.

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
