# eiai unknown-blade experiment runbook

Last reviewed: 2026-09-04

Scope: supervised experimental single-view bootstrap through active planning

Workcell: ES68 + D435i, placement-dependent blade/fixture geometry

This runbook records operator commands and evidence collection. The complete 2026-09-04
control/mapping/planning regression is offline-verified but still requires one physical
validation. Check
`docs/CURRENT_STATE.md` before starting a run.

## 1. Identity rules

- `placement_id` identifies one unchanged physical blade/fixture placement.
- `run_id` identifies one software attempt.
- `output` must be a new directory for every attempt.
- If the blade, fixture, camera mount, or robot base moves, create a new `placement_id` and
  do not reuse the old ROI, occupancy evidence, proxy, or planned motion.

Current unchanged placement reported by the operator:

```text
blade-placement-20260901-01
```

## 2. Pull without damaging the environment

From the eiai repository:

```bash
cd /home/eiai/Documents/wh/BiBladeFusion
git status --short --branch
git pull --ff-only origin main
git log -1 --oneline
git status --short --branch
```

Do not automatically run `uv sync --frozen` after a normal pull. It previously removed
the private Elite SDK wheel and GPU/FoundationStereo dependencies because those packages
are deployment additions. Rebuild only when dependencies actually changed, using the
project GPU bootstrap procedure and the real Elite SDK wheel.

The composite HoloRobot planner adds one optional pinned runtime wheel. Install only that
wheel without reconciling or removing the rest of the deployment environment:

```bash
/usr/bin/env -u PYTHONPATH uv pip install \
  --python .venv/bin/python \
  'ompl==2.0.1'

/usr/bin/env -u PYTHONPATH .venv/bin/python -c \
  "import ompl; print('OMPL: OK')"
```

Always keep ROS Python 3.10 packages out of this Python 3.12 environment:

```bash
/usr/bin/env -u PYTHONPATH .venv/bin/python -c "
import sys, torch, pinocchio, hppfcl
import elite_cs_sdk
print('python:', sys.executable)
print('cuda:', torch.cuda.is_available(), torch.cuda.device_count())
print('pinocchio:', pinocchio.__version__)
print('hppfcl:', hppfcl.__version__)
print('elite sdk: OK')
"
```

### 2.1 D025 motion-contract migration

After pulling D025, ensure `configs/local.yaml` contains the reviewed HoloRobot ES68
acceleration vector:

```yaml
motion_preflight:
  maximum_joint_acceleration_rad_s2: [4.0, 4.0, 4.0, 4.0, 4.0, 4.0]
```

This changes the immutable motion-control contract. The old
`es68_d435i_motion_envelope_002` path/ID must not be relabelled as current. Complete the
motion-envelope acceptance procedure for this slower velocity-and-acceleration-limited
stream, record a new immutable output, and replace both
`motion_envelope_acceptance_path` and `motion_envelope_acceptance_id` in
`configs/local.yaml`. Do not start a moving run while `scan doctor` reports a contract
mismatch.

## 3. Non-moving readiness check

```bash
/usr/bin/env -u PYTHONPATH .venv/bin/bbf \
  scan doctor \
  --mode unknown \
  --experimental \
  --ray-integration-backend cuda \
  --config configs/local.yaml
```

Required before proceeding:

- Elite SDK, FoundationStereo dependencies, CUDA, CUDA DDA, collision backends,
  calibration, static-free acceptance, and coarse-to-fine policy pass.
- `supervised_scan_holorobot_single_arm` must report the fixed-step sampled online
  contract and `ompl_fallback_available: true`. Offline continuous-proof capability may
  still appear in JSON details, but it is not executed by the online NBV loop.
- xFormers and FlashAttention warnings are optional acceleration warnings.
- `supervised_scan_realsense_sdk` must pass. Kinematics and stereo calibration checks now
  parse their contents and verify the configured image size; a readable malformed file is
  no longer sufficient.
- `Motion authorized: no; hardware acceptance is a separate gate` is an informational
  release statement in experimental mode; it is not itself the runtime failure.

Confirm the active single-arm planner details explicitly:

```bash
/usr/bin/env -u PYTHONPATH .venv/bin/bbf \
  scan doctor \
  --mode unknown \
  --experimental \
  --ray-integration-backend cuda \
  --config configs/local.yaml \
  --json | jq '.[] | select(.name == "supervised_scan_holorobot_single_arm")'
```

If `unknown_blade_coarse_to_fine` fails, get the exact details instead of guessing:

```bash
/usr/bin/env -u PYTHONPATH .venv/bin/bbf \
  scan doctor \
  --mode unknown \
  --experimental \
  --ray-integration-backend cuda \
  --config configs/local.yaml \
  --json | jq '.[] | select(.name == "unknown_blade_coarse_to_fine")'
```

## 4. Start a fresh run

The current code is ready for one physical validation. Replace `<RUN_ID>` with a new
value, keep the output matched to it, and do not repeat the attempt before preserving and
reviewing any failure evidence.

### 4.1 Capture and annotate a new first view

```bash
sudo systemd-run --wait --collect --pty \
  --property=User=eiai \
  --property=WorkingDirectory=/home/eiai/Documents/wh/BiBladeFusion \
  --property=LimitRTPRIO=99 \
  /usr/bin/env -u PYTHONPATH \
  /home/eiai/Documents/wh/BiBladeFusion/.venv/bin/bbf \
  scan run-unknown \
  --experimental \
  --ray-integration-backend cuda \
  --config configs/local.yaml \
  --operator-id eiai \
  --placement-id blade-placement-20260901-01 \
  --run-id <RUN_ID> \
  --first-side front \
  --bootstrap-seed-mode hard_roi \
  --output data/experiments/blade-placement-20260901-01-<RUN_ID>
```

At the first prompt, capture only while the robot and scene are stopped. The runtime then
prints the exact rectified image and default JSON paths and waits for the annotation.

### 4.2 X-AnyLabeling command

In a second terminal, use the option required by the installed X-AnyLabeling CLI:

```bash
cd /home/eiai/Documents/wh/X-AnyLabeling
uv run python anylabeling/app.py \
  --filename /absolute/path/from/runtime/bootstrap_annotation/left_rectified.png
```

Draw exactly one polygon labelled `blade`, including visible blade and fins. Save it as
the default adjacent `left_rectified.json`. Do not move the robot, camera, blade, or
fixture between capture and save. Return to the runtime terminal and press Enter to use
the printed default JSON path.

### 4.3 Reuse an already verified first-view polygon

Only when the physical first-view relationship is unchanged:

```bash
export BBF_REUSED_ROI=/absolute/path/to/bootstrap_annotation/left_rectified.json

/usr/bin/env -u PYTHONPATH .venv/bin/python - <<'PY'
import os
from biblade_fusion.workflows.unknown_blade_runtime import _read_hard_roi_seed

seed = _read_hard_roi_seed(os.environ["BBF_REUSED_ROI"])
print("mode:", seed.mode)
print("kind:", seed.kind)
print("vertices:", len(seed.vertices_uv))
PY
```

The `--bootstrap-polygon` argument receives the path, not the JSON text:

```bash
sudo systemd-run --wait --collect --pty \
  --property=User=eiai \
  --property=WorkingDirectory=/home/eiai/Documents/wh/BiBladeFusion \
  --property=LimitRTPRIO=99 \
  /usr/bin/env -u PYTHONPATH \
  /home/eiai/Documents/wh/BiBladeFusion/.venv/bin/bbf \
  scan run-unknown \
  --experimental \
  --ray-integration-backend cuda \
  --config configs/local.yaml \
  --operator-id eiai \
  --placement-id blade-placement-20260901-01 \
  --run-id <RUN_ID> \
  --first-side front \
  --bootstrap-seed-mode hard_roi \
  --bootstrap-polygon "$BBF_REUSED_ROI" \
  --output data/experiments/blade-placement-20260901-01-<RUN_ID>
```

Environment variables defined in the interactive shell are expanded before `sudo
systemd-run` receives the command. A missing or empty value must be treated as an error.

## 5. Approval and expected motion sequence

The normal transition is:

```text
bootstrap_motion_ready/map_ready
  -> "Planning next view with HoloRobot single-arm composite planning ..."
  -> complete NBV-selector-bounded queue in unchanged science-rank order
  -> re-solve each ranked camera pose from the latest stopped joint posture
  -> every IK branch receives a fail-fast exact URDF/STL endpoint check
  -> straight 0.02-rad route first (finer than HoloRobot's effective 0.025 rad)
  -> if endpoints are clear and only the route interior is blocked, one bounded
     RRTConnect attempt, then full resampling/recheck
  -> waiting_approval
  -> exact EXECUTE token entered
  -> permit consumed, then permit-bound power/brake preparation
  -> one reverse-control session opened at approved resume
  -> one velocity-and-acceleration-limited ServoJ segment
  -> final approved endpoint held until RTSI joint feedback converges
  -> HoloRobot-compatible writeIdle stop latch and sampled stationary confirmation
  -> next capture
```

Straight and OMPL interpolation retain the exact current-state and selected-IK tuples at
their boundaries. They never recreate a boundary with floating-point arithmetic. This is
required by the waypoint/hash identity check and prevents the one-ULP endpoint failure
observed in `planning-test-20260904-154403`.

Bound online occupancy checks stop at the first blocking robot STL, matching HoloRobot's
fail-fast motion contract. A standalone diagnostic pose query may still enumerate all
blocking links. The difference is diagnostic completeness only: either result is a hard
motion veto, and no threshold or UNKNOWN policy changes.

The planning line must be followed by either `waiting_approval` or a typed candidate
rejection; it no longer starts a recursive six-dimensional certificate. The active checks
still use original URDF/STL robot geometry and conservative occupancy. `sampled` does not
mean collision checking is disabled. One path hashes/binds its immutable occupancy at the
two transaction boundaries; it does not re-hash the whole map for every robot pose. The
legacy `maximum_ranked_preflight_candidates` value is ignored: the selector has already
bounded the queue, and stopping after three can discard a safe fourth or fifth path. The
default OMPL budget is one 1.0 s solve, not HoloRobot's generic five-by-five-second
application budget. UNKNOWN, stale/mismatched map evidence, and start/goal collision fail
immediately; they are never sent to OMPL. If OMPL cannot find a detour within its budget,
the coordinator records that candidate and continues to the next ranked endpoint.

The complete NBV selection and ranked path queue shares one 30-second cooperative
responsiveness deadline. Python candidate/IK/collision loops poll it and OMPL receives only
the remaining budget. This is not asynchronous thread termination: one indivisible
Pinocchio/FCL/KDL/hash/NumPy/state-read call may finish after expiry before the next check
stops planning. If no safe automatic route exists from the current state, the console
remains in the coarse scan and prompts for an operator-positioned safety refresh instead
of terminating the experiment. Manually place the stopped arm at a clearly safe posture,
enter exactly `c`, and do not provide a side label. That frame updates occupancy only; it
does not count as a blade science view. The prior proposal is discarded and the next plan
regenerates fin IK from the new stopped joints.

After any successful automatic capture, fin discovery is re-evaluated from that exact
stopped posture and written under `coarse_science/fin_discovery_revisions/`. A candidate
that was unreachable at the initial view may therefore become usable later; the runtime
does not reuse its first-view joint solution.

Paste the entire exact token, including `EXECUTE`, once. After Enter, a short preparation
interval is expected while the driver connects and controller state is established. The
full already-bound path is not proved again. A live start outside HoloRobot's 0.001 rad
tolerance is rejected rather than bridged by an unplanned command. Permit lifetime is
checked when the token is consumed;
elapsed enable/recovery time cannot retroactively expire an already consumed permit. A
separately configured measured segment-duration watchdog may still stop a genuinely
overlong move.

The live joints must remain within HoloRobot's `0.001 rad` plan-start tolerance from the
stored preflight start. Once the approved segment succeeds, there is exactly one normal
`writeIdle(0)` and one full settled window. The automatic capture reuses that boundary only
while the expected view/purpose and stop generation/latch are unchanged; it must not print
a second reverse-port recovery or issue a second stop for the same leg. The console prints
an `Approved cycle still running` line every five seconds with the current runner phase, so
motion, settling, inference and subsequent planning are no longer one opaque wait.

Keep a hand on the physical emergency stop. Abort on unexpected physical motion, cable
pull, loss of visibility, or approach to the blade/fixture that disagrees with the
displayed plan.

The normal online path now reuses the already loaded HoloRobot Pinocchio/URDF model for
IK and should not emit repeated vendor KDL warnings. It searches bounded neighboring
joint seeds and asymmetric signed-half-plane fin poses; KDL output is expected only from
an explicitly requested historical reproduction path.

For the first post-fix run, observe the reverse-port lines after approval. There should
be one group of ports 50002/50003/50004 connecting before the stream. Normal segment
completion sends `writeIdle(0)` and deliberately does not call Dashboard `stopProgram`,
so it must not create the former close/reconnect cycle. A sampled RTSI `runtime_state`
may remain `PLAYING`; the unchanged stop generation/latch plus bounded joint/TCP samples
is the stop-and-capture authority.

Each perception cycle should reuse that existing EliteArm RTSI connection. A second
process-isolated RTSI sampler must not connect for FoundationStereo. Strict stationarity
sampling ends immediately after the synchronized camera bracket; inference, CUDA ray
integration, and artifact writes run afterward and are not part of the exposure-stability
window.

## 6. Read-only live supervision

In another terminal, use the exact output root printed by the active run:

```bash
/usr/bin/env -u PYTHONPATH .venv/bin/bbf \
  supervise replay \
  --snapshot /absolute/run/output/live_timeline \
  --follow
```

This is read-only and must not send robot commands.

The follow-mode window now includes a `候选视点与规划预检` table and camera frusta. It
shows the science-ranked queue, active/selected candidate, IK and endpoint gates,
straight-path and bounded-RRT results, known durations, and the exact blocking reason.
Grey/blue/green frusta mean queued/active/selected; the current camera is yellow. A field
without independently recorded evidence stays `UNKNOWN` or `PENDING`.

Planning progress is carried by the small atomic file
`<OUTPUT>/live_timeline/live_planning.json`; point clouds and meshes are not rewritten on
each progress event. This sidecar and the GUI are diagnostic-only and cannot approve or
execute motion. An observer failure must produce a warning but must not stop the scan.
The current viewer is still PySide, so local display on eiai is preferred over slow SSH
X11 forwarding. A browser/SSE viewer is a later read-only transport phase.

## 7. Resume an interrupted experimental chain

Use this only when the output chain is intact and the blade, fixture, camera mount and
robot base have not moved. Pass the existing experiment root; do not provide a new
`placement_id`, `run_id`, ROI or output directory:

```bash
export BBF_EXISTING_OUTPUT=/absolute/path/to/the/interrupted/experiment

sudo systemd-run --wait --collect --pty \
  --property=User=eiai \
  --property=WorkingDirectory=/home/eiai/Documents/wh/BiBladeFusion \
  --property=LimitRTPRIO=99 \
  /usr/bin/env -u PYTHONPATH \
  /home/eiai/Documents/wh/BiBladeFusion/.venv/bin/bbf \
  scan run-unknown \
  --experimental \
  --resume \
  --ray-integration-backend cuda \
  --config configs/local.yaml \
  --operator-id eiai \
  --output "$BBF_EXISTING_OUTPUT"
```

Resume validates the append-only handoff chain before hardware use. It never restores an
old proposal, approval permit, occupancy freshness, source window or in-flight segment.
After establishing a real stopped state it asks for one operator-positioned
`SAFETY_REFRESH` capture, allocates a new cycle/view identity, and replans from the current
joint posture. It does not request the first-view hard ROI again and the refresh does not
count as blade science. Do not resume when the physical placement changed; begin a fresh
placement/run instead.

## 8. Evidence to collect after any failure

Do not immediately start another run. First record:

```bash
export BBF_RUN_ROOT=/absolute/path/to/the/failed/run

find "$BBF_RUN_ROOT/live_timeline" -mindepth 2 -maxdepth 2 \
  -name snapshot.json -print | sort | tail -n 5

find "$BBF_RUN_ROOT" -type f \
  \( -name '*.json' -o -name '*.jsonl' -o -name '*.log' \) \
  -print | sort | tail -n 80

git log -1 --oneline
git status --short --branch
```

Preserve the exact terminal lines from the last successful phase through the cleanup
result. Also record:

- `placement_id`, `run_id`, and output root;
- exact approval token time and first driver connection time;
- `robot_mode`, `runtime_state`, and safety mode before approval and at failure;
- latest occupancy sequence and hashes;
- snapshot/event file paths;
- whether the arm moved at all.

An ordinary stationarity timeout on the post-fix code includes `robot_mode`,
`runtime_state`, `controller_stopped`, `goal_error`, `stable_samples`, and available
joint/TCP speeds. Preserve that entire line; it replaces the earlier ambiguous timeout
message.

For permit/recovery failures, do not infer that planning failed. Separate:

```text
candidate/IK -> path preflight -> approval -> permit -> driver recovery/enable
             -> post-recovery validation -> stream -> stop/stationarity -> capture
```

## 9. Stop rules

Stop and diagnose before another physical attempt when any of these occurs:

- configuration or immutable acceptance hash mismatch;
- collision checker is not fully hash-bound;
- occupancy lifecycle is inconsistent with the documented bootstrap prefix;
- a permit is already expired before it is consumed;
- the stop generation/latch changes during the stationary window;
- stationarity cannot obtain fresh RTSI samples;
- robot, blade, fixture, or camera moved relative to reused evidence;
- the expected path or camera pose is inconsistent with the physical scene.

One failed hardware run should produce a code-level hypothesis and an offline test before
the next run.
