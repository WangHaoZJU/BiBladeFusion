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
  contract. Offline continuous-proof capability may still appear in JSON details, but it
  is not executed by the online NBV loop.
- xFormers and FlashAttention warnings are optional acceleration warnings.
- `Motion authorized: no; hardware acceptance is a separate gate` is an informational
  release statement in experimental mode; it is not itself the runtime failure.

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
  -> "Planning next view with HoloRobot single-arm sampling ..."
  -> at most 3 science-ranked candidates on the current eiai configuration
  -> fixed joint-step interpolation, 5 samples per segment, first collision exits
  -> waiting_approval
  -> exact EXECUTE token entered
  -> permit consumed, then permit-bound power/brake preparation
  -> one reverse-control session opened at approved resume
  -> one ServoJ segment
  -> final approved endpoint held until RTSI joint feedback converges
  -> HoloRobot-compatible writeIdle stop latch and sampled stationary confirmation
  -> next capture
```

The planning line must be followed by either `waiting_approval` or a typed candidate
rejection; it no longer starts a recursive six-dimensional certificate. The active checks
still use original URDF/STL robot geometry and conservative occupancy. `sampled` does not
mean collision checking is disabled.

Paste the entire exact token, including `EXECUTE`, once. After Enter, a short preparation
interval is expected while the driver connects and controller state is established. The
full already-bound path is not proved again; only a changed live-start bridge is checked in
the same HoloRobot sampled mode. Permit lifetime is checked when the token is consumed;
elapsed enable/recovery time cannot retroactively expire an already consumed permit. A
separately configured measured segment-duration watchdog may still stop a genuinely
overlong move.

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

## 6. Read-only live supervision

In another terminal, use the exact output root printed by the active run:

```bash
/usr/bin/env -u PYTHONPATH .venv/bin/bbf \
  supervise replay \
  --snapshot /absolute/run/output/live_timeline \
  --follow
```

This is read-only and must not send robot commands.

## 7. Evidence to collect after any failure

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

## 8. Stop rules

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
