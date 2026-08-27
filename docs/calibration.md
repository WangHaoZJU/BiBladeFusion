# Calibration and frame conventions

BiBladeFusion uses `parent_T_child`: a transform maps coordinates expressed in the child
frame into the parent frame. Distances are metres and angles are radians unless a field
name explicitly says otherwise.

## Eye-in-hand calibration

The authoritative result is `flange_T_left_ir`, not an RGB-camera transform and not its
inverse. The camera frame is the **raw D435i left infrared imager** (`infrared/1`). Its
intrinsics come only from the user-generated ChArUco stereo calibration; factory IR
intrinsics and extrinsics are not solver inputs.

The robot pose comes from the HoloRobot 709-pose calibrated ES68 FK:

```text
base_T_target(i)
  = base_T_flange(q_controller + joint_zero_offsets)
  · flange_T_left_ir
  · left_ir_T_target(i)
```

The fixed board makes `base_T_target(i)` constant. RTSI `base_T_tcp` is recorded only
for the independent check
`base_T_flange · flange_T_tcp ≈ base_T_tcp`; it does not enter the hand-eye solve.

Store the measured artifact outside version control and reference it through
`hand_eye.calibration_path`. The flange-primary YAML schema is:

```yaml
schema_version: 2
calibration_type: es68_d435i_left_ir_eye_in_hand
robot_model: es68
camera_stream: infrared/1
parent_frame: flange
child_frame: left_ir
method: OpenCV Daniilidis + LM bundle adjustment
matrix:
  - [r00, r01, r02, tx]
  - [r10, r11, r12, ty]
  - [r20, r21, r22, tz]
  - [0.0, 0.0, 0.0, 1.0]
quality:
  sample_count: <integer>
  translation_rmse_m: <metres>
  rotation_rmse_deg: <degrees>
  rotation_span_deg: <degrees>
  translation_span_m: <metres>
  rotation_axis_diversity: <ratio from 0 to 1>
bundle_adjustment:
  initial_rmse_px: <pixels>
  final_rmse_px: <pixels>
```

The artifact also retains the independently fitted HoloRobot `flange_T_tcp`, a derived
runtime-compatible `tcp_T_left_ir`, input file hashes, the exact user-calibrated left-IR
intrinsics, and fixed-board closure metrics. The loader quality-gates all required
metrics. Schema 1 TCP artifacts remain readable for old experiment records but are not
produced by the new workflow.

### Live capture

First generate the raw IR stereo YAML with `bbf calibration stereo-gui`. A successful
solve automatically publishes the verified result to
`data/calibrations/d435i_ir_active.yaml`, which the default configuration already uses.
Only machine-specific robot addresses need to be set in a Git-ignored local config:

```yaml
robot:
  model: es68
  robot_ip: <ES68 address>
realsense:
  serial_number: <D435i serial>
  stereo_calibration_path: data/calibrations/d435i_ir_active.yaml
  infrared_emitter_enabled: false
```

Start the read-only capture application:

```bash
uv run bbf calibration hand-eye-gui \
  --config configs/local.yaml \
  --target configs/charuco_dict5x5_14x9_20mm_15mm.yaml \
  --output data/calibrations/es68_left_ir_hand_eye
```

The window starts idle and does not connect until **1. Start** is clicked. The
application never commands motion. Manually move the robot, stop at each pose, and
press **C** (or the save button) only after ChArUco PnP, stationarity, timestamp
bracketing, FK/TCP consistency, and pose-novelty gates are green. **Backspace** marks
the latest sample excluded without deleting its raw files. Use at least 20 training
poses with translation, image-region, distance, and rotation diversity about multiple
axes. Keep the final joint approach direction consistent and avoid changing J6 during
this first calibration.

Robot-pose semantics intentionally match the HoloRobot ES68 calibration path: the
solver consumes controller joint angles, evaluates the copied 709-pose calibrated ES68
FK, and passes `base_T_flange` to Park/BA. The controller-reported `base_T_tcp` is stored
and compared against `base_T_flange · flange_T_tcp` only as a capture-quality gate; it
is never a hand-eye solver input. For this calibrated ES68 chain the HoloRobot reference
uses `Δq=0`; optional `kinematics.joint_zero_offsets_rad` must therefore remain zero
unless a newer, independently accepted ES68 calibration explicitly changes it.

Every accepted pose stores raw left/right audit images, joint readings,
`base_T_flange`, observed `base_T_tcp`, all ChArUco IDs and 2D/3D correspondences, PnP
quality, D435i frame number, synchronization window, and FK/TCP discrepancy. The GUI
uses the HoloRobot-aligned Park-Martin initialization followed by joint LM/BA refinement
of `flange_T_left_ir` and the fixed `base_T_target` over every observed corner.

After solving, the GUI changes to an independent-validation phase. Collect at least
five **new** poses; these observations evaluate the fixed candidate and never refit it.
Only a candidate that passes fixed-board translation/rotation closure and corner
reprojection thresholds is atomically published to
`data/calibrations/es68_left_ir_hand_eye_active.yaml`. A failed validation remains a
candidate and cannot silently replace the active runtime calibration.

`--output` is a collection directory. Every click of Start creates a unique UTC session
below it. The session binds copies and SHA-256 hashes of the target, stereo calibration,
HoloRobot ES68 kinematics, flange/TCP offset, runtime settings, all raw/audit images,
accepted and excluded samples, candidate, validation reports, and final result. Treat
this directory as the immutable digital asset, not just the final YAML.

### Offline extraction and solving

The schema-2 sample set has explicit flange coordinates and BA observations:

```yaml
schema_version: 2
robot_model: es68
camera_stream: infrared/1
samples:
  - sample_id: pose-000
    base_T_flange: [[...], [...], [...], [0.0, 0.0, 0.0, 1.0]]
    base_T_tcp_observed: [[...], [...], [...], [0.0, 0.0, 0.0, 1.0]]
    joint_positions_rad: [q1, q2, q3, q4, q5, q6]
    left_ir_T_target: [[...], [...], [...], [0.0, 0.0, 0.0, 1.0]]
    detection:
      charuco_ids: [...]
      image_points_px: [[u, v], ...]
      object_points_m: [[x, y, z], ...]
```

Configure `hand_eye.target.square_length_m` and `marker_length_m` from physical
measurements of the printed ChArUco board. Board square counts, dictionary, and
`legacy_pattern` must match the generated print exactly. Then extract samples from one
or more stored sessions (repeat `--session` as needed):

```bash
uv run bbf calibration extract-hand-eye \
  --session data/<calibration-session-1> \
  --session data/<calibration-session-2> \
  --config configs/local.yaml \
  --output data/calibrations/hand_eye_samples.yaml
```

Extraction never connects to hardware. It uses each view's selected synchronized robot
state, raw left-IR intrinsics/distortion, identified ChArUco corners, and planar IPPE
pose estimation. Views are rejected when corner count, reprojection RMSE, positive-depth
geometry, or the primary/secondary planar-pose separation is inadequate. Both accepted
samples and rejection reasons are saved.

Solve it without connecting to any device:

```bash
uv run bbf calibration solve-hand-eye \
  --samples data/calibrations/hand_eye_samples.yaml \
  --stereo-calibration data/calibrations/d435i_ir_stereo_calibration.yaml \
  --config configs/local.yaml \
  --output data/calibrations/hand_eye.yaml
```

The default is Park-Martin + LM/BA. Daniilidis, Tsai-Lenz, Horaud, and Andreff remain
available for diagnostic comparison. Acceptance requires sample-count, motion
observability, FK/TCP consistency, fixed-target translation/rotation closure, and BA
reprojection thresholds. A real calibration still requires independent held-out poses
before any motion planning result is trusted; the offline solver writes a candidate and
does not auto-publish it.

## Controller-specific CS68 kinematics

The vendor KDL solver needs the modified DH parameters reported by the actual
controller. Acquire them through the read-only Primary interface:

```bash
uv run bbf robot export-kinematics \
  --config configs/local.yaml \
  --output data/calibrations/cs68_mdh.yaml
```

This command calls `PrimaryClientInterface.getPackage(KinematicsInfo)` and does not load
tasks, send scripts, release brakes, or command motion. Configure the resulting file as
`kinematics.model_path`. Offline IK converts each planned `base_T_left_ir` through the
stored hand-eye transform into `base_T_tcp`, then calls the packaged
`libelite_kdl_kinematics` solver near the joint state captured with the seed view.

An IK solution validates only an endpoint pose. It does not validate self-collision,
fixtures, cables, joint-space interpolation, singularity clearance, or a complete
trajectory.
