# Calibration and frame conventions

BiBladeFusion uses `parent_T_child`: a transform maps coordinates expressed in the child
frame into the parent frame. Distances are metres and angles are radians unless a field
name explicitly says otherwise.

## Eye-in-hand calibration

The required camera transform is `tcp_T_left_ir`, not its inverse. Store the measured
artifact outside version control and reference it through `hand_eye.calibration_path`.
The YAML schema is:

```yaml
schema_version: 1
parent_frame: tcp
child_frame: left_ir
method: <calibration method>
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
```

The loader checks the rigid transform, frame names, sample count, and configured RMSE
limits. An identity transform used as a placeholder is unsafe and intentionally
unsupported by the real-data workflow.

The offline solver consumes a fixed-target eye-in-hand sample set. For every sample,
`base_T_tcp` comes from the synchronized Elite state and `left_ir_T_target` comes from a
calibration-target pose estimate in the raw left infrared camera frame:

```yaml
schema_version: 1
samples:
  - sample_id: pose-000
    base_T_tcp: [[...], [...], [...], [0.0, 0.0, 0.0, 1.0]]
    left_ir_T_target: [[...], [...], [...], [0.0, 0.0, 0.0, 1.0]]
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
  --config configs/local.yaml \
  --output data/calibrations/hand_eye.yaml
```

The default Park-Martin solution is accepted only when the dataset meets sample-count,
rotation-span, translation-span, and rotation-axis-diversity thresholds. Quality is the
fixed-target closure RMSE: each sample reconstructs
`base_T_target = base_T_tcp · tcp_T_left_ir · left_ir_T_target`, and those reconstructed
target poses must agree. A real calibration still requires independent validation on
held-out poses before any motion planning result is trusted.

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
