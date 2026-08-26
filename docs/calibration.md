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
```

The loader checks the rigid transform, frame names, sample count, and configured RMSE
limits. The repository does not currently estimate hand-eye calibration; importing an
identity transform as a placeholder is unsafe and intentionally unsupported by the
real-data workflow.

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
