# Static native-depth overlap validation

This experiment validates the runtime coordinate chain used to register a D435i native
depth cloud into the ES68 base frame:

```text
base_T_depth = base_T_tcp · tcp_T_left_ir · left_ir_T_depth
```

It is intentionally independent of FoundationStereo. Native D435i depth isolates
synchronized robot state, the published hand-eye transform, the RealSense depth-stream
geometry, and point-cloud projection. It tests static-scene internal consistency; it is
not a traceable absolute-dimensional calibration.

## Acquisition

Fix a matte scene with non-coplanar structure. Move the robot only with the teach pendant,
wait until it is stationary, and capture one immutable session per pose:

```bash
uv run bbf acquire snapshot --view-id overlap_00_reference --emitter
uv run bbf acquire snapshot --view-id overlap_01_left --emitter
uv run bbf acquire snapshot --view-id overlap_02_right --emitter
uv run bbf acquire snapshot --view-id overlap_03_up --emitter
uv run bbf acquire snapshot --view-id overlap_04_down --emitter
```

The command is read-only with respect to the robot. Every session records raw infrared
images, native depth, bracketed controller states, stream calibration, synchronization
metrics, and the effective configuration including the command-scoped projector state.

## Evaluation

Pass the reference session once and repeat `--session` for every comparison view:

```bash
uv run bbf evaluate native-overlap \
  --reference data/<reference-session> \
  --session data/<left-session> \
  --session data/<right-session> \
  --session data/<up-session> \
  --session data/<down-session> \
  --config configs/default.yaml \
  --output data/validations/native_overlap_<experiment>
```

The primary calculation is symmetric and does not use ICP:

1. Deproject smooth, range-valid reference depth pixels in the native `depth` frame.
2. Apply the unmodified calibrated transform chain into `base`.
3. Transform those base points into each comparison depth camera and project them into
   its native depth image.
4. Compare predicted and measured axial depths after rejecting invalid pixels, local
   depth discontinuities, field-of-view loss, and explicit occlusion/surface-change
   residuals.
5. Repeat in the reverse direction and combine the signed residuals.

Each pair reports projected counts, same-surface inlier fraction, signed bias, MAE,
median absolute error, RMSE, P95, and agreement at 2 mm and 5 mm. Translation and rotation
pose spans are separate observability gates.

## ICP boundary

A bounded point-to-plane ICP correction is calculated only as a diagnostic. Its matrix,
translation, rotation, correspondence count, and before/after residual are recorded, but
the correction is never applied to:

- primary projective residuals;
- pass/fail thresholds;
- exported overlay points; or
- any hand-eye or runtime calibration file.

This prevents registration optimization from hiding a bad hand-eye transform. Large ICP
corrections on a mostly planar scene can also be underconstrained and must not be treated
as a replacement calibration result.

## Immutable output

The output directory contains:

```text
native_overlap_report.json   source hashes, transforms, metrics and verdict
metrics.csv                  compact per-view table
overview.png                 three PCA projections of the uncorrected overlay
overlay_base_frame.ply       coloured uncorrected clouds in the ES68 base frame
arrays/
  overlay_points_m.npy
  overlay_view_indices.npy
  residual_*.npy             symmetric signed residual evidence
```

Reading the artifact verifies every source manifest, configuration snapshot, view
metadata file, native-depth array and hand-eye YAML, then recomputes the complete report.
Changed sources, arrays, metrics or visual products are rejected.

## 2026-08-27 hardware result

Five real ES68/D435i views covered a 188.01 mm translation span and 23.683 degree rotation
span. With no ICP correction applied, all four reference comparisons passed:

| View | Inliers | Median | RMSE | P95 | Within 5 mm |
|---|---:|---:|---:|---:|---:|
| left | 99.75% | 1.424 mm | 2.470 mm | 4.993 mm | 95.02% |
| right | 99.91% | 1.220 mm | 2.070 mm | 4.205 mm | 97.36% |
| up | 99.64% | 1.402 mm | 2.300 mm | 4.617 mm | 96.33% |
| down | 99.39% | 1.337 mm | 2.398 mm | 4.876 mm | 95.38% |

The retained artifact is
`data/validations/native_overlap_20260827_static_v1`. These results support the internal
coordinate-chain consistency required for initial native-depth coarse scanning; physical
dimension accuracy still requires a known-size or metrology reference.
