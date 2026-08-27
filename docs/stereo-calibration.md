# D435i infrared stereo calibration

BiBladeFusion calibrates the two raw D435i infrared imagers without using the factory
IR intrinsics or factory left/right extrinsics. The live application reads only the
synchronized Y8 image arrays from the device.

The configured physical target is
`configs/charuco_dict5x5_14x9_20mm_15mm.yaml`: 14 by 9 squares, 20 mm square length,
15 mm marker length, and `DICT_5X5_100`. These values must match the printed board.

Install the optional desktop dependency and launch the application:

```bash
uv sync --extra calibration-gui
uv run bbf calibration stereo-gui \
  --target configs/charuco_dict5x5_14x9_20mm_15mm.yaml \
  --config configs/local.yaml \
  --output data/calibrations/d435i_ir
```

Only a frame with enough detected ChArUco corners in both cameras can be accepted.
Collect the board near the image center and all four corners, at multiple distances,
with substantial pitch, yaw, and roll. Avoid consecutive nearly identical frames.

The solver first runs independent Zhang calibration for the left and right cameras.
Those results initialize a joint nonlinear stereo optimization which refines both
camera matrices, both distortion vectors, and the fixed `right_ir_T_left_ir` transform
against all common stereo ChArUco observations. The output reports monocular RMS, joint
stereo RMS, epipolar RMSE/P95, and baseline.

The GUI provides four distortion choices:

- `brown5` (default): `k1, k2, p1, p2, k3`;
- `radial2`: `k1, k2`, with tangential and higher-order coefficients fixed to zero;
- `rational8`: `k1, k2, p1, p2, k3, k4, k5, k6`;
- automatic comparison: deterministically holds out every fifth accepted view, compares
  all three models using validation reprojection and epipolar errors, favors the
  simplest model within a small tolerance of the best validation result, and then
  refits the selected model on all observations.

Automatic comparison requires at least 20 accepted observations. Model selection never
uses the factory calibration and never relies only on training RMS.

The output directory contains immutable source image pairs and
`d435i_ir_stereo_calibration.yaml`. To use it for normal capture and FoundationStereo
rectification, add this to the local application configuration:

```yaml
realsense:
  stereo_calibration_path: data/calibrations/d435i_ir/d435i_ir_stereo_calibration.yaml
```

The loader rejects artifacts that do not explicitly record
`factory_intrinsics_used: false` and rejects a stream resolution that differs from the
calibrated resolution. Native RealSense depth remains a separate vendor-calibrated
sensor product; it is not used to calibrate the raw left/right infrared stereo pair.
