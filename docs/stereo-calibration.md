# D435i infrared stereo calibration

BiBladeFusion calibrates the two raw D435i infrared imagers without using the factory
IR intrinsics or factory left/right extrinsics. Acquisition and calibration are two
separate phases: the live application only previews and stores the newest synchronized
Y8 pair, then detects ChArUco corners and solves the calibration offline after capture.

The configured physical target is
`configs/charuco_dict5x5_14x9_20mm_15mm.yaml`: 14 by 9 squares, 20 mm square length,
15 mm marker length, and `DICT_5X5_100`. These values must match the printed board.

Install the optional desktop dependency and launch the application:

```bash
uv sync --extra calibration-gui
./.venv/bin/bbf calibration stereo-gui \
  --target configs/charuco_dict5x5_14x9_20mm_15mm.yaml \
  --config configs/default.yaml \
  --output data/calibrations/d435i_ir_20260827
```

`--output` is an asset collection root, not one overwriteable result directory. Every
GUI launch initially opens an idle window: it does not connect the camera, create a
session or count samples until the operator clicks **开始**. That action creates a unique
UTC-named session below the collection root and starts the count at zero. The preview
intentionally does not run corner detection. Place the board, hold it still, and click
**保存最新同步原始双目帧**.
Collect 50 to 60 raw poses near the image center, four corners and four edges, at
multiple distances, with substantial pitch, yaw, and roll. Avoid consecutive nearly
identical poses. When finished, choose a model and click the offline detection, Zhang
initialization and stereo BA button.

Each session is a self-contained digital asset:

```text
data/calibrations/d435i_ir_20260827/
└── session_YYYYMMDDTHHMMSS_ffffffZ/
    ├── session_manifest.json
    ├── configuration/
    │   └── charuco_target.yaml
    ├── raw_pairs/
    │   ├── pair_0000/
    │   │   ├── left_ir.png
    │   │   ├── right_ir.png
    │   │   └── frame_metadata.json
    │   └── ...
    └── analyses/
        └── analysis_001/
            ├── detection_summary.json
            ├── pairs/pair_XXXX/
            │   ├── detection.json
            │   ├── left_detection.png
            │   └── right_detection.png
            └── result/d435i_ir_stereo_calibration.yaml
```

Raw pair directories are appended atomically and never overwritten. The manifest binds
the copied target, D435i identity, stream settings, every frame number/timestamp and all
raw/result files with SHA-256. A completed session rejects further capture. A failed
offline run remains under its own `analysis_XXX` directory; the raw data stays intact,
so more poses can be captured and a new analysis can be run without losing evidence.

If the GUI was closed after acquisition, solve the preserved session directly:

```bash
./.venv/bin/bbf calibration stereo-solve-assets \
  --session data/calibrations/d435i_ir_20260827/session_YYYYMMDDTHHMMSS_ffffffZ \
  --minimum-samples 20 \
  --distortion-model auto
```

This creates a new `analysis_XXX` directory and does not modify any raw pair.

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

After a successful solve, the program checks the solver's configured acceptance gates and
atomically publishes the solver-accepted result to the fixed runtime path used by later
workflows:

```yaml
realsense:
  stereo_calibration_path: data/calibrations/d435i_ir_active.yaml
```

Publication activates the runtime copy; it does not replace the independent hold-out
experiment below. The timestamped session result remains the authoritative digital asset. The fixed
runtime copy removes manual path editing: stereo rectification, FoundationStereo,
hand-eye capture, reconstruction and calibrated view planning load it through the
normal application settings. If it is missing or invalid, D435i production capture
fails closed; factory IR intrinsics/extrinsics are never used as a fallback.

The loader rejects artifacts that do not explicitly record
`factory_intrinsics_used: false` and rejects a stream resolution that differs from the
calibrated resolution. Native RealSense depth remains a separate vendor-calibrated
sensor product; it is not used to calibrate the raw left/right infrared stereo pair.

## Independent hold-out validation

Do not judge a calibration only with images that participated in its solution. After
publishing the user calibration, acquire 8 to 12 new ChArUco poses with the dedicated
validation application:

```bash
./.venv/bin/bbf calibration stereo-validate-gui \
  --target configs/charuco_dict5x5_14x9_20mm_15mm.yaml \
  --config configs/default.yaml \
  --output data/calibrations/d435i_ir_validation
```

The calibration defaults to `realsense.stereo_calibration_path`. Use
`--calibration PATH` only when validating a different timestamped calibration asset.
The window starts idle and its three numbered buttons define the complete workflow:

1. **开始并连接相机** creates a unique validation session and starts raw preview;
2. **保存当前同步双目图像** manually stores a new hold-out pair;
3. **采集完成，离线验证固定标定参数** stops capture and performs detection,
   rectification, visualization, and metric calculation.

No solver is reachable from this command. The copied camera matrices, distortion
vectors and `right_ir_T_left_ir` remain fixed, and every manifest/report explicitly
records `calibration_refit_performed: false`.

Collect poses that were not used during calibration. Cover the image center, four
edges and corners, at least two distances, and meaningful pitch/yaw/roll. Keep the
board still when saving. Do not select consecutive nearly identical frames merely to
reach the minimum count.

Each launch produces a self-contained digital asset:

```text
data/calibrations/d435i_ir_validation/
└── validation_YYYYMMDDTHHMMSS_ffffffZ/
    ├── validation_manifest.json
    ├── configuration/
    │   ├── charuco_target.yaml
    │   └── fixed_stereo_calibration.yaml
    ├── raw_pairs/pair_XXXX/
    │   ├── left_ir.png
    │   ├── right_ir.png
    │   └── frame_metadata.json
    └── analyses/analysis_001/
        ├── validation_report.json
        ├── validation_summary.txt
        └── pairs/pair_XXXX/
            ├── validation.json
            ├── left_detection.png
            ├── right_detection.png
            ├── left_rectified.png
            ├── right_rectified.png
            └── rectified_epipolar_overlay.png
```

The default quality gates are intentionally explicit and are stored in the manifest:

- at least 8 accepted independent pairs;
- rectified vertical disparity RMSE no more than 0.5 px;
- rectified vertical disparity P95 no more than 1.0 px;
- left and right independent reprojection RMSE no more than 0.5 px;
- left-to-right stereo transfer RMSE no more than 1.0 px.

These are validation gates, not a substitute for examining spatial trends in the
per-pair overlays. A systematic signed vertical offset, errors concentrated near one
image edge, or errors increasing with board distance should be investigated even when
the aggregate result passes. Thresholds can be overridden on the GUI command and the
actual values used are always retained in the asset.

If a captured validation session was closed before analysis, process it without the
camera:

```bash
./.venv/bin/bbf calibration stereo-validate-assets \
  --session data/calibrations/d435i_ir_validation/validation_YYYYMMDDTHHMMSS_ffffffZ
```
