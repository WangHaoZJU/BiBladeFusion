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

To use the completed result for normal capture and FoundationStereo rectification, add
the actual session/result path to the local application configuration:

```yaml
realsense:
  stereo_calibration_path: data/calibrations/d435i_ir_20260827/session_.../analyses/analysis_001/result/d435i_ir_stereo_calibration.yaml
```

The loader rejects artifacts that do not explicitly record
`factory_intrinsics_used: false` and rejects a stream resolution that differs from the
calibrated resolution. Native RealSense depth remains a separate vendor-calibrated
sensor product; it is not used to calibrate the raw left/right infrared stereo pair.
