# BiBladeFusion

**BiBladeFusion** is a robot-guided bilateral 3D geometry and thermal reconstruction
system for thin-walled blades.

The current development stage provides a Python 3.12 application, validated
configuration, read-only Elite ES68 commands plus a default-off guarded control backend,
synchronized raw
stereo acquisition from an Intel RealSense D435i, reproducible session storage, a
calibrated FoundationStereo inference path, paired native/stereo depth evaluation, a
conservative single-view blade proxy, paper-derived true curved-surface partitioning,
thin-wall-aware multi-view TSDF/mesh reconstruction, real-surface quality feedback, and
offline Elite KDL endpoint IK. Robot motion is disabled by default.

## Bootstrap

```bash
./scripts/bootstrap.sh
```

The bootstrap script creates the project virtual environment with `uv`, synchronizes
the locked dependencies, and installs the local Elite CS SDK wheel.

Initialize the pinned official stereo source and install its optional runtime only on a
FoundationStereo inference machine:

```bash
git submodule update --init --recursive
uv sync --extra foundation-stereo
```

## Verify

```bash
uv run bbf version
uv run bbf doctor
uv run bbf stereo doctor
uv run pytest
uv run ruff check .
```

`bbf stereo doctor` intentionally fails until the official FoundationStereo source,
checkpoint, inference dependencies, and requested CUDA device are present. The main
project remains on Python 3.12; the upstream Python 3.11 environment is treated as a
tested baseline, not as a hard-coded interpreter restriction.

After placing the official checkpoint and its adjacent `cfg.yaml` at the configured
paths, infer from an immutable stored session without touching hardware:

```bash
uv run bbf stereo infer-session \
  --session data/<session> \
  --view-id seed \
  --config configs/local.yaml \
  --output outputs/stereo_seed
```

The output stores rectified images, full-resolution-pixel disparity, metric depth,
validity masks, calibration transforms, model provenance, and per-array SHA-256 values.
Use the [paired depth evaluator](docs/depth-comparison.md) to compare it against native
D435i depth in calibrated rectified-left coordinates without treating either source as
metrology ground truth.
Use a Boolean mask in rectified-left coordinates to initialize the same conservative
blade proxy from this depth source:

```bash
uv run bbf initialize stereo-depth \
  --session data/<session> \
  --stereo outputs/stereo_seed \
  --view-id seed \
  --mask data/blade_mask_rectified.npy \
  --config configs/local.yaml \
  --output outputs/initialization_stereo
```

## Read-only acquisition

Set `robot.robot_ip` in a Git-ignored `configs/local.yaml`, then run:

```bash
uv run bbf robot status --config configs/local.yaml
uv run bbf camera list
uv run bbf acquire snapshot --config configs/local.yaml --view-id seed
uv run bbf robot export-kinematics \
  --config configs/local.yaml \
  --output data/calibrations/cs68_mdh.yaml
```

The synchronized snapshot brackets the D435i capture with two RTSI robot states and
rejects it when timing or stationary-state tolerances are exceeded. It does not issue
robot motion commands.

## User-calibrated D435i infrared stereo

The raw left/right IR calibration path does not use D435i factory IR intrinsics or
factory stereo extrinsics. A PySide6 application first stores latest-frame synchronized
Y8 pairs as unique, checksummed digital-asset sessions, then performs ChArUco detection,
independent Zhang initialization and joint stereo optimization offline. Raw pairs,
accept/reject evidence and every calibration result remain traceable and non-overwriting. See
[D435i infrared stereo calibration](docs/stereo-calibration.md).

## Bilateral initialization

The initial visible-face point cloud is reduced to a density-balanced voxel cloud and
used to estimate the blade's two in-plane principal axes. The unseen side is explicitly
extruded away from the initial camera by `estimated_thickness_m`, with separate visible,
hidden, and tangential safety margins. Proxy construction refuses to continue when the
thickness prior is unset, the cloud is degenerate, or the initial view is too grazing.

`estimated_planar_extents_m` is optional and ordered as `(major, minor)`. When supplied,
the proxy uses the larger of the observed dimensions and these conservative prior
dimensions. The resulting proxy center is a planning-volume center, not a claim about
the blade's physical center of mass.

Hand-eye input is quality-gated and is solved as `flange_T_left_ir` from the raw D435i
left-IR stream and calibrated ES68 FK. See
[calibration and frame conventions](docs/calibration.md) before processing real data.
Current implementation status and the prioritized remaining work are tracked in the
[development log](docs/development-log.md).
The copied/adapted HoloRobot control lifecycle, pose convention, collision preflight,
and approval boundary are documented in the
[Elite CS68 control safety contract](docs/elite-cs68-control.md).

Stored calibration sessions can be converted to an auditable ChArUco sample set and
solved entirely offline:

```bash
uv run bbf calibration extract-hand-eye \
  --session data/<calibration-session> \
  --config configs/local.yaml \
  --output data/calibrations/hand_eye_samples.yaml
uv run bbf calibration solve-hand-eye \
  --samples data/calibrations/hand_eye_samples.yaml \
  --stereo-calibration data/calibrations/d435i_ir_stereo_calibration.yaml \
  --config configs/local.yaml \
  --output data/calibrations/hand_eye.yaml
```

For direct synchronized manual-pose collection, use the PySide6 workflow:

```bash
uv run bbf calibration hand-eye-gui \
  --config configs/local.yaml \
  --output data/calibrations/es68_left_ir_hand_eye_run_01
```

## Offline planning workflow

For the native-depth path, create a Boolean `.npy` blade mask in the native depth-image
coordinate system. Then set
`hand_eye.calibration_path`, `kinematics.model_path`, `proxy_model.estimated_thickness_m`,
`view_planning.standoff_distance_m`, and measured workcell bounds in the local config.
For curved fine planning, also set the validated
`minimum_standoff_distance_m`/`maximum_standoff_distance_m` pair; the baseline partition
footprint is derived from the stored user-calibrated left-IR intrinsics rather than a
fixed physical rectangle.

```bash
uv run bbf initialize native-depth \
  --session data/<session> \
  --view-id seed \
  --mask data/blade_mask.npy \
  --config configs/local.yaml \
  --output outputs/initialization

uv run bbf plan views \
  --initialization outputs/initialization \
  --config configs/local.yaml \
  --output outputs/view_plan

uv run bbf coverage seed \
  --plan outputs/view_plan \
  --initialization outputs/initialization \
  --config configs/local.yaml \
  --output outputs/coverage_000
```

The planner creates both front and back partitions from the conservative proxy. Each
view is checked for optical alignment, incidence, coverage, standoff, camera clearance,
workspace bounds, forbidden volumes, duplicate poses, and—when configured—offline CS68
IK. A `geometry_only` view has not passed IK/workspace validation. An
`endpoint_feasible` view still has **not** passed robot-body collision or trajectory
validation. Every exported plan contains `motion_authorized: false`; no current command
executes a planned pose.

The coverage ledger uses pose-registered base-frame blade points to fill independent
front/back per-patch occupancy grids. See [coverage and replanning](docs/coverage.md) for
its evidence rules and current limitations.

After proxy views have produced overlapping coarse observations on both physical sides,
use the [paper-derived curved reconstruction](docs/curved-reconstruction.md) workflow.
It replaces proxy planes with measured curved patches, independently partitions the
leading edge, trailing edge, root, and tip, robustly fits the irregular four-boundary
outline, uses curve-driven equal-arc coordinates with a shared bilateral base grid,
separates the specimen's one front and one back protruding fin, plans their two faces,
attachment roots, and free rims with dedicated normals, protects both blade and measured
fin thickness during TSDF integration, and reports component-level surface/mesh quality. Install
`uv sync --extra tsdf-open3d` to enable the optional calibrated Open3D backend; the
locked NumPy fallback remains available without it.

Before attempting robot feasibility, run `bbf reconstruct inspect-fine-plan` on the
schema-4 coarse model. It exports portable inspection geometry/reports and can open a
read-only PySide6 orbit viewer; a geometric pass still records robot feasibility as
unverified and cannot authorize motion.

Explicit ordered view sequences can be checked offline with the fail-closed
[CS68 collision and path validator](docs/collision-validation.md). It uses configured
capsule/workcell geometry and controller-specific MDH data, but remains a conservative
prefilter and never authorizes motion.

After an ordered sequence has endpoint-feasible IK solutions, generate the copied
HoloRobot mesh-collision and velocity-limited ServoJ preflight artifact:

```bash
uv run bbf safety preflight-path \
  --plan outputs/view_plan \
  --initialization outputs/initialization \
  --view-id front_r00_c00 \
  --view-id back_r00_c00 \
  --config configs/local.yaml \
  --output outputs/motion_preflight
```

The artifact binds and verifies its source hashes, re-derives every leg when read, and
always stores `motion_authorized: false`. This command does not connect to the robot.
