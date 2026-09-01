# BiBladeFusion

**BiBladeFusion** is a robot-guided bilateral 3D-geometry reconstruction system for
thin-walled blades, with a planned thermal-reconstruction extension.

The current development stage provides a Python 3.12 application, Elite ES68 diagnostics
plus a fail-closed supervised-control stack, synchronized raw stereo acquisition
from an Intel RealSense D435i, reproducible session storage, a calibrated
FoundationStereo inference path, paired native/stereo depth evaluation, a conservative
single-view blade proxy, paper-derived true curved-surface partitioning, thin-wall-aware
multi-view TSDF/mesh reconstruction, real-surface quality feedback, and offline Elite KDL
endpoint IK. A fixed-reference fine-coverage ledger, deterministic bilateral-fin
next-view selector, and transactional fine-scan foreground/reconstruction/coverage
branch are implemented. It also contains a fail-closed FoundationStereo-derived
three-state safety occupancy layer, conservative continuous mesh and occupancy sweep
certificates, an operator-guided unknown-blade coarse loop, a one-way schema-5 handoff,
an append-only experiment checkpoint/completion chain, resumable fail-closed execution,
and a read-only live/replay supervisory console. The curved reconstruction
chain is currently regression-verified on deterministic synthetic bilateral-blade data;
real-blade accuracy and hardware thresholds still require recorded experiments. Thermal
capture/fusion is not implemented beyond a disabled interface placeholder. Every ordinary
Elite-arm motion method remains sealed; the supervised path can reach the private driver
capability only after a fresh map, both continuous proofs, an exact one-segment preflight
and the operator's segment-specific confirmation. It remains disabled by default and has
not been validated on the physical blade/ES68/D435i system. Software completion must not
be interpreted as physical acceptance.

## Bootstrap

```bash
./scripts/bootstrap.sh
```

The bootstrap script creates the project virtual environment with `uv`, synchronizes
the locked dependencies, and installs the local Elite CS SDK wheel.

The committed default configuration currently retains the present laboratory bring-up IP
addresses and absolute SDK-wheel path. Keep them unchanged for this rig; before public
release or deployment to another workstation, move machine-specific addresses and paths
to a Git-ignored `configs/local.yaml` and remove them from the distributable defaults.

Initialize the pinned official stereo source and install its optional runtime only on a
FoundationStereo inference machine:

```bash
git submodule update --init --recursive
uv sync --extra foundation-stereo
```

For a fresh GPU workstation, the strict bootstrap also checks CUDA, installs the local
Elite SDK wheel, and runs the non-moving stereo doctor. Pass the wheel explicitly when
the laboratory path is not present on that machine:

```bash
./scripts/bootstrap-gpu.sh \
  /absolute/path/to/elite_cs_sdk-1.0.0-cp312-cp312-linux_x86_64.whl \
  configs/local.yaml
```

This is an environment check, not a FoundationStereo accuracy/latency acceptance or a
robot-motion release. A saved-session inference smoke test is still required.

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
  --output data/calibrations/es68_mdh.yaml
```

The synchronized snapshot brackets the D435i capture with two RTSI robot states and
rejects it when timing or stationary-state tolerances are exceeded. It does not issue
robot motion commands.

## User-calibrated D435i infrared stereo

The raw left/right IR calibration path does not use D435i factory IR intrinsics or
factory stereo extrinsics. A PySide6 application first stores latest-frame synchronized
Y8 pairs as unique, checksummed digital-asset sessions, then performs ChArUco detection,
independent Zhang initialization and joint stereo optimization offline. Raw pairs,
accept/reject evidence and every calibration result remain traceable and non-overwriting.
The solver-accepted result is automatically published to the fixed runtime path consumed
by later workflows; publication activates a result but is not independent hold-out
validation. Missing user calibration fails closed without factory IR fallback. See
[D435i infrared stereo calibration](docs/stereo-calibration.md).

Validate the published parameters on new images that were not used by the solver:

```bash
uv sync --extra calibration-gui
./.venv/bin/bbf calibration stereo-validate-gui \
  --config configs/default.yaml \
  --output data/calibrations/d435i_ir_validation
```

The validation command never refits calibration parameters. It archives raw hold-out
pairs, rectified epipolar overlays, per-pair metrics, aggregate pass/fail gates, and
SHA-256-bound provenance; see the independent-validation section in
[D435i infrared stereo calibration](docs/stereo-calibration.md).

Static robot/hand-eye/native-depth coordinate-chain acceptance is documented in
[static native-depth overlap validation](docs/native-overlap-validation.md). Its primary
metrics never apply ICP; optional ICP output is diagnostic only.

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

Production unknown-blade runs additionally require a measured base-frame blade envelope:
`proxy_model.blade_envelope_min_m`, `blade_envelope_max_m`, and
`minimum_envelope_retained_fraction`. After the hard ROI is back-projected and transformed
into `base`, only its intersection with that AABB supports proxy PCA. Every later coarse
view repeats and persists the same support selection; proxy coverage, multi-view PCA/ICP,
TSDF, surface partitioning, and fine-reference construction consume only those support
points. Complete hard-ROI clouds and pixel mappings remain immutable. Initialization schema
8 and coarse-view schema 2 store aligned support masks plus input/retained bounds and counts.
Too few retained points or a retained fraction below the configured gate fails closed.

Hand-eye input is quality-gated and is solved as `flange_T_left_ir` from the raw D435i
left-IR stream and calibrated ES68 FK. See
[calibration and frame conventions](docs/calibration.md) before processing real data.
Native and FoundationStereo reconstruction use synchronized joints plus the configured
joint-zero offsets to reproduce `base_T_flange`; the controller-reported `base_T_tcp`
is accepted only as a validation residual within 2 mm and 0.3 degrees by default.
Legacy TCP-primary hand-eye artifacts remain readable for inspection but are rejected by
reconstruction, planning, collision, and motion-preflight paths.
Current implementation status and the prioritized remaining work are tracked in the
[development log](docs/development-log.md).
The copied/adapted HoloRobot control lifecycle, pose convention, collision preflight,
and approval boundary are documented in the
[Elite ES68 control safety contract](docs/elite-cs68-control.md).

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
uv sync --extra calibration-gui
uv run bbf calibration hand-eye-gui \
  --config configs/local.yaml \
  --output data/calibrations/es68_left_ir_hand_eye
```

The GUI starts idle, saves manually with `C`, solves with Park-Martin + LM/BA, then
requires new held-out poses before atomically publishing the active runtime YAML. Each
run is stored as a unique, hash-bound digital-asset session below `--output`.

After a completed run, collect additional frozen-parameter evidence without retraining:

```bash
uv run bbf calibration hand-eye-validate-gui \
  --calibration data/calibrations/es68_left_ir_hand_eye_active.yaml \
  --config configs/default.yaml \
  --output data/calibrations/es68_left_ir_hand_eye_validation
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

uv run bbf coverage next-plan \
  --ledger outputs/coverage_000 \
  --plan outputs/view_plan \
  --start-side front \
  --output outputs/next_view_plan_000
```

The planner creates both front and back partitions from the conservative proxy. Each
view is checked for optical alignment, incidence, coverage, standoff, camera clearance,
workspace bounds, forbidden volumes, duplicate poses, and—when configured—offline ES68
IK. A `geometry_only` view has not passed IK/workspace validation. An
`endpoint_feasible` view still has **not** passed robot-body collision or trajectory
validation. Every exported plan contains `motion_authorized: false`; the commands in this
offline plan/coverage workflow never execute an exported pose. The separate production
entry point, `bbf scan run-unknown`, can command the robot only after the complete runtime
authority chain, live stop/stationarity checks, collision revalidation, and an exact
single-segment operator approval have all succeeded.

The coverage ledger uses pose-registered base-frame blade points to fill independent
front/back per-patch occupancy grids. See [coverage and replanning](docs/coverage.md) for
its evidence rules and current limitations.

Coverage-plan schema 2 removes completed proxy patches, admits only
`endpoint_feasible` candidates with stored joint solutions into the proposed order, and
finishes the selected side with a deterministic row-wise snake before changing sides.
`front` is the initial camera-visible proxy side by construction. `geometry_only` views
remain auditable but deferred. Joint travel is not an ordering objective, and this
proposal carries no collision evidence or motion authority.

After proxy views have produced overlapping coarse observations on both physical sides,
use the [paper-derived curved reconstruction](docs/curved-reconstruction.md) workflow.
It replaces proxy planes with measured curved patches, independently partitions the
leading edge, trailing edge, root, and tip, robustly fits the irregular four-boundary
outline, uses curve-driven equal-arc coordinates with a shared bilateral base grid,
separates the specimen's one front and one back protruding fin, plans their two faces,
attachment roots, and free rims with dedicated normals, protects both blade and measured
fin thickness during TSDF integration, and reports component-level surface/mesh quality.
These functions are software-verified on deterministic synthetic data; real D435i/ES68
coarse scans and dimensional references are still required for physical validation.
The [coverage-driven fine selector](docs/coverage-next-view-selector.md) keeps cumulative
scientific surface evidence separate from the bounded rolling safety-occupancy window and
never interprets an unreachable incomplete state as completion. The concrete cycle engine
can now stage a reference-projected foreground mask, a FoundationStereo reconstructed view,
and exactly one fine-coverage successor as one library-level transaction. Its schema-3
scientific view is replayed from the bound stereo depth, occupancy-derived eligible mask,
foreground mask, point-cloud configuration and raw/rectified camera chain; an online
recovery rejects legacy schema-2 observations anywhere in the accepted lineage. The branch
is disabled in the coarse engine and requires an explicitly pinned schema-5 coarse model
and optional recovery generation at construction. The supervised `scan run-unknown`
composition enables it only after a verified one-way schema-5 handoff; it never transfers
an old permit, prepared segment, map publication, or coverage state. It is software-tested
only; no real-blade accuracy or motion-release claim follows.
Install
`uv sync --extra tsdf-open3d` to enable the optional calibrated Open3D backend; the
locked NumPy fallback remains available without it.

Before attempting robot feasibility, run `bbf reconstruct inspect-fine-plan` on the
schema-5 coarse model. It exports portable inspection geometry/reports and can open a
read-only PySide6 orbit viewer; a geometric pass still records robot feasibility as
unverified and cannot authorize motion.

```bash
uv sync --extra supervision-gui
```

Manual or coverage-derived ordered view sequences can be checked offline with the fail-closed
[ES68 collision and path validator](docs/collision-validation.md). It uses configured
capsule/workcell geometry and controller-specific MDH data, but remains a conservative
prefilter and never authorizes motion.

## Unknown-blade occupancy and supervision

For an unknown blade, environment collision evidence is derived from settled,
pose-registered FoundationStereo observations rather than a prior blade STL. Each
accepted observation must carry the configured left-right consistency evidence and
its derived score array. The stored score is
`exp(-|d_L-d_R| / configured_LR_threshold)`; it is a deterministic,
non-probabilistic consistency score, not a calibrated probability. Calibrated ES68 FK
from the synchronized joint vector is the
authoritative pose source: `base_T_flange · flange_T_left_ir` places the camera and depth
in `base`. The synchronized RTSI TCP is validation-only; before any ray enters the map,
`base_T_flange · flange_T_tcp` must agree with it within 2 mm and 0.3 degrees by default.
Both source poses and residuals are stored in the immutable evidence chain and re-derived
on read. The final ES68+D435i collision meshes are rendered into the left-IR camera and
depth-consistently masked before ray integration. Masked pixels do not clear the robot or
the space hidden behind it.

The sparse voxel map has exactly three semantic states:

- `FREE`: traversed by enough geometrically independent accepted depth rays before their
  measured surfaces;
- `OCCUPIED`: supported by accepted surface evidence;
- `UNKNOWN`: neither proven free nor occupied, including out-of-grid space and pixels
  removed by the robot self-mask.

`UNKNOWN` blocks motion. A voxel needs three independent FREE votes by default, with at
most one vote per view. A new supporting view must differ from every prior supporting
view by at least 20 mm of camera-centre translation or 5 degrees of optical-axis angle;
changing only a view identifier is rejected before ray integration. Mapping is
stop-and-capture. Committed source views remain available until a later generation is
accepted; the motion-authorization age starts when that generation is atomically
published. This is not continuous dynamic obstacle avoidance. The following command reconstructs
immutable mapping evidence from at least three previously stored FoundationStereo
artifacts:

```bash
uv run bbf occupancy build-replay \
  --stereo outputs/stereo_view_000 \
  --stereo outputs/stereo_view_001 \
  --stereo outputs/stereo_view_002 \
  --config configs/local.yaml \
  --output outputs/occupancy_replay
uv run bbf occupancy inspect --artifact outputs/occupancy_replay
```

`occupancy build-replay` **always** seals the result as `STALE`; it is useful for
algorithm verification and audit only, and can never become live motion evidence. The
production renderer also fails closed when the ready, final ES68+D435i STL manifest is
absent or mismatched. Follow the [final ES68+D435i collision-model activation
checklist](src/biblade_fusion/robotics/resources/elite_cs/collision_models/es68_d435i/README.md)
when installing those meshes. The supervised coordinator atomically combines stopped
robot state, FoundationStereo inference, fresh-map publication, scientific-asset staging,
planning invalidation, two conservative continuous sweep proofs, one-segment approval and
execution freshness. It remains default-off and has not been hardware-verified.

The current HoloRobot-derived ES68 and D435i-only collision assembly can be checked in a
fully offline Qt3D viewer. The command exposes no robot IP and never opens a robot or
camera driver:

```bash
uv sync --extra robot-model-gui
uv run bbf robot inspect-model --config configs/default.yaml
```

Use the six joint controls and per-STL visibility switches to check the wrist/flange
mount orientation and link motion. This visual audit does not replace FCL path validation
or final dimensional acceptance on the physical installation.

Create and open a self-contained, read-only supervisory replay snapshot with:

```bash
uv sync --extra supervision-gui
uv run bbf supervise build-replay \
  --occupancy outputs/occupancy_replay \
  --current-view outputs/reconstructed_view_002 \
  --coarse-model outputs/coarse_model \
  --output outputs/supervision/snapshot_0002
uv run bbf supervise replay \
  --snapshot outputs/supervision/snapshot_0002
```

The GUI visualizes only evidence it can bind: occupancy and implicit unknown workspace,
the historical robot chain and camera pose, current/fused blade point clouds, sensor
quality, copied source manifests and blocking events. Exact ES68+D435i meshes appear only
when the active final collision model reproduces the mapping geometry hash; planned TCP
paths appear only from a canonically derived preflight. Actual-path samples currently mean
stopped perception stations, not high-rate ServoJ tracking. The GUI is strictly an observer: replay snapshots remain
`REPLAY/BLOCKED`, and it contains no approval or robot-command path. `--follow` only
polls atomically published replay snapshots; it does not provide online avoidance or a
deterministic control loop.

## Supervised unknown-blade runtime

The production composition is exposed as one deliberately interactive command. First
copy the default configuration to a Git-ignored local file and complete the measured
workspace, occupancy, stop-and-capture, collision, standoff, calibration and
static-free, motion-envelope and geometry-science acceptance fields, including measured
timing budgets. Audit the complete coarse-to-fine chain without opening
the robot or camera:

```bash
uv run bbf scan doctor --mode unknown --config configs/local.yaml
```

Only after that audit passes, create a new non-overwriting experiment root:

```bash
uv run bbf scan run-unknown \
  --config configs/local.yaml \
  --output data/experiments/blade-placement-20260831-01-attempt-01 \
  --placement-id blade-placement-20260831-01 \
  --run-id blade-placement-20260831-01-attempt-01 \
  --operator-id vale
```

`placement-id` identifies one unchanged physical blade+fixture placement and is bound
into the immutable experiment INIT event. A software retry without moving the workpiece
reuses that placement ID but uses a new run ID and output root. Any physical movement
requires a new placement ID; no occupancy map, schema-5 reference, or fine coverage is
reused across placements.

The first stage never chooses a path through unknown space. The operator manually places
the stopped arm at at least three independently safe views and presses exactly `c` once
per capture. After a fresh `MAP_READY` generation exists, the program prepares only one
bounded segment, prints its exact approval token, and moves only if the same token is
pasted back. A successful segment is followed by an explicit stop and one automatic
capture. Coarse completion produces a verified schema-5 model; the fine coordinator then
starts with a new map publication and no inherited permit, approval, prepared segment or
coverage state. The top-level write-once chain is
`INIT -> COARSE_CHECKPOINT+ -> PREPARED -> FINE_START_CANDIDATE+ -> FINE_STARTED -> FINE_CHECKPOINT* -> FINE_COMPLETED`.
It binds each accepted checkpoint to its exact run-event boundary and science generation,
binds the schema-5/reference handoff separately, makes only the latest durable candidate
eligible for atomic fine-start publication, and seals the terminal fine coverage plus
strictly replayed final reconstruction. A candidate left by a crash is non-authoritative;
resume creates a new fine run and candidate. StopScan append and fine-start publication
share one canonical-root thread/process lock, so no fine event can cross the publication
linearization point unnoticed. A failed write or replay leaves the runtime
blocked and never transfers or completes active-runner authority. If fine safety refresh
has no expected source ID, `c` performs one explicitly stopped replenishment capture; it
does not start automatic continuous acquisition.

Resume only the explicitly named immutable experiment root:

```bash
uv run bbf scan run-unknown \
  --resume \
  --config configs/local.yaml \
  --output data/experiments/blade-placement-20260831-01-attempt-01 \
  --placement-id blade-placement-20260831-01 \
  --operator-id vale
```

Resume derives its phase solely from the verified top-level chain and rejects missing,
spliced, renumbered or source-mutated evidence. It never restores an old permit, approval,
prepared segment, motion freshness or controller authority. A sealed `FINE_COMPLETED`
experiment is reported read-only without connecting the robot or camera.

Every FoundationStereo logical cycle stores each retry in a fresh immutable
`attempt_<uuid>` directory. Failed or cancelled attempts remain diagnostic assets, while
only an atomically created `committed.json` selects the accepted attempt. Occupancy schema
7 deduplicates by a physical source identity derived from the session manifest, exact
view metadata, sequence and camera frame number—not by a reusable logical view label.
Schema-6 occupancy data is replay-only and can never become motion evidence.
Retries remain finite under the configured policy, and an accepted attempt is proven
against its raw session manifest, exact view metadata, physical frame identity,
stationarity evidence and derived science/safety authorities. Reusing a logical label is
not source proof.

Before `scan doctor --mode unknown` can pass, record physical geometry-science acceptance
from a completed declaration and copy its emitted path/ID pair into
`science_acceptance.path` and `science_acceptance.acceptance_id`:

```bash
uv run bbf safety record-science-acceptance \
  --declaration configs/science_acceptance.completed.json \
  --config configs/local.yaml \
  --output data/acceptance/geometry_science_001
```

The record must cover the configured working-distance and incidence-angle envelope and
binds the exact calibration, FoundationStereo source/checkpoint/model configuration,
foreground, coarse/fine reconstruction and selection policies. It authorizes no motion.
Doctor also requires one immutable runtime-timing acceptance whose path/ID matches all four
configured bounds: complete perception cycle, operator reposition interval, segment
execution, and stop/checkpoint/schema-5/fine handoff. Those measurements are enforced
again at their runtime boundaries; missing, stale, or
mismatched timing/science evidence remains blocking. The exact cold/warm trace workflow and
non-overwriting commands are defined in
[the supervised experiment protocol](docs/supervised-blade-experiment.md#7-科学验收与schema-5时序预算).

The command prints a second-terminal observer command. It normally resolves to:

```bash
uv run bbf supervise replay \
  --snapshot data/experiments/unknown_blade_001/live_timeline \
  --follow
```

This observer retains coarse and fine point-cloud history across the handoff but has no
robot, approval or stop capability. Its ES68+D435i triangles are reloaded from the exact
active collision manifest and STL files, and every displayed point-cloud source is
recorded in a disk-backed append-only hash chain. Source or mesh mutation therefore
blocks publication rather than silently changing the screen. `q`, `Ctrl-C`, an explicitly
stale map,
changed evidence, a failed continuous proof, or any unsupported runtime state
stops/blocks the run. Passing the
software doctor still does not constitute the hardware acceptance listed in
[the supervised experiment protocol](docs/supervised-blade-experiment.md).

After an ordered sequence has endpoint-feasible IK solutions, the offline preflight
interface below remains useful for audit. It requires a fresh `MAP_READY` occupancy asset
from the live FoundationStereo stop-and-capture coordinator and cannot be supplied by
`build-replay`.

```bash
uv run bbf safety preflight-path \
  --plan outputs/view_plan \
  --initialization outputs/initialization \
  --occupancy outputs/fresh_map_ready_occupancy \
  --coverage-plan outputs/next_view_plan_000 \
  --config configs/local.yaml \
  --output outputs/motion_preflight
```

When supplied with a separately produced valid fresh asset, the artifact binds and
verifies its source hashes, ES68+D435i motion-model contract,
FoundationStereo occupancy sequence/hash/freshness horizon, and configured ServoJ runtime
contract. The full reader re-verifies the raw session arrays, user stereo calibration,
rectification, official FoundationStereo source/checkpoint/configuration, hand-eye/FK
chain and active robot-depth rendering before issuing a process-local semantic
attestation. That attestation is bound through collision evidence, preflight, permit and
guarded execution; replay-only assets never receive one. The artifact re-derives every
leg when read and always stores `motion_authorized: false`. Approval eligibility also
requires both the mesh/FCL interval certificate and the independent robot-versus-voxel
interval certificate. Failure to prove an interval is reported as `UNKNOWN` and blocks;
a finite set of clear samples is never promoted to a continuous result. This command does
not connect to the robot. Passing an offline `build-replay` occupancy asset is intentionally
blocked because that asset is `STALE`.

Exactly one ordering source is accepted: repeated manual `--view-id` options or one
`--coverage-plan`. In the latter mode, preflight verifies the coverage artifact's source
view-plan identity, requires the exact stored order, binds its checksum, and repeats
those checks during readback. Automatic ordering remains a non-executable selection
proposal; it is not mesh, occupancy, or trajectory clearance.

The current implementation must not be interpreted as physical motion clearance. Robot
pixels removed by self-masking remain `UNKNOWN`, so the robot's own volume can block its
occupancy query. A narrowly scoped exception exists only for complete voxels inside an
immutable, physically accepted static-free AABB; `OCCUPIED` always wins, and these AABBs
must never overlap the blade, fixture, support or their possible envelope. Final model,
workcell, timing, sensing and guarded-motion acceptance remain mandatory.
