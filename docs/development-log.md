# Development log

Last updated: 2026-08-27

This log distinguishes verified implementation from pending work. Commit history is the
authoritative fine-grained record; this page records the experiment-facing state.

## Non-negotiable constraints

- Python 3.12 with `uv`; Elite SDK is installed from the local CPython 3.12 wheel.
- All currently exposed CLI commands remain read-only. A library-level Elite control
  backend now exists but is blocked by default configuration, offline preflight, exact
  operator confirmation, an expiring one-shot permit, and live revalidation.
- Every exported view plan has `motion_authorized: false`.
- Raw synchronized observations are immutable; derived products use separate outputs.
- Thermal capture remains an explicit disabled placeholder until hardware is selected.
- `latex/`, model checkpoints, data, and local configuration are not committed.

## Completed and verified

- D435i IR stereo calibration now separates responsive raw acquisition from offline
  ChArUco detection and solving. Every launch creates a unique append-only asset session;
  it records the copied board definition, device identity, synchronized frame provenance,
  raw pairs, detection overlays and accept/reject reasons, analysis attempts and final
  calibration under a SHA-256-bound manifest. The preview retains only the latest frame
  rather than accumulating GUI events, and completed sessions reject further writes.
- ES68 read-only hardware bring-up now follows HoloRobot's RTSI ownership contract:
  the status adapter subscribes to an explicit output-variable list and passes an empty
  input recipe, so observation cannot claim speed-slider or I/O write channels.
- Added a HoloRobot-style, manifest-driven ES68+D435i articulated collision template.
  It reserves independent STL slots for all seven collision links and the flange-mounted
  camera/bracket assembly, records units and transforms explicitly, materializes the
  calibrated ES68 chain into a Pinocchio/FCL URDF, and fails closed until the operator
  supplies every mesh, validates the flange attachment, and marks the manifest ready.
- A schema-bound fine-plan inspection command now verifies every persisted candidate's
  transform, camera-to-target distance, optical alignment, incidence, projection,
  coarse-cloud visibility, adaptive bounds, duplicate-pose status, and bilateral region
  presence. It atomically exports JSON/CSV plus region-coloured PLY, OBJ camera frusta,
  and a three-projection SVG. The optional PySide6 orbit viewer provides side/region
  filters, view/normal toggles, selection highlighting, and rejection details. Inspection
  explicitly reports robot feasibility as unverified and never changes
  `motion_authorized: false`.
- Fine-scan planning now uses a baseline-plus-region-adaptive distance policy. The base
  footprint is derived from the user-calibrated left-IR intrinsics, baseline distance,
  image margin, and utilization factor; no fixed 80x60 mm production fallback remains.
  Each true-surface or fin patch searches an explicit validated distance interval,
  records its selected distance and nominal footprint, and must pass whole-patch image/
  depth projection plus coarse z-buffer visibility gates. High-curvature, boundary,
  fin-root, and fin-rim regions prefer the closest feasible distance; flat main/fin faces
  prefer the baseline. Infeasible patches are visibility-split to a bounded depth and
  then fail closed. Schema-4 coarse-model artifacts persist the per-candidate evidence.
- Paper-derived coarse-model reconstruction now consumes multiple existing
  pose-registered D435i views, assigns immutable front/back membership from achieved
  camera centres, voxel-fuses each side, and applies robust point-to-plane residual
  refinement with robot-pose regularization, hard correction bounds, and no cross-side
  correspondences.
- True curved-surface planning now implements improved Angle Criterion boundary evidence,
  supported outer-contour ordering, four topological junctions, robust endpoint-consistent
  3D B-splines for root/trailing/tip/leading boundaries, equal-arc sampling, and an
  invertible Coons-grid irregular surface domain with boundary snapping. Fit/fold gates
  have explicit recorded fallback or fail-closed behaviour. Front/back use a shared
  conservative base grid; each populated
  patch then receives a PCA OBB centre, spherical-histogram main normal, and optional
  curvature-adaptive split. Fine views use measured main normals and remain non-executable.
- The photographed specimen's fixed topology is now explicit: robust per-side main-height
  fitting and height/normal-seeded 3D region growth require one thin fin on the front and
  one on the back. Fin points are removed before the paper boundary fit. Each retained
  component has independently persisted face, attachment-root, and free-rim regions;
  face-normal, root-bisector, and rim views; independent coverage gates; and measured-fin
  thickness protection in TSDF. Missing fins, multiple significant protrusions, non-thin
  components, and sub-voxel protected bands fail closed.
- Bilateral sparse projective TSDF uses a measured-thickness-protected truncation band,
  integrates front/back independently, and extracts a triangle mesh with a pure NumPy
  marching-tetrahedra fallback. Calibrated pixel/intrinsic/pose metadata enables the
  optional locked Open3D scalable backend when installed.
- Real-surface coverage replaces proxy-plane bins at the coarse-model stage: each patch
  records sample coverage, residual RMSE, local-normal consistency, curvature, and
  explicit quality-gate reasons; four edge-region completion ratios and TSDF mesh
  boundary/watertight evidence are reported separately.
- `bbf reconstruct coarse-model` validates common hand-eye provenance, runs the full
  fusion/partition/view/TSDF/quality chain, and atomically writes source-bound,
  SHA-256-verified arrays and metadata with `motion_authorized: false`.

- HoloRobot-aligned ES68/D435i eye-in-hand workflow: the exact 709-pose calibrated ES68
  FK and flange-to-RTSI-TCP validation offset are separately packaged under `es68`;
  synchronized PySide6 capture uses only raw D435i `infrared/1` and user-calibrated
  intrinsics, records complete ChArUco/robot/timing evidence, gates FK/TCP agreement,
  solves `flange_T_left_ir` with Daniilidis plus joint SE(3) LM/BA, and exports a
  flange-primary schema-2 artifact with input hashes and before/after quality metrics.
- PySide6 raw D435i IR stereo-calibration workflow using the stored 14x9 ChArUco target:
  synchronized Y8 capture without factory IR calibration access, offline independent
  Zhang initialization, joint stereo bundle adjustment, epipolar metrics, selectable
  radial2/Brown5/Rational8 distortion models, held-out automatic model comparison, and
  user-calibration YAML export/load with resolution checks.
- Read-only Elite RTSI state acquisition and controller MDH export.
- D435i synchronized infrared/native-depth capture with calibration snapshots.
- Atomic schema-v2 session writer and validated reader.
- Native-depth point-cloud initialization, conservative thin-blade proxy, bilateral
  partitioning, candidate generation/filtering/scoring, and non-executable plan export.
- Offline CS68 KDL endpoint IK validation with captured seed joints.
- Calibrated D435i stereo rectification with explicit frame-chain transforms.
- Official FoundationStereo source pinned as a Git submodule.
- Lazy FoundationStereo adapter with no implicit EdgeNeXt or DINOv2 network download;
  inference scale is converted back to full-resolution disparity pixel units.
- Rectified left/right valid-region filtering and disparity-to-metric-depth conversion.
- Atomic, checksummed stereo inference artifacts and `bbf stereo infer-session`.
- Offline Park-Martin/Tsai/Horaud/Andreff/Daniilidis initial solving, motion
  observability gates, fixed-target closure RMSE, atomic artifacts, and CLI integration.
- Identified ChArUco detection from raw stored left-IR frames, positive-depth IPPE pose
  selection, planar-ambiguity/reprojection gates, automatic sample extraction, and
  durable rejection reasons.
- FoundationStereo-depth proxy initialization with source-identity checks, correct
  `base_T_left_rectified` geometry, and an end-to-end raw-session-to-plan integration
  path. Initialization schema 5 records depth source/projection frame and reads schema 4.
- Bilateral per-patch coverage grids, independent front/back evidence, incomplete versus
  blocked replanning state, and immutable checksummed seed-coverage artifacts.
- Native and FoundationStereo pose-registered view artifacts, with source identity,
  checksummed clouds/masks, hand-eye provenance, duplicate-frame prevention, and
  immutable coverage-ledger append support.
- Coverage-driven next-view artifacts that cryptographically bind the source plan and
  ledger, re-derive their contents on read, distinguish completed/remaining/blocked
  patches, and explicitly forbid motion.
- Calibrated paired native/stereo depth comparison in `left_rectified`, including
  z-buffered native reprojection, shared-valid-pixel metrics, checksummed arrays, source
  verification, and explicit non-ground-truth interpretation.
- Manifest-driven depth aggregation with duplicate-frame rejection, view-balanced and
  pixel-pooled metrics, plus retained front/back and incidence-angle strata.
- Initialization schema 6 adds SHA-256, dtype, and shape manifests for base clouds,
  pixel provenance, and masks while retaining schema 4/5 read compatibility.
- Achieved-pose experiment labeling composes robot, hand-eye, and rectification
  transforms to derive proxy side and incidence; ambiguous mid-plane/away-facing views
  are rejected, and generated manifests bind the fixed initialization metadata.
- Correct Elite KDL IK orientation encoding: the vendor plugin consumes roll/pitch/yaw,
  which is intentionally distinct from the controller TCP rotation-vector encoding.
- Exact vendor-convention MDH link origins, fail-closed capsule/workcell geometry,
  joint-limit checks, continuous joint-space sampling, explicit ordered view-sequence
  validation, and immutable reports that always forbid motion.
- `bbf doctor` collision-readiness diagnostics enumerate missing radii, tool geometry,
  joint limits, and required workcell obstacles before path validation is attempted.
- View-plan schema 2 cryptographically binds endpoint-feasible IK solutions to their
  controller-specific MDH artifact; safety validation rejects legacy or mismatched
  kinematics provenance while retaining schema-1 geometry-only read compatibility.
- Copied HoloRobot CS68 YAML/URDF/STL resources, D435i wrist collision mesh, matched
  YAML/Pinocchio forward kinematics, and Pinocchio/FCL self-collision/path sampling.
- HoloRobot-aligned Elite Dashboard/RTSI/EliteDriver lifecycle, RPY TCP convention,
  point trajectories, SpeedJ, ServoJ prewarm/hold/streaming, stop, and safety faults.
- Conservative linear-joint motion preflight using copied velocity limits, plus exact
  preflight-hash confirmation, expiring one-shot execution permits, live-start checks,
  and immediate collision revalidation. No motion command is exposed through the CLI.
- Immutable ordered view-sequence motion-preflight artifacts bind plan/initialization
  hashes, re-derive Pinocchio/FCL and ServoJ evidence on read, and are generated offline
  by `bbf safety preflight-path` with `motion_authorized: false`.
- Mesh motion preflight now persists calibrated `base_T_tcp` goals and sequence cost;
  configured workcell AABBs are clearance-expanded hpp-fcl geometry checked against
  all copied CS68/D435i meshes. Required missing obstacle geometry is blocking.
- Current local verification: 206 tests, Ruff, offline wheel build, packaged robot-resource
  audit, and Elite/Pinocchio/hpp-fcl/trimesh imports pass. Curved reconstruction is
  software-verified on deterministic data and remains pending physical hardware validation.

## In progress

- Robot-stack migration to the pinned HoloRobot implementation is active. Model,
  self/workcell collision, control, ServoJ trajectory, guarded-execution, and ordered
  tool-goal preflight artifact layers are copied/adapted. Hardware acceptance and
  traversal-order optimization remain.
- The migration sequence and safety boundary are recorded in
  `docs/robot-stack-migration.md`. Existing MDH/capsule code remains temporarily for
  artifact compatibility and will not receive new motion functionality.
- Real D435i/ES68 coarse scans are not yet available, so the new curved reconstruction
  is verified on deterministic synthetic bilateral blade data. Hardware threshold
  tuning, edge occlusion acceptance, and Open3D-versus-NumPy mesh comparison remain.

## Pending, in priority order

1. Record overlapping front/back coarse blade views, run `bbf reconstruct coarse-model`,
   and tune footprint, Angle Criterion, curvature, ICP, TSDF, and quality gates against
   physical dimensional references.
2. Run a real FoundationStereo checkpoint/CUDA smoke test; no compatible checkpoint or
   CUDA device is currently available in this workspace.
3. Collect paired blade observations and compare FoundationStereo with native RealSense
   depth using the offline evaluator and aggregate report.
4. Optimize front/back traversal order using the persisted motion cost, then perform a
   separately approved known-safe-pose hardware acceptance before considering an
   interactive execution command. Add occupancy collision for unmodeled objects later.
5. Implement the thermal-camera adapter after its model and radiometric SDK are known.
