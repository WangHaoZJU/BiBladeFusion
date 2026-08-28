# Development log

## 2026-08-28 — coverage-derived coarse-scan ordering and preflight binding

- Coverage-plan schema 2 now turns incomplete proxy patches into a deterministic,
  non-executable traversal proposal. It finishes the selected proxy side first and
  applies a stable row-wise snake using the original row parity, so deleting completed
  cells cannot reverse a later row during replanning.
- Coverage and reachability remain separate hard gates. Only `endpoint_feasible` views
  carrying persisted six-axis joint solutions enter `ordered_view_ids`;
  `geometry_only` views are retained as `deferred_unverified_view_ids`, while rejected
  incomplete patches remain blocked. Occupied fraction is persisted as audit evidence
  but does not reorder the path, and joint travel is deliberately not an objective.
- `bbf safety preflight-path` accepts exactly one ordering source: repeated manual
  `--view-id` values or a coverage-plan artifact. Automatic mode verifies the source
  view-plan identity, requires exact ordered-ID equality, includes the coverage manifest
  in the SHA-256 source chain, and repeats the checks during artifact readback.
- The order is still only a proposal. It does not authorize motion or prove the
  front-to-back leg. Mesh and robot-versus-voxel paths remain independently fail-closed
  because their current bounded-step checks do not constitute continuous swept-volume
  evidence.
- Closed the increment with `423 passed`, repository-wide Ruff, bytecode compilation,
  and whitespace-integrity checks.

## 2026-08-28 — active ES68+D435i collision assembly and offline inspector

- Activated the current D435i-only ES68 collision manifest from the matching HoloRobot
  model. The seven arm meshes remain in their URDF link frames; the payload uses the
  documented identity flange joint and the `depth_camera_mount.stl` collision origin
  `[-0.0505, -0.031815, 0] m` with zero rotation. The eight copied STL files are byte-for-
  byte identical to their HoloRobot sources and retain metre units.
- Added a completely offline PySide6/Qt3D assembly inspector. It renders the exact eight
  collision meshes, drives them from the same packaged ES68 forward kinematics and joint-
  zero offsets used by safety code, provides six joint controls, selectable STL layers,
  orbit/zoom controls, link positions and mesh-loader status. The command has no robot-IP
  option, opens no device backend and contains no motion or authorization path.
- Exercised the production Pinocchio/FCL chain with the active manifest: all eight
  geometries loaded, 20 filtered collision pairs were constructed, and zero plus three
  nonzero diagnostic poses were clear under the configured 10 mm policy. These discrete
  checks establish software loading and assembly consistency; they are not a continuous
  swept-path proof or hardware dimensional acceptance.
- Audited mesh quality separately. The source set is intentionally detailed (about
  674,000 triangles); several meshes are not watertight and five degenerate faces remain,
  although hpp-fcl 2.4.4 loads the complete set. Keep this provenance set for acceptance
  and introduce simplified collision meshes later only under a new model identity with
  conservative-envelope regression checks.
- Completed a real desktop-display launch and operator visual check, then closed the
  change with the complete regression suite (`414 passed`), repository-wide Ruff,
  bytecode compilation and lockfile consistency checks.

## 2026-08-28 — FK-authority native-depth re-evaluation

- Reprocessed the five preserved real ES68/D435i sessions with the current schema-2
  `FK(joints + zero offsets) · flange_T_left_ir · left_ir_T_depth` authority chain;
  controller TCP remained validation-only.
- The new immutable report
  `data/validations/native_overlap_20260828_fk_authority_v2` passed strict full
  recomputation. Across the four comparisons, median error was 1.220–1.423 mm, RMSE
  2.069–2.469 mm, P95 4.205–4.992 mm, and 5 mm agreement 95.03–97.36%, over a
  188.01 mm/23.683 degree pose span. ICP remained diagnostic-only.
- Preserved the schema-1 TCP-primary report unchanged and added a separately named
  integrity-only legacy replay reader; legacy values are never promoted to current
  FK-authority evidence.

## 2026-08-28 — unknown-blade occupancy safety and supervisory replay

- Added stop-and-capture occupancy construction from stored FoundationStereo depth in
  calibrated `base` coordinates. Mapping requires left-right consistency evidence, its
  explicitly non-probabilistic consistency-score array, a bounded depth range, synchronized
  ES68 joints, the accepted flange-primary left-IR hand-eye transform, and at least three
  geometrically independent settled views. Each new view must differ from every prior
  view by 20 mm of camera-centre translation or 5 degrees of optical-axis angle by
  default; changing only its identifier fails before ray integration. Calibrated FK is
  the mapping pose authority; synchronized RTSI
  TCP is validation-only, with both poses and their residuals retained and independently
  reproduced during asset readback.
- Added a final-model ES68+D435i renderer and conservative depth-consistent robot
  self-mask. Measurements clearly in front of the rendered robot are retained as
  possible unknown surfaces; matching or farther measurements are removed so a stereo
  dropout cannot ray-clear through the robot. Removed pixels and their occluded rays
  remain `UNKNOWN` rather than being cleared as free space.
- Added immutable sparse occupancy assets with `FREE`, `OCCUPIED`, and implicit
  `UNKNOWN` voxels, an explicit `UNMAPPED/MAPPING/MAP_READY/STALE` lifecycle, per-frame
  quality arrays, hash-chained evidence, mapping-context binding, and read-time
  reproduction of masks, integration and snapshots. Out-of-grid and unknown space are
  fail-closed. A voxel needs three independent FREE votes by default, while OCCUPIED
  remains dominant; map freshness starts at the first frame of a complete rebuild cycle.
  Occupancy asset schema 6, snapshot format 4, and mapping-context schema 4 additionally
  retain the supporting camera poses, FK flange pose, predicted/observed TCP poses and
  flange-primary camera chain. The reader re-runs packaged ES68 FK from every stored
  joint vector before accepting them.
- Added strict source-to-motion semantic verification. The full occupancy reader
  reproduces raw-session integrity, user stereo calibration and rectification, official
  FoundationStereo source/checkpoint/configuration, self masking, integration and active
  robot geometry before issuing a typed attestation. Replay has no attestation. The
  occupancy checker, motion-preflight schema 5, one-shot permit and guarded executor bind
  that exact proof; protocol fakes, mutable snapshots and metadata changes fail closed.
- Added occupancy-aware motion preflight. The artifact binds the occupancy sequence,
  content hash and freshness horizon together with the complete ES68+D435i motion-model
  contract and ServoJ runtime configuration. Offline `occupancy build-replay` output is
  deliberately sealed `STALE`, so it can exercise storage and visualization but cannot
  satisfy motion preflight.
- Added a self-contained supervisory snapshot bridge and PySide6 replay console for the
  historical robot/camera scene, occupancy, current/fused blade point clouds, sensor
  evidence, copied provenance manifests and blocking events. Exact collision meshes are
  shown only after the active final model reproduces the historical geometry hash, and
  planned TCP targets only after canonical preflight replay; no continuous actual TCP
  trace is claimed. The GUI is read-only, always labels replay as `REPLAY/BLOCKED`, and
  exposes no approval or motion command.
- Preserved the physical-release boundary. A missing or unready final ES68+D435i STL
  manifest fails closed. Robot-self-masked volume remains unknown and can block the
  current bounding-sphere occupancy query; robot/environment paths are still evaluated
  at discrete joint samples rather than by an exact swept mesh. The native real-time
  coordinator has not yet been implemented or hardware-verified. These are blocking
  items, not merely performance optimizations.
- Sealed the public Elite motion methods behind the guarded executor's private capability.
  Even that path re-derives the exact ServoJ stream and rechecks every command segment;
  the missing continuous swept-mesh and swept-occupancy proofs still stop it before driver
  preparation.
- Verified this increment with the complete repository regression suite (`406 passed`),
  repository-wide Ruff checks, bytecode compilation, CLI smoke checks, and a locked-
  dependency consistency check.

## 2026-08-27 — native-depth validation infrastructure and legacy baseline

- Added `evaluate native-overlap`. The current schema-2 implementation validates
  synchronized D435i native depth transformed by the authoritative
  `FK(joints + zero offsets) · flange_T_left_ir · left_ir_T_depth` chain. Symmetric
  projective residuals reject depth edges, invalid pixels, occlusions, and field-of-view
  loss without applying registration corrections.
- Added explicit thresholds for projected support, same-surface inliers, median/RMSE/P95,
  5 mm agreement, and camera-pose observability. A deliberately wrong rotating hand-eye
  offset is covered by regression tests and must fail.
- Added bounded point-to-plane ICP as diagnostic evidence only. The correction cannot
  affect primary metrics, pass/fail, overlay points, or active calibration files.
- Added append-only, fully recomputable assets with source/config/hand-eye hashes,
  per-pair residual arrays, CSV metrics, a coloured base-frame PLY, and three-view PNG.
- The first five-view ES68/D435i run passed without ICP: median errors 1.220–1.424 mm,
  RMSE 2.070–2.470 mm, P95 4.205–4.993 mm, 5 mm agreement 95.02–97.36%, over a
  188.01 mm/23.683 degree pose span. Evidence is retained under
  `data/validations/native_overlap_20260827_static_v1`. That retained report is schema 1
  and explicitly used the legacy TCP-primary
  `base_T_tcp · tcp_T_left_ir · left_ir_T_depth` chain. Its numbers are a historical
  baseline, not validation of the current FK-authority path; the raw sessions must be
  reprocessed to a separate schema-2 output before making that claim.

## 2026-08-27 — native-depth validation acquisition override

- Added `--emitter/--no-emitter` to synchronized `acquire snapshot` and standalone
  `camera capture`. The override is command-scoped, leaves the hand-eye/stereo default
  configuration unchanged, and is preserved in synchronized session configuration
  snapshots so native-depth experiments remain reproducible.

## 2026-08-27 — Park+BA ES68/D435i hand-eye calibration closure

- Rebuilt the PySide6 hand-eye application as an idle-first, operator-controlled
  workflow: devices connect only after **Start**, `C` accepts one synchronized pose,
  Backspace recoverably excludes the last pose, and raw left IR plus detected ChArUco
  corners remain visible side by side.
- Isolated acquisition from preview analysis with a latest-frame mailbox and a reused
  detector. Slow corner processing can drop preview frames but cannot build an event
  backlog or change which full-resolution frame is atomically saved.
- Made Park-Martin the default initializer and retained the HoloRobot-aligned joint
  LM/BA refinement of `flange_T_left_ir` and fixed `base_T_target`, with live
  motion-observability, pose-novelty, synchronization, PnP, and ES68 FK/TCP gates.
- Locked robot-pose semantics to the HoloRobot ES68 reference: recorded joint angles
  drive the copied 709-pose calibrated FK and produce solver `base_T_flange`; RTSI
  `base_T_tcp` is validation-only. Persisted samples/results state this role explicitly,
  and a regression test proves changing controller TCP observations cannot change the
  solved transform.
- Added a strict held-out stage using at least five new poses. It evaluates the fixed
  candidate with board-closure and corner-reprojection metrics without refitting; only
  a passing report is atomically published to
  `data/calibrations/es68_left_ir_hand_eye_active.yaml`.
- Added `calibration hand-eye-validate-gui` for later supplemental evidence against an
  already completed schema-2 result. The new session hash-binds the unchanged candidate,
  stereo calibration and target, exposes no training/solve controls, never invokes Park
  or BA, and fails before hardware connection if provenance differs.
- Added unique, append-only hand-eye digital-asset sessions that copy and hash-bind the
  ChArUco target, D435i stereo calibration, packaged HoloRobot ES68 kinematics and
  flange/TCP offset, settings, raw/audit images, samples, candidate, validation attempts,
  and final result. A disconnected nonempty run is sealed instead of being mixed into a
  new session.
- Added regression coverage for bounded preview delivery and both pass/fail fixed-
  parameter validation geometry. The offline solver now also defaults to Park+BA but
  deliberately writes only a candidate; GUI-held-out validation controls publication.

## 2026-08-27 — independent D435i IR stereo validation

- Added the `calibration stereo-validate-gui` workflow with an idle startup window and
  three explicit operator steps: connect, save a synchronized hold-out pair, and run
  fixed-parameter offline validation.
- Added a validation-only digital asset schema that copies and SHA-256 binds the
  ChArUco target and exact stereo calibration, atomically appends raw Y8 pairs, records
  D435i identity/timestamps, and explicitly certifies that no calibration refit was
  performed.
- Added offline ChArUco detection, calibrated image/point rectification, horizontal
  epipolar-line overlays with matched corner colours, per-pair evidence, and aggregate
  vertical-disparity RMSE/P95/max, monocular reprojection RMSE, and stereo-transfer
  RMSE metrics with recorded pass/fail thresholds.
- Added `calibration stereo-validate-assets` for processing a preserved session after
  acquisition, plus unit coverage for fixed-input provenance, successful ideal
  geometry, checksum tamper rejection, and calibration/stream resolution mismatch.

Last updated: 2026-08-28

This log distinguishes verified implementation from pending work. Commit history is the
authoritative fine-grained record; this page records the experiment-facing state.

## Non-negotiable constraints

- Python 3.12 with `uv`; Elite SDK is installed from the local CPython 3.12 wheel.
- All currently exposed CLI commands remain non-moving with respect to the robot; commands
  that acquire or derive data write only their declared digital assets. A library-level
  Elite control backend exists but is blocked by default configuration, offline preflight,
  exact operator confirmation, an expiring one-shot permit, and live revalidation.
- Every exported view plan has `motion_authorized: false`.
- Raw synchronized observations are immutable; derived products use separate outputs.
- Thermal capture remains an explicit disabled placeholder until hardware is selected.
- `latex/`, model checkpoints, data, and local configuration are not committed.

## Completed and verified

- D435i IR stereo calibration now separates responsive raw acquisition from offline
  ChArUco detection and solving. Every operator-started run creates a unique append-only
  asset session;
  it records the copied board definition, device identity, synchronized frame provenance,
  raw pairs, detection overlays and accept/reject reasons, analysis attempts and final
  calibration under a SHA-256-bound manifest. The preview retains only the latest frame
  rather than accumulating GUI events, and completed sessions reject further writes.
  The GUI starts idle; only an explicit operator click on **开始** connects the camera,
  creates the session and starts sample statistics at zero.
- A successful D435i IR solve now atomically publishes the solver-accepted result to
  `data/calibrations/d435i_ir_active.yaml`, the fixed path used by the default runtime
  configuration. All later calibrated capture paths therefore consume the latest
  completed user result without manual path editing. Missing user calibration fails
  closed; the RealSense adapter no longer falls back to factory IR intrinsics or stereo
  extrinsics. A live D435i capture verified that the default path returned the published
  left/right focal lengths and 49.990 mm user-calibrated baseline exactly.
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
  solves `flange_T_left_ir` with Park-Martin plus joint SE(3) LM/BA, validates the fixed
  candidate on new poses, and publishes a flange-primary schema-2 artifact only after
  the held-out gates pass.
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
- Offline ES68 KDL endpoint IK validation with captured seed joints.
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
- Initialization schema 7 adds source-pose authority and TCP-validation evidence on top
  of the SHA-256, dtype, and shape manifests for base clouds, pixel provenance, and masks;
  the reader retains schema 4/5/6 compatibility.
- Achieved-pose experiment labeling composes robot, hand-eye, and rectification
  transforms to derive proxy side and incidence; ambiguous mid-plane/away-facing views
  are rejected, and generated manifests bind the fixed initialization metadata.
- Correct Elite KDL IK orientation encoding: the vendor plugin consumes roll/pitch/yaw,
  which is intentionally distinct from the controller TCP rotation-vector encoding.
- Exact vendor-convention MDH link origins, fail-closed capsule/workcell geometry,
  joint-limit checks, bounded-step discrete joint-space sampling, explicit ordered view-sequence
  validation, and immutable reports that always forbid motion.
- `bbf doctor` collision-readiness diagnostics enumerate missing radii, tool geometry,
  joint limits, and required workcell obstacles before path validation is attempted.
- View-plan schema 3 cryptographically binds endpoint-feasible IK solutions to their
  controller-specific MDH artifact and six joint-zero offsets; safety validation rejects
  legacy or mismatched endpoint provenance while retaining older geometry-only plans for
  read-only compatibility.
- Adapted the pinned HoloRobot structural resources (whose upstream package paths retain
  `cs68` identifiers) to the calibrated ES68 chain and D435i wrist geometry. Development
  fixtures exercise YAML/Pinocchio FK and FCL sampling, while production resolution now
  requires the separately accepted final ES68+D435i manifest and never falls back to the
  upstream-labelled meshes.
- HoloRobot-aligned Elite Dashboard/RTSI/EliteDriver lifecycle, RPY TCP convention,
  point trajectories, SpeedJ, ServoJ prewarm/hold/streaming, stop, and safety faults.
- Conservative linear-joint motion preflight using copied velocity limits, plus exact
  preflight-hash confirmation, expiring one-shot execution permits, live-start checks,
  and immediate collision revalidation. No motion command is exposed through the CLI.
- Immutable ordered view-sequence motion-preflight schema-5 artifacts bind plan,
  initialization, occupancy, and motion-model hashes and re-derive the fail-closed report
  on read. The production path currently stops at missing continuous swept-mesh evidence
  before ServoJ generation; diagnostic-only library overrides do not create approval
  evidence. `bbf safety preflight-path` always writes `motion_authorized: false`.
- Mesh motion preflight persists calibrated `base_T_tcp` goals and bounded-step sequence
  cost evidence;
  configured workcell AABBs are clearance-expanded hpp-fcl geometry checked against
  the resolved ES68+D435i model. Missing required production geometry is blocking.
- The 2026-08-27 software baseline passed 206 tests, Ruff, an offline wheel build, the
  packaged-resource audit, and Elite/Pinocchio/hpp-fcl/trimesh import checks. Curved
  reconstruction remains software-verified on deterministic data and pending physical
  hardware validation; later commits add their own regression evidence.

## In progress

- Robot-stack migration to the pinned HoloRobot implementation is active. Model,
  self/workcell collision, control, ServoJ trajectory, guarded-execution, and ordered
  tool-goal preflight artifact layers are copied/adapted. FoundationStereo occupancy
  storage, conservative queries and replay visualization are implemented, but the live
  stop-and-capture coordinator is not implemented; hardware acceptance remains.
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
4. Install and accept the final ES68+D435i STL manifest, then close the self-masked
   `UNKNOWN`-volume problem without ray-clearing occluded space. Implement and validate
   two independent continuous proofs for every segment: swept ES68+D435i mesh/FCL
   clearance and swept robot-versus-voxel occupancy clearance. Discrete bounding-sphere
   samples remain diagnostic only.
5. Implement and hardware-verify the native stop-and-capture coordinator that publishes
   fresh `MAP_READY` snapshots, invalidates plans on every update, and preserves the
   required freshness horizon through execution. Offline `build-replay` must remain
   `STALE/BLOCKED`.
6. Hardware-validate the deterministic coverage-derived sequence from separately
   approved known-safe poses. Joint-motion cost may later be evaluated only as a
   non-safety tie-breaker, not as a reachability or clearance substitute.
7. Implement the thermal-camera adapter after its model and radiometric SDK are known.
