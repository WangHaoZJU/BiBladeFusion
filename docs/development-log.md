# Development log

Last updated: 2026-08-27

This log distinguishes verified implementation from pending work. Commit history is the
authoritative fine-grained record; this page records the experiment-facing state.

## Non-negotiable constraints

- Python 3.12 with `uv`; Elite SDK is installed from the local CPython 3.12 wheel.
- Elite CS68 access remains read-only. No implemented command loads a robot task,
  releases brakes, or sends a motion command.
- Every exported view plan has `motion_authorized: false`.
- Raw synchronized observations are immutable; derived products use separate outputs.
- Thermal capture remains an explicit disabled placeholder until hardware is selected.
- `latex/`, model checkpoints, data, and local configuration are not committed.

## Completed and verified

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
- Offline Park-Martin/Tsai/Horaud/Andreff/Daniilidis eye-in-hand solving, motion
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
- Current verification: 127 tests and Ruff pass. Through commit `3ea4e83`, all prior
  commits, package build, and Elite SDK import were verified and pushed.

## In progress

- No partial implementation remains at this checkpoint. The next safety increment is
  CAD/controller-trajectory validation beyond the conservative capsule prefilter.

## Pending, in priority order

1. Run a real FoundationStereo checkpoint/CUDA smoke test; no compatible checkpoint or
   CUDA device is currently available in this workspace.
2. Collect paired blade observations and compare FoundationStereo with native RealSense
   depth using the offline evaluator and aggregate report.
3. Validate configured capsule geometry against CS68/workcell CAD, then add exact-mesh,
   controller-timed trajectory, singularity, dynamics, and stopping-distance checks
   before any future motion-execution feature is considered.
4. Implement the thermal-camera adapter after its model and radiometric SDK are known.
