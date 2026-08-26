# Development log

Last updated: 2026-08-26

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
- Current verification: 105 tests, Ruff, package build, and Elite SDK import pass.
  Through commit `66947e4`, all earlier commits were pushed to GitHub `main`;
  the newer small commits are pushed together after this integration checkpoint.

## In progress

- Export the reduced coverage-driven next-view plan as a separate non-executable
  artifact, including completed, remaining, and blocked patch identities.

## Pending, in priority order

1. Run a real FoundationStereo checkpoint/CUDA smoke test; no compatible checkpoint or
   CUDA device is currently available in this workspace.
2. Compare FoundationStereo and native RealSense depth experimentally on the blade.
3. Add multi-view registration, bilateral coverage accounting, and coverage-driven
   replanning while preserving the front/back surface identity.
4. Add robot-body/workcell collision and continuous trajectory validation before any
   future motion-execution feature is considered.
5. Implement the thermal-camera adapter after its model and radiometric SDK are known.
