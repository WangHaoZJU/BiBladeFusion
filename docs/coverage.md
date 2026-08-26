# Bilateral coverage and replanning

Coverage is tracked against the fixed conservative proxy and the exact patch IDs in an
offline view plan. Front and back evidence are never merged. The observed side is
selected from the calibrated camera center in the proxy frame; a camera too close to the
proxy mid-plane is rejected as ambiguous.

Within each surface patch, BiBladeFusion accumulates a square grid of point counts. A
patch becomes complete only when at least `coverage.completed_fraction` of its bins each
contain `coverage.minimum_points_per_bin` points. Points must lie inside the planned
surface extents and within `maximum_surface_distance_m` of the corresponding proxy face.
Views with too few usable points are rejected instead of counting as empty evidence.

Create the first immutable ledger from the initial pose-registered cloud:

```bash
uv run bbf coverage seed \
  --plan outputs/view_plan \
  --initialization outputs/initialization \
  --config configs/local.yaml \
  --output outputs/coverage_000
```

The artifact stores integer bin counts in a SHA-256-verified array and records its source
plan, source initialization, configuration, observation IDs, front/back completion, and
`motion_authorized: false`. A new output path is required for every future update.

## Registration assumption

Current clouds are registered in `base` through synchronized `base_T_tcp`, the
quality-gated `tcp_T_left_ir`, and the calibrated camera-stream transform. This preserves
metric provenance and avoids unconstrained ICP on a thin, locally smooth blade, where
sliding or front/back convergence is a serious failure mode. Pairwise geometric
refinement will only be added with explicit correspondence/uncertainty checks.

For every later captured view, first create a pose-registered artifact. Choose the depth
source used by that experiment:

```bash
uv run bbf reconstruct native-depth \
  --session data/<later-session> --view-id front_r00_c01 \
  --mask data/<later-mask>.npy --config configs/local.yaml \
  --output outputs/registered_front_r00_c01

uv run bbf reconstruct stereo-depth \
  --session data/<later-session> --view-id front_r00_c01 \
  --stereo outputs/<later-stereo> --mask data/<later-rectified-mask>.npy \
  --config configs/local.yaml --output outputs/registered_front_r00_c01_stereo
```

Append exactly one registered artifact to a new immutable ledger:

```bash
uv run bbf coverage add \
  --ledger outputs/coverage_000 \
  --plan outputs/view_plan \
  --initialization outputs/initialization \
  --view outputs/registered_front_r00_c01 \
  --output outputs/coverage_001
```

The update validates plan/initialization provenance and requires the same hand-eye
matrix used by the reference initialization. A stable identity derived from source
session, sequence, camera frame number, and view ID prevents counting one physical frame
twice—even if it was reconstructed once with native depth and once with stereo depth.

Export the next offline view set from any ledger version:

```bash
uv run bbf coverage next-plan \
  --ledger outputs/coverage_001 \
  --plan outputs/view_plan \
  --output outputs/next_view_plan_001
```

The resulting JSON references SHA-256-verified plan and coverage sources and separates
completed patches, remaining non-rejected views, and blocked patches. Reading it derives
the selection again from its sources and rejects stale or edited summaries. No
reconstruction or coverage command moves the robot.
