# Bilateral coverage and replanning

This page describes the conservative **proxy-stage** ledger retained for initial
planning compatibility. The later coarse-model workflow evaluates samples on the actual
curved front/back surfaces and TSDF mesh; see
[paper-derived curved reconstruction](curved-reconstruction.md). The two ledgers are
deliberately distinct so proxy evidence cannot be mistaken for final reconstruction
quality.

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

Proxy-ledger clouds are registered in `base` through synchronized joints, configured
joint-zero offsets, packaged ES68 FK, flange-primary `flange_T_left_ir`, and the calibrated
camera-stream transform. Controller `base_T_tcp` is validation-only and must pass the
configured residual gate. The later coarse-model workflow adds only bounded,
pose-regularized, same-side point-to-plane refinement with correspondence-count and
residual checks. It never runs unconstrained front/back ICP on the thin blade.

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

The update validates plan/initialization provenance and requires the same flange-primary
hand-eye matrix and FK authority used by the reference initialization. A stable identity derived from source
session, sequence, camera frame number, and view ID prevents counting one physical frame
twice—even if it was reconstructed once with native depth and once with stereo depth.

Export the next offline view set from any ledger version:

```bash
uv run bbf coverage next-plan \
  --ledger outputs/coverage_001 \
  --plan outputs/view_plan \
  --start-side front \
  --output outputs/next_view_plan_001
```

The resulting JSON references SHA-256-verified plan and coverage sources and separates
completed patches, remaining non-rejected views, and blocked patches. It also records a
deterministic **proxy-coarse-scan proposal**:

1. completed patches are removed as a hard coverage gate;
2. only `endpoint_feasible` candidates carrying a stored six-axis IK solution enter the
   ordered list;
3. the selected start side is completed first, row by row, with even rows traversed in
   ascending column order and odd rows in descending order;
4. the opposite side follows with the same snake rule; and
5. `geometry_only` views remain visible as `deferred_unverified_view_ids` and can never
   enter a motion preflight.

`front` is the default because proxy construction defines the initial camera-visible
surface as the proxy front. `--start-side back` is available when replanning after the
camera has already changed sides. Removing completed cells does not renumber rows or
reverse their original snake direction. Joint travel is not an optimization objective;
the proposal prioritizes coverage topology and endpoint reachability.

Every ordered entry preserves side, row, column, snake rank, measured occupied fraction,
and endpoint status. Reading the artifact derives all fields again from the hashed source
plan and ledger and rejects stale or edited ordering evidence. This first policy applies
only to the bilateral proxy coarse grid; the irregular curved/fin fine plan requires its
own region-aware ordering policy.

The ordering is a proposal, not a path-safety result. Bind it to the latest eligible
occupancy asset when creating the per-leg preflight:

```bash
uv run bbf safety preflight-path \
  --plan outputs/view_plan \
  --initialization outputs/initialization \
  --occupancy outputs/fresh_map_ready_occupancy \
  --coverage-plan outputs/next_view_plan_001 \
  --config configs/local.yaml \
  --output outputs/motion_preflight_001
```

The command rejects a coverage artifact from another view plan or any order that differs
from the hashed coverage proposal. Each leg is still independently subject to endpoint,
mesh, occupancy, freshness, and continuous-sweep gates. No reconstruction, coverage, or
preflight command connects to or moves the robot, and every artifact retains
`motion_authorized: false`.
