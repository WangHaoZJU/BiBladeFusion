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

The current CLI seeds coverage from the initialization view. Persisting and adding later
registered observations is the next increment; no coverage command moves the robot.
