# Paper-derived curved-blade reconstruction

This workflow is a software adaptation of the coarse-model-to-fine-view method from
`航空叶片形貌高精度结构光扫描视点规划.pdf` for a D435i/ES68 bilateral thin-blade
experiment. It is an offline geometry workflow and cannot command the robot. The complete
chain is currently regression-verified on deterministic synthetic bilateral single-fin
data; no real-blade accuracy claim is made without the hardware experiments described
below.

## Pipeline and paper mapping

1. **Coarse-view collection and fusion.** Each input is an existing immutable
   pose-registered view in `base`. The first `--view` is the deterministic front-side
   anchor; all camera centres are classified relative to that oriented blade normal.
   Voxel fusion is followed by robust point-to-plane residual refinement, regularized
   toward the robot/hand-eye pose. Corrections exceeding the configured translation or
   rotation bound are rejected. Correspondences are never formed across blade sides.
2. **One fin per physical side.** Before main-outline extraction, a robust quadratic
   height field is fitted from points whose normals agree with the dominant main blade.
   Height-plus-normal seeds grow through a 3D voxel-neighbour graph to the attached
   root and free rim. The largest supported thin component is retained on each side;
   missing fins or a second significant protrusion fail closed in the specimen-default
   `required_single_per_side` mode. Fin points are excluded from all main-boundary fits.
3. **Improved Angle Criterion boundary evidence.** Neighbours are evaluated in the
   blade PCA plane. The largest empty polar angle around each sample is compared with
   `angle_criterion_threshold_deg` (90 degrees by default), matching the paper's
   interior/boundary distinction while working on unordered depth clouds. Isolated
   candidates are removed by nearest-neighbour support and angular bins retain the
   outermost closed contour rather than internal holes or noise spikes. Each cyclic
   corner-to-corner path then rejects points whose perpendicular chord residual is not
   supported by its neighbours.
4. **Four junctions and four fitted boundaries.** Root-leading, root-trailing,
   tip-trailing, and tip-leading junctions are identified as topological extrema. The
   ordered contour is split into root, trailing edge, tip, and leading edge. Every part
   is fitted in 3D with an endpoint-consistent B-spline using chord-length parameters,
   Huber iteratively reweighted least squares, and a second-difference smoothness term.
   Fit RMSE and inlier fraction are quality gates; fitted controls, knots, source
   contour, corners, and metrics are persisted in the coarse-model artifact.
5. **Curve-driven exact surface domain and equal-arc partitioning.** The four fitted
   curves define the actual irregular outline. They are sampled by cumulative 3D arc
   length and blended as a Coons parameter grid whose four sides exactly reproduce the
   root, trailing edge, tip, and leading edge. A fold-over Jacobian gate rejects invalid
   domains; nearest-grid inversion maps each measured point into normalized major/minor
   coordinates and snaps supported fitted-boundary samples to the exact domain edge.
   Thus a rectangular proxy no longer invents material outside the measured outline.
   Partition counts use the larger front/back measured arc length and a shared row/column
   grid, preserving conservative footprint coverage and bilateral base-cell identity.
6. **Controlled fallback.** If boundary support or fit quality is insufficient,
   `boundary_allow_fallback: true` uses the earlier median section-line coordinates and
   writes the exact failure reason plus `section_fallback` to metadata. Set it to `false`
   for experiments that must fail closed rather than proceed without fitted boundaries.
7. **Independent blade regions.** The front and back each contain `surface`,
   `leading_edge`, `trailing_edge`, `root`, and `tip` regions. Corner ownership is
   deterministic: root/tip take precedence over leading/trailing edges. Region width is
   controlled by `boundary_band_fraction`.
8. **Fin face, root, and free-rim regions.** Each detected fin receives a local PCA OBB.
   When both physical faces are resolved, one patch family is created per face; otherwise
   two conservative normal hypotheses are retained for later completion. The attached
   root receives two views along the fin/main-normal bisectors and the outer rim receives
   a view along the corresponding main-blade half-space normal. These regions have
   independent coverage and quality entries.
9. **Patch centre and direction.** Every populated point set gets its own PCA OBB. The
   OBB centre is the target point. Local PCA normals are consistently oriented outward,
   binned in azimuth/elevation, and the dominant bin supplies the main normal.
10. **Curvature adaptation.** The 90th-percentile angular deviation from the main normal
   is the patch curvature score. Patches above the configured threshold are recursively
   split along their longer section-coordinate direction.
11. **Baseline plus region-adaptive fine views.** The calibrated left-IR intrinsics,
   baseline standoff, image margin, and utilization factor define the base partition
   footprint. Each patch then searches the explicitly configured distance interval.
   Flat main/fin faces prefer the feasible distance nearest the baseline; high-curvature
   patches, the four blade boundaries, fin roots, and free rims prefer the nearest
   feasible distance. Every patch point must pass the conservative image/depth gate, and
   a coarse-cloud z-buffer enforces the visibility threshold. An infeasible patch is
   recursively split; exhausting the split limit fails closed. Per-view distance,
   footprint, projection fraction, visibility fraction, and policy are persisted.
   Projection geometry is generated as `base_T_left_rectified`, then composed with the
   persisted `left_rectified_T_left_ir` calibration to obtain physical
   `base_T_left_ir` for hand-eye and IK. Both poses are stored and verified. The view set
   remains `motion_authorized: false`; workspace, IK, full camera/robot collision, and
   trajectory preflight are still required separately.
12. **Thin-wall TSDF and mesh.** Front and back use separate volumes. Their truncation
   band is `min(configured_truncation, measured_thickness * thin_wall_band_fraction)`,
   where the fraction must be below 0.5. This prevents opposing observations from
   cancelling the wall. When both fin faces are observed, their measured separation also
   limits the truncation band. Calibrated image metadata uses the optional Open3D scalable
   backend when installed; otherwise a dependency-free sparse projective TSDF and
   marching-tetrahedra extractor are used.
13. **Real-surface feedback.** Fine coverage is measured against every fixed coarse
   curved-surface sample, not a proxy plane. Generation zero is empty and deliberately
   does not import the views that created the coarse reference. Each successor appends
   exactly one verified FoundationStereo reconstructed candidate view and is replayed
   from its predecessor. Per patch it records observed fraction, surface RMSE, normal
   consistency, curvature, and failure reasons. Leading/trailing/root/tip plus fin-face,
   fin-root, and free-rim completion are reported separately. The fixed coarse mesh is
   not presented as fine reconstruction mesh quality.

## Running on reconstructed coarse views

First create the pose-registered native or FoundationStereo artifacts using the existing
`bbf reconstruct native-depth` or `bbf reconstruct stereo-depth` commands. Include
overlapping observations from both physical sides, then run:

```bash
uv run bbf reconstruct coarse-model \
  --view outputs/coarse_front_00 \
  --view outputs/coarse_front_fin_upper \
  --view outputs/coarse_front_fin_lower \
  --view outputs/coarse_back_00 \
  --view outputs/coarse_back_fin_upper \
  --view outputs/coarse_back_fin_lower \
  --config configs/local.yaml \
  --output outputs/blade_coarse_model
```

The command rejects mixed hand-eye matrices and fails when both camera sides are not
represented. The immutable schema-5 output contains checksummed fused
points/normals/side labels, patch samples and metadata, raw and rectified fine-view
transforms plus their calibration, both sparse TSDFs, extracted mesh, coarse-workflow
diagnostics, pose refinements, configurations, source hashes, and quality results.
Independent fine coverage starts in a separate surface-coverage generation; coarse
acquisition views never count as fine scans.

## Library-level online fine-scan transaction

Once a schema-5 coarse model exists, the optional FoundationStereo cycle branch can use
that exact immutable surface as its scientific foreground prior. It projects all coarse
surface samples into the current distortion-free `left_rectified` image with a nearest
depth z-buffer, then accepts only eligible measured depths inside configurable asymmetric
front/back tolerances. Eligibility is inherited from the safety observation's stereo
quality, range and robot-self-mask gates. This does **not** make safety occupancy blade-only:
the occupancy map continues to integrate every eligible scene surface, while the scientific
mask is the smaller reference-consistent subset. The masker deliberately performs no
connected-component selection or erosion, because either can remove the specimen's narrow
fin faces, attachment roots, free rims or one-pixel boundary evidence.
For the commanded target patch, samples must additionally face the camera according to the
configured main-normal incidence gate and own the nearest depth in the **full-surface**
z-buffer. An opposite thin-wall side or intervening fin therefore cannot satisfy target
support merely by falling inside the numerical depth tolerance.

The online branch is purpose-dependent. With no accepted scientific generation, the first
`BOOTSTRAP` or `SAFETY_REFRESH` observation that produces a fresh `MAP_READY` safety map
creates empty fine-coverage generation zero. A recovered `BOOTSTRAP`, every `TRANSIT`, and a
`SAFETY_REFRESH` with an accepted generation carry that exact generation without
manufacturing a scientific observation. A formal fixed-reference candidate must create,
inside the same immutable cycle directory, a source-bound foreground asset, one
foreground-bound schema-3 FoundationStereo reconstructed view and exactly one coverage
successor. The mask reader replays the projection from the bound stereo, occupancy
integration-valid mask and schema-5 coarse arrays. The reconstructed-view reader then
replays pixel selection and base-frame points from the same depth, mask, stored point-cloud
configuration and raw/rectified camera chain. Recovery requires every non-empty generation
in the recursively replayed lineage to use this schema-3 evidence. These assets remain
staged until coordinator readback and transaction acceptance; cancellation keeps the files
as diagnostic evidence but does not advance the accepted generation.

The default configuration disables this path. The supervised `scan run-unknown`
composition enters it only after the coarse session has published and re-read a complete
schema-5 handoff; it forks fresh committed perception sources into a new coordinator and
does not migrate a permit, prepared segment, occupancy publication or fine coverage. There
is intentionally no command that lets an operator bypass that handoff by injecting an
arbitrary reference into a moving run. Its visibility owner is a finite coarse-point splat
with a configured pixel radius, not continuous triangle rasterisation. Consequently the
projected sampling gap at each real standoff must be measured and accepted (or replaced by
mesh rasterisation) before using it as experimental occlusion evidence. Mask tolerances,
reflective-metal completeness, fin retention, latency and all quality thresholds still
require real ES68/D435i blade data. Continuous swept-mesh and robot-versus-voxel evidence
remain independent mandatory gates for each physical segment.

## Inspecting the fine plan before robot feasibility

Create immutable inspection evidence and open the read-only PySide6 orbit viewer:

```bash
uv sync --extra supervision-gui
uv run bbf reconstruct inspect-fine-plan \
  --coarse-model outputs/blade_coarse_model \
  --config configs/local.yaml \
  --output outputs/blade_fine_plan_inspection
```

Use `--no-gui` on a headless machine. The artifact contains `views.csv`, a region-coloured
`patches.ply`, line-based `view_frusta.obj`, a portable three-projection `overview.svg`,
and checksummed `metadata.json`. The viewer supports orbit/zoom, front/back and region
filters, camera-frustum and normal toggles, row selection, and rejection details.

The audit checks transform integrity, target distance, optical alignment, incidence,
configured projection/visibility gates, adaptive distance bounds, duplicate camera poses,
and independent front/back region presence. A failed audit exits with code 2 after writing
the evidence. Passing means **geometry only**: the report always records robot feasibility
as `unverified` and `motion_authorized: false`. ES68 IK, mounted-camera/robot collision,
workcell obstacles, ordering, and continuous-trajectory preflight remain separate gates.

## Configuration guidance

- Set `view_planning.standoff_distance_m` only after the real distance/accuracy
  experiment. Configure `minimum_standoff_distance_m` and
  `maximum_standoff_distance_m` together as the validated sensor-accuracy and clearance
  interval. Production mode keeps `surface_partition.derive_footprint_from_intrinsics:
  true`; the fixed `usable_footprint_m` override is reserved for synthetic or controlled
  footprint experiments and must be explicitly enabled by setting derivation to false.
- `image_edge_margin_px` and `footprint_utilization` shrink the calibrated image before
  partitioning. `minimum_patch_projection_fraction` and
  `minimum_patch_visibility_fraction` are hard planning gates, not quality scores.
- Keep `multi_view_fusion.maximum_translation_correction_m` and
  `maximum_rotation_correction_deg` below the error magnitude that would hide a bad
  hand-eye or robot pose. Residual ICP is refinement, not a replacement calibration.
- Choose TSDF voxel size below the smallest feature that must survive and ensure the
  protected truncation distance remains at least one voxel.
- Tune the curvature threshold on coarse-scan noise. A threshold below normal noise
  over-partitions; one above real curvature removes the adaptive benefit.
- The provided specimen configuration expects exactly one fin in each main-blade
  half-space. Use `fin_mode: disabled` only for finless calibration fixtures. Set the fin
  seed height above main-surface noise and below the shortest protrusion that must be
  retained; the grow height should be lower so region growth reaches the attachment.
- Resolving a fin thinner than the main blade can require a smaller TSDF voxel. The
  workflow deliberately fails if the thickness-protected truncation band is below one
  voxel instead of silently erasing the fin.
- Start with the stored boundary defaults, then inspect each curve's `fit_rmse_m` and
  `inlier_fraction`. Tighten `boundary_max_fit_rmse_m` only after measuring D435i noise;
  excessive control points or too little smoothing can follow edge outliers.
- Treat `blade_foreground` projection radius (pixels) and depth tolerances (metres) as
  experimental gates. A radius that is too small loses sparse edge/fin support; one that
  is too large can admit nearby surfaces when their depths also fall inside tolerance.
  Enabling the block without explicitly pinning the matching schema-5 reference is an
  invalid library assembly, not an automatic coarse-model discovery mechanism.

## Current validation boundary

Synthetic irregular, curved, and bilateral single-fin thin-wall tests verify separation,
same-side refinement, support filtering of isolated AC false candidates, four junctions,
robust four-curve fitting, approximately equal-arc sampling, curve-driven coordinates,
explicit fallback/fail-closed behaviour, shared bilateral base cells, all five regions,
OBB/main-normal construction, adaptive splitting, true-normal viewpoints, protected
TSDF, non-empty mesh extraction, independent real-surface quality feedback, artifact
checksums, raw/rectified frame composition, and the non-motion invariant. The fin tests
additionally verify main-surface decontamination,
two-face separation, root/free-rim regions, opposing and bisector views, fin-thickness
TSDF protection, coverage categories, schema-5 persistence, and missing-fin failure.
The online foreground/storage and purpose-dependent fine-transaction policies additionally
have deterministic software tests, but they are not a physical segmentation or accuracy
validation. Hardware accuracy, final thresholds, reflective-metal depth completeness,
thin-fin retention, final line-of-sight acceptance, timing, and real Open3D output still
require recorded D435i/ES68 scans and calibrated transforms.
