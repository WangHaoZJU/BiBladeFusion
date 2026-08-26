# Paired native and FoundationStereo depth comparison

This experiment compares two depth estimates from the same immutable D435i capture:
native RealSense depth and FoundationStereo depth. Native depth is a reference for
cross-sensor agreement, not geometric ground truth. Accuracy claims require an external
traceable reference such as a calibrated structured-light scan or metrology system.

## Coordinate alignment

FoundationStereo depth is axial depth in the calibrated `left_rectified` frame. Native
depth starts in the D435i depth imager and is not pixel-aligned with the infrared image.
BiBladeFusion therefore:

1. deprojects native depth using its own intrinsics and distortion model;
2. applies `left_rectified_T_left_ir @ left_ir_T_depth`;
3. projects into the rectified-left intrinsics with a nearest-depth z-buffer; and
4. evaluates only the intersection of the rectified blade mask and both valid depths.

This avoids treating equal array indices from different optical imagers as corresponding
surface samples.

## Run one paired comparison

The `--mask` input must be a Boolean `.npy` image in `left_rectified` coordinates and
must correspond to the stored stereo inference view:

```bash
uv run bbf evaluate depth-pair \
  --session data/<session> \
  --view-id seed \
  --stereo outputs/stereo_seed \
  --mask data/blade_mask_rectified.npy \
  --config configs/local.yaml \
  --output outputs/depth_comparison_seed
```

The artifact stores the projected native depth, shared-pixel mask, signed error map,
configuration, source identities, and SHA-256 provenance. Reading the artifact repeats
the calibrated calculation from its sources and rejects changed arrays or metrics.
Signed error is defined as:

```text
FoundationStereo depth - native RealSense depth
```

Reported metrics include each method's valid blade-pixel fraction, shared-pixel
fraction, signed mean/median difference, MAE, RMSE, P95 absolute difference, median depth
ratio, and agreement fractions at configured thresholds. The default thresholds are
5, 10, and 20 mm and should be changed only in a Git-tracked experiment configuration.

## Experimental interpretation

Use identical masks and settings across methods and retain per-view results. Report
front and back views separately before pooling them because incidence, edge occlusion,
infrared texture, and material response can differ by side. A method with lower error on
the shared subset but much lower valid-pixel coverage is not unconditionally better.
Likewise, paired agreement cannot detect a bias shared by both depth methods.

## Stratified aggregation

Create a version-controlled YAML manifest. Side and incidence are explicit experimental
labels; do not infer them from an arbitrary filename. Until the acquisition executor
records achieved poses, compute incidence from the calibrated camera pose and the fixed
proxy normal, and retain that calculation in the experiment notebook.

```yaml
schema_version: 1
incidence_bin_edges_deg: [0, 15, 30, 45, 60, 75, 90]
comparisons:
  - artifact: ../outputs/depth_comparison_front_000
    side: front
    incidence_angle_deg: 4.2
  - artifact: ../outputs/depth_comparison_back_000
    side: back
    incidence_angle_deg: 5.1
```

Paths are resolved relative to the manifest. Generate the aggregate with:

```bash
uv run bbf evaluate aggregate-depth \
  --manifest experiments/depth_comparison.yaml \
  --output outputs/depth_aggregate
```

The report always contains an overall group and preserves every populated side and
incidence bin. It reports both view-balanced averages and pixel-pooled errors because
the latter can be dominated by a small number of high-resolution/high-coverage views.
Duplicate physical source frames and mixed agreement thresholds are rejected. The
aggregate cryptographically binds its manifest and comparison metadata, then reopens
and re-evaluates every source whenever it is read.
