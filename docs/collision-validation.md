# Offline CS68 collision and path validation

BiBladeFusion provides a fail-closed, non-executable safety precheck for an explicitly
ordered set of planned views. It does not connect to the controller and never authorizes
motion.

## Geometry contract

The validator reproduces the exact Elite KDL chain found in the installed SDK plugin:
for each axis it applies the fixed MDH `RotX(alpha) * Trans(a, 0, d)` transform followed
by the revolute `RotZ(q)` joint. A nontrivial numerical comparison against the vendor
plugin produced a maximum end-pose difference of approximately `1.1e-16`.

Robot links are conservatively represented by six capsules between consecutive MDH
joint origins. A seventh capsule spans the final flange/TCP origin to the calibrated
left-infrared camera center. Workcell objects are clearance-expanded axis-aligned boxes.
Non-adjacent capsules are checked for self-collision using exact segment-to-segment
distance; capsule-to-box checks use conservative Minkowski-expanded boxes.

The default configuration deliberately contains no guessed CS68 dimensions or limits.
Copy `configs/default.yaml` to the Git-ignored local configuration and provide:

- six measured or conservatively enlarged `collision.link_radii_m` values;
- `collision.camera_tool_radius_m`, enclosing the flange, mount, and D435i;
- controller-appropriate six-axis minimum and maximum joint limits; and
- measured workcell obstacle boxes in the robot `base` frame.

`ignored_capsule_indices` may be assigned per obstacle only when an intentional overlap
is physically unavoidable—for example, a pedestal box containing the fixed base
capsule. Capsule indices are `0..5` for robot links and `6` for the camera/tool.

## Validate an explicit sequence

Only `endpoint_feasible` views with stored IK joint solutions are accepted. Repeat
`--view-id` in the exact proposed traversal order:

```bash
uv run bbf safety validate-path \
  --plan outputs/view_plan \
  --initialization outputs/initialization \
  --view-id front_r00_c00 \
  --view-id front_r00_c01 \
  --view-id back_r00_c01 \
  --config configs/local.yaml \
  --output outputs/path_validation_000
```

Each leg is linearly interpolated in joint space so no sampled joint changes by more
than `collision.maximum_joint_step_rad`. Every sample is checked against configured
joint limits, non-adjacent self-collision, and workcell boxes with
`collision.minimum_clearance_m`. The immutable report binds the view plan,
initialization, and controller-specific MDH artifact by SHA-256 and re-runs validation
when read.

## Safety boundary

This capsule/AABB model is an offline conservative prefilter, not a certified robot
safety system. It does not model exact link meshes, cables, payload deformation,
velocity/acceleration/jerk, singularity margins, stopping distance, controller blending,
or the controller's time-parameterized trajectory. Before any future execution feature,
the measured capsule model must be validated against CS68/workcell CAD and the final
controller trajectory must pass a higher-fidelity continuous collision and dynamics
check. A collision-free report always retains `motion_authorized: false`.
