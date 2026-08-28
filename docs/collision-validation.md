# Offline ES68 collision and path validation

BiBladeFusion provides a fail-closed, non-executable safety precheck for an explicitly
ordered set of planned views. It does not connect to the controller and never authorizes
motion.

## Geometry contract

The validator reproduces the exact Elite KDL chain found in the installed SDK plugin:
for each axis it applies the fixed MDH `RotX(alpha) * Trans(a, 0, d)` transform followed
by the revolute `RotZ(q)` joint. A nontrivial numerical comparison against the vendor
plugin produced a maximum end-pose difference of approximately `1.1e-16`.
This is a floating-point software-equivalence result for the sampled SDK fixture: it
checks transform order and arithmetic, not physical ES68 link dimensions, joint-zero
calibration, TCP accuracy, camera mounting accuracy, or end-effector positioning error.

Robot links are conservatively represented by six capsules between consecutive MDH
joint origins. A seventh capsule spans the final flange/TCP origin to the calibrated
left-infrared camera center. Workcell objects are clearance-expanded axis-aligned boxes.
Non-adjacent capsules are checked for self-collision using exact segment-to-segment
distance; capsule-to-box checks use conservative Minkowski-expanded boxes.

The default configuration deliberately contains no guessed ES68 capsule dimensions or
joint limits.
Copy `configs/default.yaml` to the Git-ignored local configuration and provide:

- six measured or conservatively enlarged `collision.link_radii_m` values;
- `collision.camera_tool_radius_m`, enclosing the flange, mount, and D435i;
- controller-appropriate six-axis minimum and maximum joint limits; and
- measured workcell obstacle boxes in the robot `base` frame.

`ignored_capsule_indices` may be assigned per obstacle only when an intentional overlap
is physically unavoidable—for example, a pedestal box containing the fixed base
capsule. Capsule indices are `0..5` for robot links and `6` for the camera/tool.
Run `uv run bbf doctor --config configs/local.yaml` to see the exact missing collision
fields before attempting path validation.

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

View-plan schema 3 binds every endpoint-feasible joint solution to the exact
controller-specific MDH file used by IK and the six joint-zero offsets used by FK. The
validator rejects legacy endpoint plans without this provenance, a supplied MDH path
that differs from the plan, or offsets that differ from the recorded plan contract.

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
the measured capsule model must be validated against ES68/workcell CAD and the final
controller trajectory must pass a higher-fidelity continuous collision and dynamics
check. A collision-free report always retains `motion_authorized: false`.

## HoloRobot mesh motion preflight

`bbf safety preflight-path` is the higher-fidelity successor for motion preparation. It
strictly loads the active ES68 link meshes and D435i assembly mesh; it never falls back
to the legacy upstream-labelled `cs68` fixture. Configured `collision.obstacles` become
clearance-expanded FCL boxes in `base`. Non-adjacent robot mesh pairs also use FCL
distance queries, so a self gap below the effective minimum clearance blocks before
contact.

For each ordered leg, this preflight derives the target as
`base_T_left_ir · inverse(flange_T_left_ir) · flange_T_tcp`, records that calibrated
`base_T_tcp` goal, and evaluates bounded-step discrete joint samples. It also rejects a
joint endpoint unless HoloRobot ES68 FK composed with the packaged
`flange_T_tcp` reaches the requested TCP pose within the configured translation and
rotation thresholds. The controller MDH used by IK and the packaged ES68 chain are
different representations; this endpoint gate checks their result, not an invalid
assumption that their source-file hashes should match.

The current production-mode call then stops at `continuous_swept_mesh_unavailable`
before velocity-limited ServoJ stream generation. The stored schema-5 report therefore
records the blocked leg and diagnostic joint-travel evidence; zero estimated ServoJ
duration is not evidence of a valid or executable trajectory. Library-only options can
disable a continuous-proof requirement for offline stream inspection, but such output is
explicitly diagnostic and cannot become approval evidence. The artifact binds its view
plan, initialization, occupancy, and motion-model inputs by SHA-256 and fully re-derives
the blocked report when read.

The original `validate-path` artifact remains for compatibility and for comparing the
conservative capsule model with the mesh model. Neither command connects to hardware or
authorizes execution.

The production ES68 path never falls back to the packaged legacy-labelled checker. Its
`robot_geometry_hash` covers the generated ES68 URDF, ES68 kinematics and joint-limit
files, all ES68/D435i collision meshes, and the configured joint-zero offsets. The
separate `motion_model_contract_hash` additionally covers every workcell obstacle,
effective minimum clearance, collision-pair filtering policy, resolved geometry-pair
set, and Pinocchio/hpp-fcl versions. Preflight evidence,
operator permits, live collision revalidation, and occupancy self-mask evidence must
agree on these identities or motion remains blocked.

The current mesh path remains discretely sampled and does not provide continuous swept
volume evidence. The occupancy path is independently discretely sampled and likewise
lacks continuous robot-versus-voxel swept-volume evidence; one proof cannot substitute
for the other. Consequently both are diagnostics, and production preflight plus guarded
execution remain blocked until both continuous backends are implemented and validated.
