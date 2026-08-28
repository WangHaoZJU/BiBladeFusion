# ES68 + D435i collision model

The active `manifest.yaml` describes the current laboratory ES68 with the D435i-only
mount. Its eight collision meshes and fixed-joint relation are copied from the matching
HoloRobot ES68 model rather than inferred from a photograph:

- the seven arm meshes are expressed directly in their corresponding URDF link frames;
- `wrist_3_link_T_flange` and the fixed `wrist_3-depth_camera_mount` joint are identity;
- `depth_camera_mount.stl` therefore uses the HoloRobot collision origin
  `xyz = [-0.0505, -0.031815, 0] m`, `rpy = [0, 0, 0]` relative to `flange`;
- all STL vertices are in metres, so `mesh_units: m` must remain unchanged.

Inspect the assembled model without connecting to the robot or camera:

```bash
uv sync --extra robot-model-gui
uv run bbf robot inspect-model --config configs/default.yaml
```

An optional initial controller pose can be supplied as six degrees:

```bash
uv run bbf robot inspect-model \
  --config configs/default.yaml \
  --joints-deg "0,-60,90,-60,-90,0"
```

The viewer displays the exact eight collision STLs, updates them from packaged ES68
forward kinematics, provides per-link visibility and reports each STL loader state. It
has no robot address, device backend, permit or motion command. It is an assembly audit,
not trajectory validation or physical release evidence.

When the thermal camera or a different bracket is added, export one collision STL for
the complete new flange payload, replace the attachment mesh and fixed transform, assign
a new `model_id`, then repeat visual, FCL and hardware-envelope validation before setting
`ready: true`.

For activating a replacement model from scratch:

1. Export one collision STL per moving robot link and one STL for the complete flange
   payload. Follow the filenames in `manifest.template.yaml`.
2. Put the files under `meshes/es68_d435i/collision/` and create `manifest.yaml` from the
   template.
3. Set `mesh_units` to the actual export unit. Do not guess: STL has no unit metadata.
4. Fill `attachment.joint_origin` with the CAD/measured
   `flange_T_d435i_collision_link`. This is not `flange_T_left_ir`.
5. Inspect several assembled poses and validate collision pairs before setting
   `ready: true`.

`Es68D435iCollisionResources.load_active()` rejects an absent/inactive manifest, missing
meshes, an incomplete articulated link set, absolute paths and paths escaping the model
root. `build_es68_d435i_collision_urdf()` then replaces the legacy collision geometry and
materializes the calibrated ES68 joint origins for Pinocchio/FCL.

The configured safety clearance is stored as policy metadata; it does not rescale the
STL. Runtime distance enforcement must remain a separate FCL planning gate.
