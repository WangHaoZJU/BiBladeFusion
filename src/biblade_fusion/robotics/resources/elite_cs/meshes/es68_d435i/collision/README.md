# ES68 + D435i collision meshes

The active D435i-only HoloRobot assembly uses these exact names:

- `base.stl`
- `shoulder.stl`
- `upperarm.stl`
- `forearm.stl`
- `wrist1.stl`
- `wrist2.stl`
- `wrist3.stl`
- `depth_camera_mount.stl`

The seven robot meshes are independent and expressed in their corresponding URDF link
frames. A single assembled-robot STL cannot articulate with the six joints. The active
`depth_camera_mount.stl` is the current D435i camera/mount envelope from HoloRobot; it
does not include the future thermal camera. The optical frame and hand-eye transform are
calibration quantities and must never be substituted for the physical collision origin.

The active relation is recorded in `collision_models/es68_d435i/manifest.yaml`. For a
future complete D435i/thermal payload, create a conservative replacement mesh, update the
manifest path and fixed transform, and assign a new model identity. Missing files, paths
outside the model root and incomplete link sets are rejected. `ready: true` means that
software may load the asset; it does not by itself certify physical dimensions or safe
motion.
