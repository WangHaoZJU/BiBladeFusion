# ES68 + D435i collision meshes

Place the collision STL files in this directory with these exact names:

- `base.stl`
- `shoulder.stl`
- `upperarm.stl`
- `forearm.stl`
- `wrist1.stl`
- `wrist2.stl`
- `wrist3.stl`
- `d435i_assembly.stl`

The seven robot meshes must be exported independently in their corresponding URDF link
frames. A single assembled-robot STL cannot articulate with the six joints. The D435i
mesh must conservatively include the camera body, flange bracket, adapters and protruding
fasteners, but not the optical frame or hand-eye transform.

After placing the files, copy
`collision_models/es68_d435i/manifest.template.yaml` to `manifest.yaml`, configure the
STL unit and `flange_T_d435i_collision_link`, visually inspect the assembled model, and
set `ready: true`. Missing files, paths outside the model root and incomplete link sets
are rejected.
