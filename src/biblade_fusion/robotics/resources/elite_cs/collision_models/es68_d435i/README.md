# ES68 + D435i collision model activation

1. Export one collision STL per moving robot link and one STL for the complete D435i
   physical assembly. Follow the filenames in `manifest.template.yaml`.
2. Put the files under `meshes/es68_d435i/collision/`.
3. Copy `manifest.template.yaml` to `manifest.yaml`.
4. Set `mesh_units` to the actual export unit. Do not guess: STL has no unit metadata.
5. Fill `attachment.joint_origin` with the CAD/measured
   `flange_T_d435i_collision_link`. This is not `flange_T_left_ir`.
6. Inspect several assembled joint poses, then change `ready` to `true`.

`Es68D435iCollisionResources.load_active()` rejects an absent/inactive manifest, missing
meshes, an incomplete articulated link set, absolute paths and paths escaping the model
root. `build_es68_d435i_collision_urdf()` then replaces the legacy collision geometry and
materializes the calibrated ES68 joint origins for Pinocchio/FCL.

The configured safety clearance is stored as policy metadata; it does not rescale the
STL. Runtime distance enforcement must remain a separate FCL planning gate.
