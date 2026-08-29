# Fine-scan terminal reconstruction contract

Fine coverage is a prerequisite, not a completion proof. When every required
schema-5 reference patch has passed its distance, normal, incidence, and sample
quality thresholds, the fine selector now invokes a separate terminal workflow.
That workflow re-reads the complete foreground-bound schema-3 view lineage,
performs pose-prior bilateral multi-view fusion, integrates protected front/back
TSDF volumes, extracts a mesh, and re-evaluates surface quality against that mesh.

A fine run is complete only if all of the following are true:

- every fixed-reference patch remains complete in the replayed quality report;
- independently registered source views and mesh triangles exist on both blade sides;
- the reference contains one two-face-observed fin on each side, and the fin face,
  root, and free-edge regions are complete;
- mesh boundary-edge and boundary-loop counts satisfy the configured hole limits;
- the mesh satisfies the configured watertightness requirement.

The output directory contains `final_reconstruction.json`, fused point/normal/side
arrays, both sparse TSDF volumes, and the triangle mesh. The metadata binds the
terminal coverage generation, schema-5 reference, every reconstructed source view,
all reconstruction configurations, array hashes, quality results, and terminal
gate results. `replay_final_fine_reconstruction(path)` strictly re-reads those
sources and recomputes fusion, TSDF, mesh, quality, and gates; exact array or decision
drift is rejected. The asset always carries `motion_authorized: false`.

Production integration uses `finalize_fine_science(state, ...)`. Its default output
is the terminal coverage directory's sibling `final_reconstruction`. Repeated calls
do not replace it: they replay-verify the existing immutable artifact and require an
exact terminal-generation binding. The selector returns its canonical path,
artifact identity, and metadata hash in `NextViewSelection`; the coordinator writes
the same evidence into the terminal `coverage_complete` event.

Completion is then sealed in the experiment-wide chain. The runtime first strictly
replays the terminal run event's reconstruction path and verifies its artifact identity,
metadata hash and coverage generation. It appends the terminal `FINE_CHECKPOINT`, then
`FINE_COMPLETED` binds the fine run's exact final event/count, final coverage authority
and `final_reconstruction.json` authority. The complete chain is therefore
`INIT -> COARSE_CHECKPOINT+ -> PREPARED -> FINE_START_CANDIDATE+ -> FINE_STARTED -> FINE_CHECKPOINT* -> FINE_COMPLETED`.
Only the latest durable candidate may be atomically published as `FINE_STARTED`; a
candidate-only crash remains `PREPARED` and resume creates a fresh fine run.
Changing any run event, coverage generation, reconstructed source, fused array, mesh,
configuration or terminal decision makes replay fail; a partially written terminal
product is never upgraded to experiment completion. This is an evidence seal, not an
unattended-motion claim or physical accuracy acceptance.
