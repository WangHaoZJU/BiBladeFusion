# Occupancy-bound motion safety

This document defines the minimum motion-safety contract for reconstructing an
unknown blade. The blade is not assumed to have a prior CAD or STL model. Robot
self-collision is checked against the configured ES68/D435i collision meshes, while
robot-to-environment collision is checked against an immutable, depth-derived occupancy
snapshot.

The required FoundationStereo "confidence" channel is the deterministic
`exp(-left-right disparity error / configured LR threshold)` consistency score stored by
the inference artifact. It is explicitly non-probabilistic and must not be interpreted as
a calibrated obstacle-existence probability.

## Safety invariants

1. Motion uses only a `MAP_READY` occupancy snapshot expressed in `base`.
2. The snapshot must contain source depth views and valid SHA-256 `content_hash`,
   `mapping_context_hash`, and `quality_evidence_hash` values. All three hashes remain
   bound through preflight, approval, and execution. A missing or changed hash blocks.
   Freshness is measured from the first frame of the complete map-rebuild cycle, not the
   most recently appended frame. The snapshot must remain fresh for the full predicted
   ServoJ duration plus the configured execution margin—not merely at motion start. The
   default margin is one second.
3. Occupied and unknown voxels both block motion. Out-of-grid queries are unknown.
4. Every robot collision STL is conservatively represented by the bounding sphere of its
   transformed local AABB. Query radius additionally includes occupancy obstacle
   inflation and configured collision clearance.
5. A preflight records the exact map `sequence`, content, mapping-context, and quality-
   evidence hashes. Operator approval and the one-shot execution permit bind the same
   values.
6. Authorization and execution re-read the current snapshot. Any state, sequence, hash,
   freshness, query-contract, or provider change blocks before driver preparation.
7. Occupancy updates occur only during stop-and-capture phases. Updating the map
   invalidates already prepared plans and permits.
8. Mapping and motion must share the exact `robot_geometry_hash`, which binds the
   generated ES68 URDF, ES68 kinematics and limits, every collision STL, and joint-zero
   offsets. Motion additionally binds a `motion_model_contract_hash` containing the
   workcell boxes, effective clearance, collision-pair policy, the resolved FCL pair
   set, and Pinocchio/hpp-fcl versions.
9. The occupancy-query policy has its own hash over maximum map age, added clearance,
   ignored geometry names, backend, and mandatory UNKNOWN blocking. Live execution
   rejects all ignored-geometry exemptions and requires the preflight, permit, and
   current checker to carry the same policy hash.
10. Every stored IK endpoint must reproduce its requested `base_T_tcp` under the
    HoloRobot ES68 FK and packaged `flange_T_tcp`: translation error may not exceed
    2 mm and rotation error may not exceed 0.3 degrees by default.
11. Every mapping frame is subject to the same independent FK/controller-TCP agreement
    gate before its depth rays enter the map. The measured residuals are hash-bound to
    that frame's quality evidence and revalidated when the occupancy asset is read.
    Mapping uses `base_T_flange(FK) · flange_T_left_ir`; controller TCP never becomes the
    camera-pose authority after passing the gate.
12. Mesh swept-volume evidence and occupancy swept-volume evidence are independent
    requirements. A continuous mesh/FCL result cannot substitute for a continuous
    robot-versus-voxel result. The current occupancy checker samples joint states only,
    explicitly reports `continuous_swept_volume_verified=false`, and therefore blocks
    production preflight and guarded execution.
13. A voxel becomes FREE only after the configured number of independent view votes
    (three by default), with at most one vote per frame. Each new supporting camera pose
    must differ from every existing supporting pose by at least 20 mm of camera-centre
    translation or 5 degrees of optical-axis angle by default. A renamed duplicate frame
    is rejected before it can alter any vote.
14. Motion accepts only a concrete immutable `OccupancySnapshot` accompanied by a typed
    semantic attestation issued by the full reader. The attestation binds the metadata
    bytes, snapshot sequence/content, mapping context, quality evidence, robot geometry,
    and verifier contract. Replay readers, protocol lookalikes, and mutable fake providers
    cannot issue or satisfy it.

The collision condition for a joint state is therefore

```text
self/workcell FCL collision or self-clearance violation
OR occupied occupancy intersection
OR unknown occupancy intersection
OR invalid/stale/missing occupancy evidence.
```

## Read-only adapter

`OccupancyRobotCollisionChecker` accepts a provider of immutable
`mapping.OccupancySnapshot` objects. It does not expose mapping mutation. For every
sampled joint state it:

1. validates map frame, lifecycle, rebuild-cycle age, source views, all three
   map/evidence hashes, the checker-bound robot-geometry hash, typed semantic attestation,
   and the freshness horizon;
2. evaluates Pinocchio geometry placements;
3. transforms each collision mesh's local-AABB sphere into `base`;
4. invokes `query_sphere(..., unknown_is_occupied=True)`;
5. validates state/count/blocking consistency returned by the mapping query;
6. confirms that the provider still exposes the same sequence, content hash, mapping-
   context hash, and quality-evidence hash.

All adapter and provider exceptions become an `UNKNOWN` collision status with a blocking
reason. They are never treated as free space.

Occupancy artifact schema 6 stores snapshot format 4 and mapping-context schema 4. The
integrity-only reader can replay it for permanently blocked visualization, but never
returns a semantic attestation. The full motion reader additionally verifies raw-session
array hashes, the bound user stereo-calibration asset, rectification reproduction, the
official FoundationStereo source/checkpoint/model configuration, hand-eye calibration,
and active ES68+D435i robot rendering. It re-runs the packaged ES68 FK from every joint
vector, reproduces predicted-versus-observed TCP residuals, reconstructs
`base_T_left_rectified`, and replays the self mask and voxel integration before issuing
the typed attestation. The metadata bytes are checked again after semantic verification
to detect a validation-time change.

## Preflight and permit binding

Motion-preflight schema 5 binds the typed occupancy semantic attestation in addition to
the independent continuous-occupancy-sweep contract and expanded evidence/permit
identity. Schema 4 and earlier artifacts are rejected; they cannot be silently
reinterpreted or upgraded into approval evidence.

`preflight_linear_joint_motion` requires both the mesh checker and occupancy checker by
default. The generated `JointMotionPreflight` is ready for approval only when both path
reports are clear, both mesh and occupancy reports independently carry continuous swept-
volume proof, and the occupancy report contains valid evidence. The current discrete
occupancy backend therefore returns `continuous_swept_occupancy_unavailable`. The
diagnostic-only `require_occupancy=False` or
`require_continuous_occupancy_sweep=False` options can produce an offline ServoJ stream
for inspection, but their reports are deliberately not approval-ready.

The execution permit includes the preflight fingerprint, both robot-model hashes, the
complete ServoJ runtime-config hash, explicit occupancy sequence/content/mapping-context/
quality-evidence hashes, the semantic-attestation hash, a positive continuous-occupancy-
sweep assertion, and the occupancy-policy hash.
Runtime tracking or timing guards must exactly equal the preflighted configuration;
execution-time overrides cannot relax them. Immediately before motion,
`GuardedEliteExecutor` checks:

- one-shot permit validity and exact preflight fingerprint;
- current snapshot identity and remaining freshness;
- live robot start-state agreement;
- complete mesh/FCL path revalidation;
- complete occupancy path revalidation against the preflight-bound snapshot;
- exact re-derivation of the velocity-limited ServoJ command stream and a continuous
  mesh/occupancy recheck for every emitted command segment;
- unchanged snapshot identity immediately before driver preparation.

Motion-preflight artifacts store `evaluated_at_utc`, require it to equal
`created_at_utc`, and bind it into the re-derived report. Historical integrity is
re-derived at that same instant, while real authorization and execution always perform
a separate freshness check against the current clock.

## Known limitations

- Bounding spheres are conservative and may reject feasible paths. Future mesh-to-voxel
  collision may reduce false positives, but must not under-approximate the STL geometry.
- Robot pixels removed by self masking remain UNKNOWN. The current checker therefore
  intentionally blocks physical motion when the robot envelope intersects that UNKNOWN
  volume. Do not mark an AABB sphere free or add broad ignored-link exceptions. A future
  release requires exact capture-pose mesh voxelization, hash-bound capture joints, and
  an observed-free clearance shell before any self-volume exemption is permitted.
- Continuous swept-mesh collision and continuous robot-versus-voxel occupancy collision
  are both unimplemented. They are distinct proofs. Discrete joint samples are
  diagnostic only: production preflight returns `continuous_swept_mesh_unavailable` or,
  once that gate is satisfied, `continuous_swept_occupancy_unavailable`. The guarded
  executor independently refuses either missing capability. These are intentional
  physical-motion blockers, not an approval that sampling is sufficient.
- This is stop-and-capture static-map avoidance, not certified continuous dynamic obstacle
  avoidance. The mapping provider must be frozen during one motion segment.
- The occupancy map is safety evidence, not the high-resolution blade reconstruction.
  Inflated safety voxels must not be fed back as blade geometry.
- Software STOP remains distinct from the controller's physical emergency stop.

## Hardware acceptance gates

Before exposing a physical execution entry, verify with the final ES68, D435i mount and
workcell configuration:

- collision STL scale, origin and link attachment;
- robot self-depth masking and synchronized `base_T_left_ir` transforms;
- measured workspace bounds and conservative UNKNOWN behavior;
- chosen voxel size, map-age limit, obstacle inflation and minimum clearance;
- occupied/unknown near-miss fixtures and first-blocked-path reporting;
- map-update invalidation between preflight, authorization and execution;
- ES68-FK endpoint agreement against the controller MDH IK results;
- continuous swept-mesh collision evidence for every trajectory segment;
- continuous robot-versus-voxel occupancy sweep evidence for every trajectory segment;
- worst-case preflight latency, ServoJ duration, tracking error and stopping behavior.
