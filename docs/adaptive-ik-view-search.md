# Offline IK-aware adaptive view search

The first planning revision keeps the paper-derived target-centred view as the ideal
measurement geometry, but no longer treats one camera pose as the only admissible robot
pose. For each target patch, the planner expands a bounded pose family over optical-axis
roll, incidence tilt and azimuth, and camera distance. Distance is bounded only by the
configured stereo-depth limits; the ideal standoff is evaluated first and deviations are
penalized during ranking.

The search order is deliberately measurement-first. Ordinary surface families retain an
ideal/direction/distance breadth pass. Fin families first test the incidence with highest
`sin(tilt)*cos(tilt)` information, interleave wrist rolls, and only then spend their
remaining budget on lower-value tilts and distance expansion. Every candidate continues
to look at the same surface target. A family is bounded by both 32 IK attempts and 1.5
seconds by default. Runtime IK reuses the collision URDF's Pinocchio model plus
HoloRobot's bounded neighboring-seed sweep; the analytic calibrated-MDH solver remains
the no-shared-model fallback. A successful endpoint is still replayed through the
authoritative ES68 FK and all later collision/path gates.

The configured camera workspace is the declared physical outer boundary and is a hard
candidate rejection. Blade/camera clearance and configured forbidden volumes are also
hard geometric rejections. This does not replace the production motion
boundary: endpoint collision, swept-path collision, trajectory feasibility, and operator
authorization are not performed, and every report explicitly stores
`motion_authorized: false`.

Run the search on an immutable initialization artifact without connecting to hardware:

```bash
uv run bbf plan search-view \
  --initialization outputs/initialization_stereo \
  --view-id front_r00_c00 \
  --config configs/local.yaml \
  --output outputs/front_r00_c00_adaptive_search.json
```

Additional IK branches can be sampled by repeating `--ik-seed` with six joint angles in
radians. The captured initialization posture is always included.

When `view_planning.adaptive_ik_view_search.enabled` is true, the ordinary `plan views`
workflow and the unknown-blade coarse session now run this search for each front/back
proxy patch. The selected candidate replaces the old one-shot normal/fixed-fallback pose,
while the complete search trace is embedded in view-plan schema 4. The runtime then uses
the unchanged FK-consistency, collision, occupancy and motion-preflight chain. A candidate
being endpoint-IK feasible still does not authorize motion.

Fin discovery uses the same bounded pose-family mechanism without treating `15 deg` as a
hard constraint. The positive and negative members remain separate semantic targets on
each proxy axis, because they are intended to reveal opposing fin faces. Their fin-axis
component must keep the requested sign, but their camera positions need not be symmetric:
configured azimuth samples and their midpoints provide a common tangential bias inside
that signed open half-plane. For each member, the planner evaluates configured incidence,
physical stereo-depth distance and wrist roll. The 45-degree information optimum is tried
first; `15 deg` remains one bounded seed rather than a prerequisite.

The fin-specific geometry score uses `sin(tilt) * cos(tilt)` as a conservative proxy for
new fin information: the first factor represents side-face exposure and the second the
remaining support on the blade surface. Thus 45 degrees is preferred when equally safe
and reachable, but it is not mandatory; IK feasibility, forbidden-volume/clearance
rejections, joint travel and distance deviation can select a different angle independently
for every positive or negative target. The complete per-target trace is stored in coarse
fin-discovery schema 2. Collision and trajectory checks remain downstream and unchanged.

## Single-initial-view coarse NBV

After the operator supplies one initial view, coarse selection no longer follows a fixed
front/negative/front/positive/back sequence. Every endpoint-IK-feasible proxy or fin
candidate receives a blade-ROI discovery gain:

```text
measurement_quality = cbrt(visibility * projection * incidence)
expected_gain = measurement_quality * (
    w_surface * proxy_coverage_deficit
    + w_side * side_observation_deficit
    + w_fin * opposing_fin_evidence
)
```

`proxy_coverage_deficit` is replayed from the immutable proxy coverage ledger. A fin
candidate has no one-to-one proxy patch, so it uses the mean deficit on its blade side.
`side_observation_deficit` explicitly favors a completely unseen back side after a front
initialization. `opposing_fin_evidence` starts with a bounded seed value and rises to one
when the candidate completes a partially observed opposing pair. Attempts, IK, geometric
clearance and pair availability remain hard gates. The selected gain components are
stored in the coordinator decision diagnostics; safety occupancy and swept-path proofs
remain downstream vetoes and contribute no positive science gain.

Both coarse discovery and fine NBV retain their deterministic science-ranked endpoint
queues. The stop-scan coordinator hard-preflights the complete bounded queue in that
unchanged order and accepts the first collision-free straight joint-space path. A blocked endpoint is
audited and skipped; no occupancy value is fed back into discovery gain, and no
collision threshold is relaxed. Asset, map-binding, or timing failures remain terminal
for the cycle rather than being misclassified as candidate-specific path vetoes.

Fine NBV uses the same bounded pose-family fallback. A nominal curved-surface candidate
is tried first; if workspace/geometry/IK rejects it, up to four highest-priority incomplete
patch families search physical stereo distances and incidence directions. Every returned
solution is independently replayed through ES68 FK before science ranking. Thus a fixed
fine standoff or incidence angle cannot terminate an otherwise measurable patch.

One NBV endpoint is now one operator-approved motion and one stopped capture. The legacy
`stop_and_capture.maximum_segment_joint_delta_rad` and
`maximum_ranked_preflight_candidates` keys remain parseable only for old reviewed config
files and are ignored. `motion_preflight.maximum_joint_step_rad` still controls internal
ServoJ interpolation and continuous collision proof; it never creates an intermediate
view, FoundationStereo inference, or occupancy rebuild.

## Evidence-gated motion after one initial view

One verified operator bootstrap view may now enter `BOOTSTRAP_MOTION_READY` when
`stop_and_capture.allow_single_view_bootstrap_motion` is enabled and an immutable
static-free acceptance asset exactly matches the configured AABBs. This is an operational
planning phase, not a map-state promotion: the occupancy snapshot remains `MAPPING`, the
event records `map_ready_claimed: false`, and the full three-independent-view rule for
`MAP_READY` remains unchanged.

During this prefix phase, an UNKNOWN voxel is usable only when its complete voxel AABB is
inside an accepted static-free volume. OCCUPIED always wins, and UNKNOWN in the blade,
fixture, out-of-grid, or any other unaccepted volume blocks the segment. The complement
of the accepted volumes therefore acts as the conservative initial target envelope; the
planner does not have to guess the unseen back side of the thin bilateral finned blade.
Every proposed segment is hash-bound to `bootstrap_mapping_prefix=true`, and the guarded
safety factory refuses to replay that proposal against either an ordinary MAP_READY policy
or a different MAPPING generation.

Both prefix and MAP_READY motion now use the original collision STL loaded from the active
URDF. HPP-FCL measures that mesh directly against potentially dangerous occupancy voxel
boxes; transformed AABBs only select nearby candidates and never decide collision. For a
continuous interval, the exact midpoint STL separation must exceed clearance, accepted
tracking uncertainty, and the certified link-displacement bound. This removes the former
single circumsphere (and temporary cell-sphere) false positives without turning the sweep
into a discrete-only check.

Attempt 11 exposed two deployment facts that software must not hide. First, 160 projected
robot pixels lay more than 10 mm in front of the rendered surface (worst residual 36.95 mm),
leaving occupied self-ghosts around the base and upper arm. Exact-STL replay found 46 old
upper-arm occupied voxels within the required clearance; a 40 mm front self-mask tolerance
and eight-pixel silhouette dilation removes all of those source hits. Masked rays remain
UNKNOWN instead of being ray-cleared. Second, the accepted static-free asset ends at
`x=0.65 m` and
`y=-0.38 m`, while the stopped robot/camera envelope reaches approximately `x=0.70 m` and
`y=-0.40 m` after clearance. The new checker reports out-of-grid and outside-acceptance
UNKNOWN counts separately, but it deliberately cannot authorize this deployment until a
new physical static-free acceptance is recorded over the larger measured workspace.
