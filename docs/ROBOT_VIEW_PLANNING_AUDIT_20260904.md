# Robot/view-planning regression audit — 2026-09-04

## Objective audited

From one operator-selected first view and one hard ROI, actively measure a thin,
double-sided finned blade by choosing views that provide useful new blade/fin information,
have a valid IK solution from the current stopped posture, and have a collision-safe
complete robot path. Generic ES68 control, IK ordering, ServoJ and straight-first/
RRTConnect behavior should follow HoloRobot rather than a second independent stack.

## Findings and disposition

| Area | Finding | Disposition |
|---|---|---|
| ServoJ | First write had no bounded session recovery; later errors risked repeated cleanup stop | First unchanged write has one HoloRobot-style reconnect; successful outer cleanup reuses confirmed stop |
| Start state | Executor allowed a 0.01 rad unstreamed bridge | Capped at HoloRobot's 0.001 rad and accepted uncertainty |
| IK | All seeds/branches could be solved before endpoint validation | Lazy branch iteration; first collision-clear branch wins |
| Adaptive views | First bounded prefix starved configured tilt/roll/distance dimensions | Representative breadth-first prefix |
| Coarse discovery | Initial joint posture froze later reachability | Immutable stopped-posture discovery revisions |
| Fin pairs | Missing first-view opposing pair could suppress useful normal NBV | Normal information-gain views remain selectable; schema-5 fin evidence stays hard |
| Path planning | Straight-first/bounded RRT was correct, but queue work lacked one response bound | Shared 30 s cooperative selection+preflight deadline; one native-call overrun remains possible; no threshold relaxation |
| Execution timing | Experimental release bypass could pass `None` as segment watchdog | Finite derived watchdog from stream/controller/settle limits |
| Recovery | `MOTION_BLOCKED` was capture-capable internally but terminal in the outer console | Explicit occupancy-only coarse `SAFETY_REFRESH` path |
| Supervision | Read-only GUI/timeline callback could terminate acquisition or motion | Best-effort observer isolated from authority callbacks |
| Doctor | Readable malformed kinematics/calibration and missing configured OMPL could pass shallow checks | Semantic parsing, resolution/API probes, configured OMPL failure |

## HoloRobot alignment retained

- controller-specific ES68 kinematics and ordered seed behavior;
- original URDF/STL collision geometry, not link bounding spheres;
- exact endpoint validation followed by straight joint path first;
- bounded RRTConnect only for an interior straight-path block;
- velocity/acceleration-aware ServoJ time parameterization;
- persistent RTSI state reads, endpoint feedback settling, and `writeIdle` segment stop.

BiBladeFusion retains the experiment-specific layers HoloRobot does not provide: blade
proxy/coverage gain, bilateral fin evidence, projected ROI propagation, immutable occupancy
and experiment assets, and the operator-approved one-segment stop-and-capture state machine.

## Verification

```text
ruff check .: passed
focused planning/runtime/storage/supervision regression: 334 passed
python -m pytest -q: 1294 passed, 3 skipped in 99.18 s
git diff --check: passed
```

The skips are optional vale environment probes for PyTorch/Open3D. No robot or camera was
opened during this audit. The code is ready for the next eiai guarded validation, but only
that validation can establish the physical controller, camera, placement and clearance
outcome.

## Known non-code boundaries

- Real hardware/network/SDK timing cannot be guaranteed by offline tests.
- A CUDA kernel stuck inside a third-party FoundationStereo call cannot be safely cancelled
  in-process; the cycle reports progress and enforces timing at return/commit boundaries.
- Production release still requires its separate science and runtime-timing acceptances.
- An intact experimental chain may now resume as experimental evidence after a real stop
  and fresh occupancy-only `SAFETY_REFRESH`. It never becomes production evidence, never
  restores an old permit/path/map, and must not resume after the physical placement changes.
