# Development log

## 2026-09-02 — three-frame cumulative replay de-dup and GPU boundary

- Corrected the performance scope from “first bootstrap frame” to the complete three-frame
  coarse bootstrap. Immutable attempt-09 evidence shows perception cycles of
  `531.567/991.799/1406.192 s`; the FoundationStereo-started to stereo-artifact timestamps
  stayed near three seconds while the later intervals grew with cumulative source-window
  replay. Attempt-09 did not record a backend-only online span, and checkpoint-to-live
  includes the old post-publication chain verification.
- Reduced the structural source-integration schedule from `18/44/92` to `13/28/57` for
  frames 1/2/3. The coarse-view writer remains the separately reviewed `3 -> 2` full-read
  change; generation now reuses one typed current view and one typed predecessor inside
  the append transaction; checkpoint validates only the new current generation against
  an already verified event prefix; read-only live ingest consumes a one-shot authority-
  bound view readback instead of replaying occupancy again.
- Added read-time metadata SHA/size to stored coarse views and generations. Private reuse
  rechecks exact view/reconstructed/stereo/occupancy and predecessor records before
  publication. Checkpoint fast validation now uses the same strict event schema and
  canonical-hash decoder as the full reader, holds the coarse-run publication lock, and
  verifies the new event/current generation both before and immediately after hard-link
  publication.
- Kept the MAP_READY transition and schema-5 promotion as independent strict boundaries.
  The third attempt-09 frame performs two full transition reads but returns COLLECTING
  because the configured gate requires six views; it does not write schema-5. A proposed
  in-memory reuse at the later schema-5 boundary was removed after safety review because
  root equality alone did not prove the complete coarse-to-fine authority closure. Resume,
  schema-5 handoff and final completion continue to execute full experiment/generation
  replay.
- Added `scripts/report_three_frame_replay_schedule.py`, which checks attempt-09's exact
  `1/2/3` occupancy, generation and checkpoint topology without hardware, then applies a
  source-reviewed call-count model. It neither instruments production reader calls nor
  makes elapsed-time claims. The `18/44/92 -> 13/28/57` boundary ends at the third capture's
  MAP_READY transition; it excludes later selection, a successful schema-5 write and the
  fine handoff. A new attended three-frame eiai run remains required for wall/CPU p50/p95
  and exact artifact equivalence.
- Repository-wide Ruff and `git diff --check` passed; the full suite reported
  `1143 passed, 2 skipped`, with both skips caused by the existing optional Open3D
  renderer dependency.
- Audited GPU feasibility without enabling it. The Python Amanatides-Woo loop is suitable
  for a per-source free/occupied bitmap CUDA backend, but same-source one-vote,
  occupied-wins, float64 tie behavior and fixed merge order must remain exact. The next
  stage should be a PyTorch CUDA shadow backend with CPU authority; no GPU production
  path, dependency, configuration or fallback was added in this change.
- Detailed formulas, safety decisions, GPU design and the eiai acceptance command are in
  `docs/phase1-three-frame-replay-dedup-2026-09-02.md`.

## 2026-09-02 — attempt-11 coarse-view writer readback de-dup

- Validated the copied attempt-11 timing authorities from commit `838d6bb`: the first
  perception cycle took `537.654 s`, including `226.127 s` of operator foreground
  preflight wait. FoundationStereo backend time was only `3.344 s`; seven measured DDA
  validations consumed `220.510 s` in the cycle, with another `31.380 s` DDA in coarse
  generation.
- Optimized one Phase-1 transaction boundary only. `write_coarse_scan_view` now reuses
  the first typed, fully verified integration source for foreground replay and frame
  identity, reducing its production full occupancy reads from three to two. A second
  unchanged production full reader remains immediately before atomic publication and
  revalidates the complete occupancy/stereo/session/hand-eye/active-robot authority
  closure. Identity, content hash, mask and the initially bound metadata record must
  remain exact; failure removes the partial output.
- Rejected a more aggressive one-read prototype after independent review found that a
  writer-local metadata/final-mask check would not cover the complete authority closure.
  The accepted two-read design closes that TOCTOU gap without duplicating the storage
  layer's manifest logic. Public readers, DDA, occupancy semantics, configuration,
  safety/science thresholds and all motion gates are unchanged.
- Added an immutable attempt-11 writer benchmark with cold 3 / warm 5 trials, an exact
  DDA-count oracle (`3` before, `2` after), strict production post-readback, normalized
  semantic comparison, input-tree no-write proof, no-clobber output, non-daemon nested
  process support and host/revision/source-file provenance. Process CPU/RSS excludes DDA
  child processes, so wall time is the authoritative performance metric.
- Final local validation reported `1090 passed, 2 skipped`; both skips are the existing
  optional Open3D renderer tests. Targeted Ruff and `git diff --check` passed. eiai was
  not used because its robot and depth camera were occupied. The expected one-DDA saving
  is about `31.5 s`, but no before/after speedup is claimed until the acquisition host
  completes the recorded cold/warm benchmark.
- Full evidence, commands and remaining gates are recorded in
  `docs/phase1-attempt11-coarse-writer-optimization-2026-09-02.md`.

## 2026-09-01 — Phase 0 performance timing and attempt-09 baseline

- Added bounded aggregate timing across stereo, occupancy integration and readback,
  coarse reconstruction/generation, live supervision, IK/FK filtering and segment
  preflight. The diagnostic path is explicitly outside safety, science and motion
  authority; it neither changes gate outcomes nor suppresses operation failures.
- Benchmarked the immutable attempt-09 assets without writing into the experiment.
  Cold 3 / warm 5 artifact readback took about `5.085/4.998 s` wall p50. Replaying
  one real occupancy source showed Python `DepthRayIntegrator` DDA at
  `17.587/17.669 s` wall p50 and `19.982/20.104 s` CPU p50; all replayed snapshots
  matched the persisted snapshots exactly. A three-source scaling smoke measured
  about `58.4–58.7 s` wall inside the integrator.
- Profiling located the measured offline single-core hotspot in per-pixel
  Amanatides-Woo/DDA traversal, repeated bounds checks and Python set/list updates.
  Source review also found that the same immutable source window can be replayed by
  rebuild, writer validation and strict reader validation. No replay, validation,
  occupancy representation or safety semantic was removed in this phase.
- Phase 0 remains incomplete until a new attended hardware run records the online
  spans for FoundationStereo CUDA, production semantic readback, coarse generation,
  live publication, IK/FK filtering and segment preflight. Until those measurements
  exist, compiled DDA, contribution caching, concurrency and octree changes remain
  proposals rather than authorized implementation work.
- Reproduction commands, input identity, p50/p95 resource measurements, caveats and
  the next-run acceptance checklist are recorded in
  `docs/phase0-performance-baseline-2026-09-01.md`.

## 2026-09-01 — attempt-09 fin-discovery reachability diagnosis

- Replayed the exact attempt-09 proxy, camera-candidate workspace and ES68 KDL seed.
  The original eight 15-degree fin-discovery endpoints were not rejected because of
  a front-side IK defect: all four front endpoints had IK solutions but their 50 mm
  camera clearance spheres left `es68_d435i_camera_candidate`; all four back endpoints
  also left that workspace and had no KDL solution.
- Added a separate `paired_fin_discovery_fallbacks` schema. It does not reinterpret the
  existing single-pose `coarse_reachability_fallbacks`: one new entry explicitly names
  one side, one proxy axis, exact distance/total/opposing angles and common-bias sense.
  Both concrete members independently pass the unchanged workspace and Elite IK filters
  before either can form an endpoint-feasible pair. Missing IK, missing workspace
  evidence, or only one reachable member still blocks.
- The exact diagnostic-only YAML shape used for the attempt-09 replay was:

  ```yaml
  view_planning:
    paired_fin_discovery_fallbacks:
      - side: front
        axis: major
        distance_offset_m: -0.05
        total_tilt_deg: 63.4
        opposing_tilt_deg: 34.5
        common_bias_sign: -1
      - side: back
        axis: major
        distance_offset_m: -0.05
        total_tilt_deg: 63.4
        opposing_tilt_deg: 15.0
        common_bias_sign: -1
  ```

  These values are candidates for attended segment preflight, not deployed defaults.
- No paired fallback is enabled by default or in `configs/local.yaml`. With the current
  deployment configuration, attempt-09 therefore remains safely blocked. The bilateral
  feasibility gate now runs immediately after the first proxy is created and reports its
  per-reason diagnostics after the first bootstrap instead of wasting two more long
  captures and failing only after `MAP_READY`.
- A read-only, in-memory dry run of two explicitly declared candidate pairs produced
  workspace-valid front/back poses and concrete `EliteCs68IkChecker` KDL solutions for
  the attempt-09 assets. The exact source hashes, candidate values, four joint solutions
  and statuses are recorded in `docs/attempt-09-fin-discovery-kdl-dry-run.json`. This is
  diagnostic evidence, not physical acceptance or motion authority; the values require
  attended measurement and ordinary segment collision preflight before configuration.
  Reproduce the native-plugin call without a robot connection or experiment-artifact
  writes via `.venv/bin/python -B scripts/reproduce_attempt_09_fin_discovery_kdl.py`.
  The replay binds the persisted first-view seed
  `[3.920504, -1.805604, 1.632554, -1.567073, -2.428674, 0.032375]` rad.
  Runtime constructs its checker from the process-start joints before the operator's
  manual reposition. A future run's start seed can differ, and there is no attempt-09
  evidence that it equals this first-view seed. The four diagnostic solutions are not
  cached or assumed reachable: every endpoint is solved again fail-closed during the
  attended run.
- Regression coverage uses a copied attempt-09 proxy and explicit in-test configuration.
  It proves the legacy generic fallback is ignored, explicitly paired geometry can fit
  the measured workspace, and no-fallback and IK-failure paths remain fail-closed.
  Targeted result: 77 tests passed.

## 2026-09-01 — TSR605 USB integration boundary

- Audited the supplied `Proj1/HCNetSDK` headers, libraries, examples and documentation
  without loading native code. The package exposes Hikvision network login/preview and
  network-channel thermometry APIs. It contains no reviewed direct-USB TSR605 enumeration,
  open-by-serial or radiometric-frame binding; USB-RNDIS/network-over-USB behavior remains
  unverified rather than assumed impossible.
- Added a non-connecting `bbf thermal audit-sdk` classifier. It identifies this HCNetSDK
  package as the Device Network SDK and exits nonzero, preventing USB configuration from
  silently loading a network ABI.
- Added explicit TSR605 USB configuration, model/serial/shape validation, typed adapter
  errors and an injected `Tsr605UsbBackend` seam for a future implementation based on the
  correct official SDK. No vendor functions or raw-count conversion formula were guessed.
- Wired standalone synchronized snapshot composition through the fail-closed thermal
  factory. Disabled sessions continue to use `NullThermalCamera`; enabling TSR605 now
  stops before robot/camera/session startup until a reviewed backend exists.
- Extended schema-3 thermal metadata with optional camera/transport/SDK provenance while
  retaining read compatibility with legacy thermal metadata that lacks provenance.
  Temperature matrices and optional raw counts remain immutable `.npy` evidence.
- Kept unknown-blade motion and fusion out of scope. Thermal remains forbidden there
  until thermal intrinsics/extrinsics, radiometric acceptance and a collision model for
  the complete D435i+TSR605 payload exist.
- Validation: 92 targeted configuration, adapter, diagnostic, acquisition, storage and
  CLI tests passed; targeted Ruff checks passed. No live USB capture claim is recorded.

## 2026-09-01 — placement-bound coarse support envelope

- Kept the operator hard ROI as the immutable source cloud and added a separate
  base-frame AABB intersection used by initial proxy PCA. The measured blade
  envelope and its minimum retained fraction must be configured together; production
  unknown-blade readiness fails while they are absent.
- Made the intersection fail closed when it retains too few points or falls below the
  placement-specific fraction gate. Diagnostics report raw/retained counts and XYZ
  bounds so an annotation, pose, or envelope mismatch cannot silently become a partial
  blade proxy. Camera-range MAD rejection remains a secondary in-envelope fallback.
- Upgraded initialization persistence to schema 8. It stores the complete hard-ROI
  `base_points_m.npy`, an aligned `proxy_support_mask.npy`, and reproducible filtering
  diagnostics; readers recompute the intersection from the sealed proxy configuration
  and reject mask or metadata drift. Schema-7 authoritative FK assets remain readable
  with their original validation path.
- Added operator-facing retained-point output and a non-moving placement checklist for
  measuring, configuring, replaying, and inspecting the envelope before the supervised
  scan may proceed.
- Closed the full coarse-chain gap found during the project audit. Coarse-view schema 2
  stores and replays an aligned support mask for every accepted view; proxy coverage,
  multi-view PCA/ICP, thickness, TSDF, curved-surface and fin partitioning now consume
  support clouds rather than the raw hard-ROI clouds. The schema-5 model records the
  exact coarse-view metadata hashes and proxy-support configuration used to build it.
- Corrected the initialization authority filename contract: the writer, discovery plan,
  selector and coarse generation now consistently bind `metadata.json`. A real-writer
  regression check prevents the former mocked `initialization.json` fixture from hiding
  this failure again. First-view planning assets are rolled back together on ordinary
  construction failure and can be reused if generation append alone is retried.
- Initialization readback now rebuilds a measured-envelope proxy from its support points
  and rejects changed axes, extents, centroid, eigenvalues, counts or camera incidence.
- Validation: `ruff check .` passed and the complete suite reported 1061 passed, 2
  skipped (the existing optional Open3D renderer tests). No hardware-replay claim is
  recorded: experiment 005 failed on an expired FoundationStereo frame before foreground
  extraction, so it contains no view-bound hard-ROI artifact. The similarly named
  `bootstrap_blade_current_position_005_surface_seed_polygon.json` belongs to a different
  capture and must not be overlaid on `operator_bootstrap_000`. A valid replay remains
  gated on receiving the exact session, stereo artifact, mask/seed, and pose from one
  successful capture chain.
- The earlier `current_position_check_005` diagnostic is not a hard-ROI replay: its
  `*_surface_seed_polygon.json` is a deliberately small surface seed. The latest locally
  available full-ROI asset is `bootstrap_blade_view2_polygon.json`; a visual-only pairing
  with `placed_blade_fixture_bootstrap_002` produced 37,222 valid ROI points and a trial
  `Z > 0` gate retained 37,179 (99.8845%). This again shows that zero-height clipping is
  not a useful foreground filter. Neither legacy polygon contains a sealed source hash,
  so production acceptance still requires a view-bound mask artifact.

## 2026-08-30 — eiai non-moving hardware P1 bring-up

- Bound the live eiai host robot interface to `192.168.6.61/24` and the only enumerated
  D435i to serial `243222074585`; ES68 `192.168.6.60` answered four ICMP probes with no
  loss. The rebuilt Elite SDK 1.0.0, RealSense enumeration, active calibrations and
  FoundationStereo CUDA doctor checks all passed while `motion_enabled` remained false.
- Captured `data/hardware_validation/gpu_host_d435i_smoke_20260830.npz` with the emitter
  disabled. Both infrared frames are 1280x720, and the bundle includes native depth plus
  the stream calibration transforms. Read-only RTSI reported `POWER_OFF/NORMAL` without
  loading a task, releasing brakes or sending a trajectory.
- Created synchronized schema-3 session
  `data/20260830T033606.728525Z_gpu_host_sync_20260830_001_fbc3ff8f`. Its 36.141 ms robot
  bracket had zero joint, TCP translation and TCP rotation deltas. Full CUDA inference
  produced 847,857/921,600 valid depth pixels, and canonical source verification followed
  and revalidated the raw-session hash chain.
- Controller-specific MDH export was initially blocked while the controller was
  `POWER_OFF`: RTSI port 30004 and Dashboard port 29999 accepted connections, but Primary
  port 30001 returned `Connection refused`. After the operator powered the controller and
  enabled remote mode, read-only status reported `RUNNING/NORMAL` and Primary exported the
  schema-2 MDH artifact with SHA-256
  `e8454a1b6c0ade50c988370232533d8287a5266ef50f3f122b4e8e03c584ed45`. No default or
  guessed MDH parameters were substituted, and BiBladeFusion issued no power, brake or
  motion command.

## 2026-08-29 — eiai GPU host environment bring-up

- Initialized the pinned FoundationStereo submodule at
  `6e8806816b533e4d13ddbb95ffa907b797060a62` and installed every locked dependency group
  and optional extra into the Python 3.12 virtual environment. The target RTX A6000 ran
  a CUDA matrix smoke test with PyTorch 2.4.1+cu121 and cuDNN 9.1.
- Corrected NVIDIA driver-version fallback parsing for driver files whose first matched
  version uses the generic dotted-version pattern. The previous fallback had no capture
  group but attempted to read group 1.
- Made the supervision replay fixture record the same numerically computed TCP rotation
  delta as the production capture path instead of assuming an exactly zero result from a
  floating-point rotation product.
- The eiai login shell exports ROS Humble Python 3.10 paths through `PYTHONPATH`; validation
  explicitly removed that incompatible external path so the Python 3.12 environment used
  its locked Pinocchio 2.7 installation. Final verification passed all 986 tests, Ruff,
  Python bytecode compilation, bootstrap shell syntax, and `uv lock --check`.
- Added an isolated PyTorch safe-global scope for the official FoundationStereo training
  checkpoint. It retains `weights_only=True`, admits only the NumPy/OmegaConf container
  types present in the official archive, and restores the process's previous allowlist
  after loading instead of falling back to unrestricted pickle execution.
- Completed full-resolution saved-session inference on the RTX A6000 with the transferred
  checkpoint and active D435i calibration. The same-process cold/warm measurements were
  15.718 s and 2.988 s; peak CUDA allocation/reservation was 5.78/10.17 GB. Both runs
  produced the same 854,577/921,600 valid pixels and byte-identical disparity, validity,
  depth and confidence arrays. Canonical asset readers revalidated all source and output
  hashes.
- The original `elite_cs_sdk-0.10.0-py3-none-any.whl` is not Python-version independent:
  its binary extension reports that it was compiled for Python 3.10 and rejects the
  required Python 3.12 interpreter. The replacement
  `elite_cs_sdk-1.0.0-cp312-cp312-linux_x86_64.whl` initially supplied with SHA-256
  `8db4db6e7fd96d45b99615106c2a2c3dd9d877c8bba73e7feac879e2eca44d03` installs under
  Python 3.12, but its bundled native SDK requires `GLIBCXX_3.4.32` and `GLIBC_2.38`,
  which exceed this Ubuntu 22.04 host. The locally rebuilt wheel with SHA-256
  `7fa0c9512f44ad0f4632652dab30b7c2346d349d2037af93c4144a62c73ad187` instead requires
  at most `GLIBCXX_3.4.29` and `GLIBC_2.34`; it installs and imports successfully using
  the host libraries. After installation, a repeated GPU saved-session inference retained
  854,577/921,600 valid depth pixels and byte-identical disparity, validity, depth and
  confidence arrays. No camera stream, robot connection or motion command was used during
  this environment bring-up.

## 2026-08-29 — pre-acceptance supervised unknown-blade scan closure

- Closed the top-level coarse-to-fine experiment authority with the write-once
  `INIT -> COARSE_CHECKPOINT+ -> PREPARED -> FINE_START_CANDIDATE+ -> FINE_STARTED -> FINE_CHECKPOINT* -> FINE_COMPLETED`
  chain. Checkpoints bind accepted science generations to exact run-event boundaries;
  PREPARED separately binds the schema-5/reference transition. A candidate is explicitly
  non-authoritative until the latest candidate is atomically committed by FINE_STARTED;
  StopScan appends and outer publication share a canonical-root process `RLock` plus
  no-follow regular lockfile/`flock`, closing the pre-publication concurrency window.
  Crash recovery abandons an orphan fine run and creates a fresh candidate. FINE_COMPLETED
  seals the terminal coverage plus strictly replayed final reconstruction. Source mutation,
  missing/spliced events or any persistence failure blocks without transferring authority
  or reporting completion.
- Added deterministic crash recovery through `scan run-unknown --resume`. Recovery is
  derived only from the explicitly named, fully replayed experiment chain; it discards
  old permits, approvals, prepared segments, map freshness and controller authority. A
  sealed completion returns a read-only report without opening robot or camera hardware.
- Added immutable geometry-science acceptance and its non-moving recording command. The
  record covers the configured distance/incidence envelope and binds calibration,
  FoundationStereo source/checkpoint/model environment, the complete installed Python/OS/
  CUDA/GPU runtime identity and all geometry policies. It now seals canonical copies of a
  machine evaluation, raw-asset manifest and independent review, then replays every hash,
  sample count and metric on read. Unknown runtime doctor blocks on missing or mismatched
  science acceptance. This gate never authorizes motion.
- Added an immutable four-budget runtime-timing authority. Cold and warm trials cover the
  complete perception cycle, operator reposition interval, guarded segment execution, and
  schema-5/fine handoff. The recorder rebuilds the report/manifest from the canonical raw
  trace-v2 assets, seals every trace under the acceptance asset, and replays them on read.
  Each trace binds the measurement implementation, current full runtime contract, Linux boot,
  a sealed measurement-session manifest, exact monotonic nanoseconds, and the canonical JSON
  operation evidence embedded with library-computed kind/SHA-256/size; cross-environment or
  no-evidence reuse is rejected. The measurement-session and final acceptance workcell must
  match exactly. Readers use one regular non-symlink snapshot for every sealed core/evidence
  file. Production rechecks the bound asset before hardware construction and enforces each
  accepted limit at its actual runtime boundary. The default configuration retains null limits
  and a null authority, so no timing value is silently guessed.
- Made FoundationStereo retries immutable and unambiguous. Every attempt has a unique
  directory, failed/cancelled evidence is retained, and exactly one atomic commit marker
  selects the accepted attempt. Commit, occupancy publication and freeze re-read their
  disk authorities. Occupancy schema 7 now counts physical captures rather than logical
  labels; schema 6 is permanently replay-only.
- Added a measured motion-envelope acceptance bound to the final robot geometry,
  collision contract and ServoJ control contract. Startup uses a Dashboard stop before
  driver construction and verifies STOPPED/IDLE/safety plus actual and target joint/TCP
  velocity channels over a complete stationary window. Segment stop acknowledgement and
  physical stationarity are persisted independently, and continuous envelopes include
  the accepted tracking/stop uncertainty. A segment deadline watchdog now requests
  Dashboard stop through a software channel independent of the ServoJ command lock;
  rejected, blocked or repeated stop failure is persisted as emergency-stop-unconfirmed
  rather than reported as normal recovery. This is not a hard-real-time safety stop and
  remains subject to physical SDK-concurrency and RTSI-stationarity acceptance.
- Replaced the live observer's alternate robot depiction with the exact active
  ES68+D435i collision manifest/STLs and added a disk-backed append-only registry for
  displayed physical point-cloud sources. Both are rehashed before publication and on
  restart; the observer remains command-incapable.

- Added a replay-verifiable initial foreground stage for an unknown blade. It uses only
  occupancy-integration-eligible FoundationStereo depth, rejects components touching the
  valid-domain boundary, fails on unseeded ambiguity, preserves thin fins by avoiding
  erosion, and supports explicit rectified-left rectangle/polygon seeds. Every decision,
  input array and source asset is stored in an immutable hash-bound artifact.
- Added the online bilateral coarse-science lineage. The first accepted stopped view
  creates the proxy initialization and endpoint-filtered normal plan; each side also gets
  paired positive/negative oblique candidates about both in-plane proxy axes. Every
  accepted view advances one write-once generation and measured proxy coverage. Promotion
  to schema 5 requires front/back view counts, proxy coverage, an opposing oblique pair on
  each side, and two observed physical faces for the single protruding fin on each side.
  The coarse-to-fine handoff is one-way and transfers no motion permit or approval state.
- Connected typed coarse assets to the FoundationStereo transaction and coordinator. A
  coarse wrapper is prepared while the continuous stationarity sampler is active, then
  independently read and appended only after the matching perception transaction commits.
  Candidate captures may therefore run without a fine reference, while coarse and fine
  asset triples remain mutually exclusive and source-identity checked.
- Implemented conservative continuous collision certificates. Robot self/workcell mesh
  clearance uses adaptive joint-interval subdivision, FCL midpoint separation and
  serial-chain displacement bounds. Robot-versus-occupancy clearance encloses each full
  interval with expanded link-geometry spheres and retains `UNKNOWN` as blocking. Proof
  limits and numerical ambiguity return `UNKNOWN`, never a sampled-path pass. Evidence
  binds the joint path, geometry/model contract, map, semantic attestation, policies,
  tolerances and termination reason through preflight and guarded execution.
- Added immutable static-free workcell acceptance for the narrow self-mask exception.
  Only complete UNKNOWN voxels inside an exact accepted AABB may be treated as free of
  external objects; OCCUPIED always blocks. The record binds operator, workcell, time,
  robot geometry, workspace, exact regions and a mandatory physical checklist, and never
  authorizes motion by itself.
- Added a supervised one-segment runner and a command-incapable live-observation bridge.
  It publishes atomic snapshots for the existing follow-mode GUI: robot/FK scene, planned
  trajectory, stopped actual samples, occupancy, FoundationStereo images/depth and
  coarse/fine point clouds. Missing evidence publishes a BLOCKED snapshot and latches the
  runner. Planned and actual display semantics explicitly avoid claiming high-rate
  tracking or TSDF fusion where those data do not exist.
- Added the public, interactive `bbf scan run-unknown` physical-motion composition root.
  Its separate `scan doctor --mode unknown` readiness audit is strictly non-moving. The
  runtime connects one real ES68, one D435i and
  one FoundationStereo backend only after offline readiness passes; initial views remain
  explicit `c` captures. Every later segment consumes an exact one-shot approval, may
  perform capability-gated power/brake preparation only after consumption, revalidates
  the complete safety binding, executes ServoJ, stops, and captures exactly once. The
  schema-5 transition forks only fresh, independently rehashed perception sources into a
  new fine coordinator; no permit, prepared trajectory, publication or coverage state is
  migrated. Coarse and fine phases reuse one read-only live timeline so copied coarse
  clouds survive the event-stream reset.
- Added GPU bootstrap and non-moving readiness checks. The readiness audit verifies the
  active calibration, kinematics, view/workspace policy, final collision assembly, both
  continuous backends, FoundationStereo/CUDA prerequisites and the exact static-free
  acceptance before a hardware runtime may be opened. Defaults remain fail-closed and all
  physical dimensions, timing thresholds and safety release evidence remain subject to
  controlled ES68+D435i acceptance.
- This entry supersedes earlier same-day statements that the continuous proofs,
  coarse-science transaction, live composition or public supervised entry point were
  absent. Those statements remain below as historical increment records, not current
  capability descriptions.

## 2026-08-29 — reference-guided foreground and transactional fine-science staging

- Connected the fixed-reference fine branch to the concrete FoundationStereo cycle at
  library level. A scientific mask is now the intersection of occupancy eligibility and
  depth agreement with the schema-5 coarse-surface z-buffer; a target patch must face the
  camera and win the full-surface z-buffer, preventing millimetre-separated sides from
  cross-satisfying target evidence. Safety occupancy still uses all eligible scene depth.
  The masker has no connected-component or erosion stage so
  narrow fins and boundary pixels are not removed by topology cleanup.
- Added immutable, source-bound foreground assets and bound each candidate reconstruction
  to its exact mask, raw session, stereo inference, occupancy evidence, rectified camera
  pose and coarse-model reference. Readers replay the decision from the bound stereo,
  occupancy integration-valid mask and coarse arrays, then re-deproject the stored mask and
  stereo depth with the recorded point-cloud configuration and camera pose to verify every
  scientific pixel and base-frame point. Candidate capture stages that mask, one
  foreground-bound schema-3 reconstructed view and exactly one coverage successor inside
  the same cycle root; recovery rejects any non-empty lineage containing a legacy schema-2
  observation.
- Added typed capture purposes. A new run creates empty generation zero on its first
  map-ready bootstrap or safety refresh; a recovered run carries its explicitly verified
  generation through bootstrap. Transit and later safety-refresh cycles carry the exact
  accepted generation without creating science observations; a formal candidate must
  publish the complete local successor.
  Source-window and accepted-coverage state advance only after coordinator readback and
  transaction acceptance. Cancellation retains immutable diagnostic files but clears the
  unaccepted in-memory transaction.
- This closes a software integration boundary only. `blade_foreground` remains disabled by
  default; callers must explicitly pin a schema-5 reference and optional recovery generation,
  and no public composition-root or robot-motion CLI currently assembles the path. It has not
  been validated on a real blade. The current visibility owner uses finite coarse-point
  splats rather than continuous triangle rasterisation, so projected sampling gaps and the
  pixel radius are also hardware acceptance items. Workspace acceptance, continuous swept
  ES68+D435i mesh and robot-versus-voxel proofs, measured timing thresholds and controlled
  hardware acceptance remain blocking.

## 2026-08-29 — fixed-reference fine coverage and concrete next-view selection

- Corrected fine-view camera semantics for non-identity stereo rectification. Look-at,
  projection, visibility and standoff are now generated and checked in
  `base_T_left_rectified`; the persisted calibration is then composed to obtain physical
  `base_T_left_ir` for hand-eye, IK and execution. Coarse-model schema 5 stores and
  cross-verifies both pose arrays and the transform between them.
- Added immutable fine-surface coverage generations against one fixed schema-5 coarse
  model. Generation zero is empty and never imports coarse-acquisition coverage; each
  successor replays exactly one FoundationStereo reconstructed candidate view from its
  predecessor. Readers verify complete source lineage, checksums, raw/rectified frames,
  bilateral single-fin semantics and independently recompute every patch quality value.
  Fixed coarse mesh diagnostics are not relabelled as fine reconstruction quality.
- Added `BladeCoverageNextViewSelector`. Completion now requires configured regions on
  both blade sides, two resolved physical faces for each protruding fin, and every
  required patch passing coverage, surface-RMSE and local-normal gates. Incomplete
  coverage with no unused workspace/IK/FK-feasible candidate raises a typed blocked
  result; it can never be returned as completion.
- Candidate image geometry uses rectified poses while workspace and IK use raw left-IR
  poses. IK is rebuilt from the current stopped joints and every solution is independently
  checked by calibrated ES68 FK against the target TCP. Deterministic ranking is
  coverage-first; joint travel is only a final tie-break. Selection policy, fixed coarse
  reference and fine generation hashes propagate into segment proposals, preflight
  diagnostics and completion events.
- Kept safety and science histories independent: fresh-window occupancy is consumed only
  by downstream short-segment safety, while fine coverage is cumulative. Online blade-mask
  production and reconstruction/coverage staging are still not connected to the concrete
  cycle engine, so missing science assets fail closed. Workspace remains unconfigured and
  continuous swept-volume proofs remain absent; no robot-motion CLI was released.
- Closed the receding-horizon transit contract: a `transit_*` capture may carry forward
  the preceding verified fine generation while refreshing safety occupancy, whereas a
  captured reference-candidate ID must publish its reconstruction and matching successor
  generation in the same cycle. New science successors cannot escape that cycle root;
  the selector pins the expected coarse-model path/hash and enforces exact generation
  continuity so another blade/run cannot be cross-wired into transit planning.
- Closed this increment with `576 passed`, repository-wide Ruff and bytecode compilation,
  whitespace-integrity checks, lockfile consistency, and a CLI smoke test. The optional
  offline package build was not used as evidence because its isolated cache did not contain
  the `hatchling` build dependency.

## 2026-08-28 — FoundationStereo-only stop-and-capture motion coordinator

- Added a library-level receding-horizon state machine for the ES68 eye-in-hand scan:
  explicit stop, sampled settle gate, one closed stereo capture, FoundationStereo
  inference, fresh-window occupancy rebuild, one short joint leg, per-leg approval,
  guarded ServoJ, explicit stop, settle and mandatory recapture. Native D435i depth is
  not a selectable fallback, and no motion command was added to the public CLI.
- Made perception acceptance transactional. Candidate raw/stereo/occupancy/stationarity
  assets do not enter the source window or become the current occupancy generation until
  independent disk-semantic verification succeeds. Source-window acceptance and map
  publication share one publisher lock, so concurrent readers cannot observe or freeze a
  half-committed generation; failed acceptance leaves the prior generation unchanged.
  `MAP_READY` and its event are emitted only after this commit completes.
- Added bounded-gap sampled stationarity evidence and write-once, hash-chained run events.
  Event persistence failure is an irreversible terminal latch for that coordinator
  instance. The evidence detects sampled drift, sampled out-and-return motion, stale or
  frozen feedback, clock regressions and invalid controller state; it is explicitly not
  claimed as continuous immobility proof between RTSI samples.
- Hardened the asynchronous stop boundary. Stop first increments a monotonic generation
  and locks out motion, then shares a short transport gate with every ServoJ write. A
  permit binds that generation across control recovery, preparation and streaming; after
  `stop()` returns, an older permit cannot write another ServoJ frame. Configuration now
  rejects non-ES68 coordination, a disabled motion driver, divergent planned/driver
  ServoJ periods, incompatible LR-consistency thresholds, and unequal component policy
  snapshots before the run starts.
- Production motion remains intentionally unavailable: defaults are disabled, the short
  segment limit and workspace bounds require measured values, and the current collision
  backends cannot issue either continuous swept-mesh or continuous robot-versus-occupancy
  evidence. The expected production result is therefore `MOTION_BLOCKED`, pending those
  proofs and hardware acceptance. Verification closed with `538 passed`, repository-wide
  Ruff and bytecode compilation, whitespace checks, lockfile consistency, and an offline
  sdist/wheel build containing all new modules.

## 2026-08-28 — coverage-derived coarse-scan ordering and preflight binding

- Coverage-plan schema 2 now turns incomplete proxy patches into a deterministic,
  non-executable traversal proposal. It finishes the selected proxy side first and
  applies a stable row-wise snake using the original row parity, so deleting completed
  cells cannot reverse a later row during replanning.
- Coverage and reachability remain separate hard gates. Only `endpoint_feasible` views
  carrying persisted six-axis joint solutions enter `ordered_view_ids`;
  `geometry_only` views are retained as `deferred_unverified_view_ids`, while rejected
  incomplete patches remain blocked. Occupied fraction is persisted as audit evidence
  but does not reorder the path, and joint travel is deliberately not an objective.
- `bbf safety preflight-path` accepts exactly one ordering source: repeated manual
  `--view-id` values or a coverage-plan artifact. Automatic mode verifies the source
  view-plan identity, requires exact ordered-ID equality, includes the coverage manifest
  in the SHA-256 source chain, and repeats the checks during artifact readback.
- The order is still only a proposal. It does not authorize motion or prove the
  front-to-back leg. Mesh and robot-versus-voxel paths remain independently fail-closed
  because their current bounded-step checks do not constitute continuous swept-volume
  evidence.
- Closed the increment with `423 passed`, repository-wide Ruff, bytecode compilation,
  and whitespace-integrity checks.

## 2026-08-28 — active ES68+D435i collision assembly and offline inspector

- Activated the current D435i-only ES68 collision manifest from the matching HoloRobot
  model. The seven arm meshes remain in their URDF link frames; the payload uses the
  documented identity flange joint and the `depth_camera_mount.stl` collision origin
  `[-0.0505, -0.031815, 0] m` with zero rotation. The eight copied STL files are byte-for-
  byte identical to their HoloRobot sources and retain metre units.
- Added a completely offline PySide6/Qt3D assembly inspector. It renders the exact eight
  collision meshes, drives them from the same packaged ES68 forward kinematics and joint-
  zero offsets used by safety code, provides six joint controls, selectable STL layers,
  orbit/zoom controls, link positions and mesh-loader status. The command has no robot-IP
  option, opens no device backend and contains no motion or authorization path.
- Exercised the production Pinocchio/FCL chain with the active manifest: all eight
  geometries loaded, 20 filtered collision pairs were constructed, and zero plus three
  nonzero diagnostic poses were clear under the configured 10 mm policy. These discrete
  checks establish software loading and assembly consistency; they are not a continuous
  swept-path proof or hardware dimensional acceptance.
- Audited mesh quality separately. The source set is intentionally detailed (about
  674,000 triangles); several meshes are not watertight and five degenerate faces remain,
  although hpp-fcl 2.4.4 loads the complete set. Keep this provenance set for acceptance
  and introduce simplified collision meshes later only under a new model identity with
  conservative-envelope regression checks.
- Completed a real desktop-display launch and operator visual check, then closed the
  change with the complete regression suite (`414 passed`), repository-wide Ruff,
  bytecode compilation and lockfile consistency checks.

## 2026-08-28 — FK-authority native-depth re-evaluation

- Reprocessed the five preserved real ES68/D435i sessions with the current schema-2
  `FK(joints + zero offsets) · flange_T_left_ir · left_ir_T_depth` authority chain;
  controller TCP remained validation-only.
- The new immutable report
  `data/validations/native_overlap_20260828_fk_authority_v2` passed strict full
  recomputation. Across the four comparisons, median error was 1.220–1.423 mm, RMSE
  2.069–2.469 mm, P95 4.205–4.992 mm, and 5 mm agreement 95.03–97.36%, over a
  188.01 mm/23.683 degree pose span. ICP remained diagnostic-only.
- Preserved the schema-1 TCP-primary report unchanged and added a separately named
  integrity-only legacy replay reader; legacy values are never promoted to current
  FK-authority evidence.

## 2026-08-28 — unknown-blade occupancy safety and supervisory replay

- Added stop-and-capture occupancy construction from stored FoundationStereo depth in
  calibrated `base` coordinates. Mapping requires left-right consistency evidence, its
  explicitly non-probabilistic consistency-score array, a bounded depth range, synchronized
  ES68 joints, the accepted flange-primary left-IR hand-eye transform, and at least three
  geometrically independent settled views. Each new view must differ from every prior
  view by 20 mm of camera-centre translation or 5 degrees of optical-axis angle by
  default; changing only its identifier fails before ray integration. Calibrated FK is
  the mapping pose authority; synchronized RTSI
  TCP is validation-only, with both poses and their residuals retained and independently
  reproduced during asset readback.
- Added a final-model ES68+D435i renderer and conservative depth-consistent robot
  self-mask. Measurements clearly in front of the rendered robot are retained as
  possible unknown surfaces; matching or farther measurements are removed so a stereo
  dropout cannot ray-clear through the robot. Removed pixels and their occluded rays
  remain `UNKNOWN` rather than being cleared as free space.
- Added immutable sparse occupancy assets with `FREE`, `OCCUPIED`, and implicit
  `UNKNOWN` voxels, an explicit `UNMAPPED/MAPPING/MAP_READY/STALE` lifecycle, per-frame
  quality arrays, hash-chained evidence, mapping-context binding, and read-time
  reproduction of masks, integration and snapshots. Out-of-grid and unknown space are
  fail-closed. A voxel needs three independent FREE votes by default, while OCCUPIED
  remains dominant; map freshness starts at the first frame of a complete rebuild cycle.
  Occupancy asset schema 6, snapshot format 4, and mapping-context schema 4 additionally
  retain the supporting camera poses, FK flange pose, predicted/observed TCP poses and
  flange-primary camera chain. The reader re-runs packaged ES68 FK from every stored
  joint vector before accepting them.
- Added strict source-to-motion semantic verification. The full occupancy reader
  reproduces raw-session integrity, user stereo calibration and rectification, official
  FoundationStereo source/checkpoint/configuration, self masking, integration and active
  robot geometry before issuing a typed attestation. Replay has no attestation. The
  occupancy checker, motion-preflight schema 5, one-shot permit and guarded executor bind
  that exact proof; protocol fakes, mutable snapshots and metadata changes fail closed.
- Added occupancy-aware motion preflight. The artifact binds the occupancy sequence,
  content hash and freshness horizon together with the complete ES68+D435i motion-model
  contract and ServoJ runtime configuration. Offline `occupancy build-replay` output is
  deliberately sealed `STALE`, so it can exercise storage and visualization but cannot
  satisfy motion preflight.
- Added a self-contained supervisory snapshot bridge and PySide6 replay console for the
  historical robot/camera scene, occupancy, current/fused blade point clouds, sensor
  evidence, copied provenance manifests and blocking events. Exact collision meshes are
  shown only after the active final model reproduces the historical geometry hash, and
  planned TCP targets only after canonical preflight replay; no continuous actual TCP
  trace is claimed. The GUI is read-only, always labels replay as `REPLAY/BLOCKED`, and
  exposes no approval or motion command.
- Preserved the physical-release boundary. A missing or unready final ES68+D435i STL
  manifest fails closed. Robot-self-masked volume remains unknown and can block the
  current bounding-sphere occupancy query; robot/environment paths are still evaluated
  at discrete joint samples rather than by an exact swept mesh. The native real-time
  coordinator has not yet been implemented or hardware-verified. These are blocking
  items, not merely performance optimizations.
- Sealed the public Elite motion methods behind the guarded executor's private capability.
  Even that path re-derives the exact ServoJ stream and rechecks every command segment;
  the missing continuous swept-mesh and swept-occupancy proofs still stop it before driver
  preparation.
- Verified this increment with the complete repository regression suite (`406 passed`),
  repository-wide Ruff checks, bytecode compilation, CLI smoke checks, and a locked-
  dependency consistency check.

## 2026-08-27 — native-depth validation infrastructure and legacy baseline

- Added `evaluate native-overlap`. The current schema-2 implementation validates
  synchronized D435i native depth transformed by the authoritative
  `FK(joints + zero offsets) · flange_T_left_ir · left_ir_T_depth` chain. Symmetric
  projective residuals reject depth edges, invalid pixels, occlusions, and field-of-view
  loss without applying registration corrections.
- Added explicit thresholds for projected support, same-surface inliers, median/RMSE/P95,
  5 mm agreement, and camera-pose observability. A deliberately wrong rotating hand-eye
  offset is covered by regression tests and must fail.
- Added bounded point-to-plane ICP as diagnostic evidence only. The correction cannot
  affect primary metrics, pass/fail, overlay points, or active calibration files.
- Added append-only, fully recomputable assets with source/config/hand-eye hashes,
  per-pair residual arrays, CSV metrics, a coloured base-frame PLY, and three-view PNG.
- The first five-view ES68/D435i run passed without ICP: median errors 1.220–1.424 mm,
  RMSE 2.070–2.470 mm, P95 4.205–4.993 mm, 5 mm agreement 95.02–97.36%, over a
  188.01 mm/23.683 degree pose span. Evidence is retained under
  `data/validations/native_overlap_20260827_static_v1`. That retained report is schema 1
  and explicitly used the legacy TCP-primary
  `base_T_tcp · tcp_T_left_ir · left_ir_T_depth` chain. Its numbers are a historical
  baseline, not validation of the current FK-authority path; the raw sessions must be
  reprocessed to a separate schema-2 output before making that claim.

## 2026-08-27 — native-depth validation acquisition override

- Added `--emitter/--no-emitter` to synchronized `acquire snapshot` and standalone
  `camera capture`. The override is command-scoped, leaves the hand-eye/stereo default
  configuration unchanged, and is preserved in synchronized session configuration
  snapshots so native-depth experiments remain reproducible.

## 2026-08-27 — Park+BA ES68/D435i hand-eye calibration closure

- Rebuilt the PySide6 hand-eye application as an idle-first, operator-controlled
  workflow: devices connect only after **Start**, `C` accepts one synchronized pose,
  Backspace recoverably excludes the last pose, and raw left IR plus detected ChArUco
  corners remain visible side by side.
- Isolated acquisition from preview analysis with a latest-frame mailbox and a reused
  detector. Slow corner processing can drop preview frames but cannot build an event
  backlog or change which full-resolution frame is atomically saved.
- Made Park-Martin the default initializer and retained the HoloRobot-aligned joint
  LM/BA refinement of `flange_T_left_ir` and fixed `base_T_target`, with live
  motion-observability, pose-novelty, synchronization, PnP, and ES68 FK/TCP gates.
- Locked robot-pose semantics to the HoloRobot ES68 reference: recorded joint angles
  drive the copied 709-pose calibrated FK and produce solver `base_T_flange`; RTSI
  `base_T_tcp` is validation-only. Persisted samples/results state this role explicitly,
  and a regression test proves changing controller TCP observations cannot change the
  solved transform.
- Added a strict held-out stage using at least five new poses. It evaluates the fixed
  candidate with board-closure and corner-reprojection metrics without refitting; only
  a passing report is atomically published to
  `data/calibrations/es68_left_ir_hand_eye_active.yaml`.
- Added `calibration hand-eye-validate-gui` for later supplemental evidence against an
  already completed schema-2 result. The new session hash-binds the unchanged candidate,
  stereo calibration and target, exposes no training/solve controls, never invokes Park
  or BA, and fails before hardware connection if provenance differs.
- Added unique, append-only hand-eye digital-asset sessions that copy and hash-bind the
  ChArUco target, D435i stereo calibration, packaged HoloRobot ES68 kinematics and
  flange/TCP offset, settings, raw/audit images, samples, candidate, validation attempts,
  and final result. A disconnected nonempty run is sealed instead of being mixed into a
  new session.
- Added regression coverage for bounded preview delivery and both pass/fail fixed-
  parameter validation geometry. The offline solver now also defaults to Park+BA but
  deliberately writes only a candidate; GUI-held-out validation controls publication.

## 2026-08-27 — independent D435i IR stereo validation

- Added the `calibration stereo-validate-gui` workflow with an idle startup window and
  three explicit operator steps: connect, save a synchronized hold-out pair, and run
  fixed-parameter offline validation.
- Added a validation-only digital asset schema that copies and SHA-256 binds the
  ChArUco target and exact stereo calibration, atomically appends raw Y8 pairs, records
  D435i identity/timestamps, and explicitly certifies that no calibration refit was
  performed.
- Added offline ChArUco detection, calibrated image/point rectification, horizontal
  epipolar-line overlays with matched corner colours, per-pair evidence, and aggregate
  vertical-disparity RMSE/P95/max, monocular reprojection RMSE, and stereo-transfer
  RMSE metrics with recorded pass/fail thresholds.
- Added `calibration stereo-validate-assets` for processing a preserved session after
  acquisition, plus unit coverage for fixed-input provenance, successful ideal
  geometry, checksum tamper rejection, and calibration/stream resolution mismatch.

Last updated: 2026-08-29

This log distinguishes verified implementation from pending work. Commit history is the
authoritative fine-grained record; this page records the experiment-facing state.

## Non-negotiable constraints

- Python 3.12 with `uv`; Elite SDK is installed from the local CPython 3.12 wheel.
- Offline planning, coverage, calibration, acquisition and doctor commands do not execute
  exported robot poses. The production `scan run-unknown` entry point can move the ES68,
  but only after its complete science/timing/motion authority chain, live stop and
  stationarity checks, collision revalidation, exact per-segment operator confirmation and
  expiring one-shot permit have all succeeded. Default configuration remains fail-closed
  until the required physical acceptance assets and limits are supplied.
- Every exported view plan has `motion_authorized: false`.
- Raw synchronized observations are immutable; derived products use separate outputs.
- Thermal capture remains an explicit disabled placeholder until hardware is selected.
- `latex/`, model checkpoints, data, and local configuration are not committed.

## Completed and verified

- D435i IR stereo calibration now separates responsive raw acquisition from offline
  ChArUco detection and solving. Every operator-started run creates a unique append-only
  asset session;
  it records the copied board definition, device identity, synchronized frame provenance,
  raw pairs, detection overlays and accept/reject reasons, analysis attempts and final
  calibration under a SHA-256-bound manifest. The preview retains only the latest frame
  rather than accumulating GUI events, and completed sessions reject further writes.
  The GUI starts idle; only an explicit operator click on **开始** connects the camera,
  creates the session and starts sample statistics at zero.
- A successful D435i IR solve now atomically publishes the solver-accepted result to
  `data/calibrations/d435i_ir_active.yaml`, the fixed path used by the default runtime
  configuration. All later calibrated capture paths therefore consume the latest
  completed user result without manual path editing. Missing user calibration fails
  closed; the RealSense adapter no longer falls back to factory IR intrinsics or stereo
  extrinsics. A live D435i capture verified that the default path returned the published
  left/right focal lengths and 49.990 mm user-calibrated baseline exactly.
- ES68 read-only hardware bring-up now follows HoloRobot's RTSI ownership contract:
  the status adapter subscribes to an explicit output-variable list and passes an empty
  input recipe, so observation cannot claim speed-slider or I/O write channels.
- Added a HoloRobot-style, manifest-driven ES68+D435i articulated collision template.
  It reserves independent STL slots for all seven collision links and the flange-mounted
  camera/bracket assembly, records units and transforms explicitly, materializes the
  calibrated ES68 chain into a Pinocchio/FCL URDF, and fails closed until the operator
  supplies every mesh, validates the flange attachment, and marks the manifest ready.
- A schema-bound fine-plan inspection command now verifies every persisted candidate's
  transform, camera-to-target distance, optical alignment, incidence, projection,
  coarse-cloud visibility, adaptive bounds, duplicate-pose status, and bilateral region
  presence. It atomically exports JSON/CSV plus region-coloured PLY, OBJ camera frusta,
  and a three-projection SVG. The optional PySide6 orbit viewer provides side/region
  filters, view/normal toggles, selection highlighting, and rejection details. Inspection
  explicitly reports robot feasibility as unverified and never changes
  `motion_authorized: false`.
- Fine-scan planning now uses a baseline-plus-region-adaptive distance policy. The base
  footprint is derived from the user-calibrated left-IR intrinsics, baseline distance,
  image margin, and utilization factor; no fixed 80x60 mm production fallback remains.
  Each true-surface or fin patch searches an explicit validated distance interval,
  records its selected distance and nominal footprint, and must pass whole-patch image/
  depth projection plus coarse z-buffer visibility gates. High-curvature, boundary,
  fin-root, and fin-rim regions prefer the closest feasible distance; flat main/fin faces
  prefer the baseline. Infeasible patches are visibility-split to a bounded depth and
  then fail closed. Schema-4 coarse-model artifacts persist the per-candidate evidence.
- Paper-derived coarse-model reconstruction now consumes multiple existing
  pose-registered D435i views, assigns immutable front/back membership from achieved
  camera centres, voxel-fuses each side, and applies robust point-to-plane residual
  refinement with robot-pose regularization, hard correction bounds, and no cross-side
  correspondences.
- True curved-surface planning now implements improved Angle Criterion boundary evidence,
  supported outer-contour ordering, four topological junctions, robust endpoint-consistent
  3D B-splines for root/trailing/tip/leading boundaries, equal-arc sampling, and an
  invertible Coons-grid irregular surface domain with boundary snapping. Fit/fold gates
  have explicit recorded fallback or fail-closed behaviour. Front/back use a shared
  conservative base grid; each populated
  patch then receives a PCA OBB centre, spherical-histogram main normal, and optional
  curvature-adaptive split. Fine views use measured main normals and remain non-executable.
- The photographed specimen's fixed topology is now explicit: robust per-side main-height
  fitting and height/normal-seeded 3D region growth require one thin fin on the front and
  one on the back. Fin points are removed before the paper boundary fit. Each retained
  component has independently persisted face, attachment-root, and free-rim regions;
  face-normal, root-bisector, and rim views; independent coverage gates; and measured-fin
  thickness protection in TSDF. Missing fins, multiple significant protrusions, non-thin
  components, and sub-voxel protected bands fail closed.
- Bilateral sparse projective TSDF uses a measured-thickness-protected truncation band,
  integrates front/back independently, and extracts a triangle mesh with a pure NumPy
  marching-tetrahedra fallback. Calibrated pixel/intrinsic/pose metadata enables the
  optional locked Open3D scalable backend when installed.
- Real-surface coverage replaces proxy-plane bins at the coarse-model stage: each patch
  records sample coverage, residual RMSE, local-normal consistency, curvature, and
  explicit quality-gate reasons; four edge-region completion ratios and TSDF mesh
  boundary/watertight evidence are reported separately.
- `bbf reconstruct coarse-model` validates common hand-eye provenance, runs the full
  fusion/partition/view/TSDF/quality chain, and atomically writes source-bound,
  SHA-256-verified arrays and metadata with `motion_authorized: false`.

- HoloRobot-aligned ES68/D435i eye-in-hand workflow: the exact 709-pose calibrated ES68
  FK and flange-to-RTSI-TCP validation offset are separately packaged under `es68`;
  synchronized PySide6 capture uses only raw D435i `infrared/1` and user-calibrated
  intrinsics, records complete ChArUco/robot/timing evidence, gates FK/TCP agreement,
  solves `flange_T_left_ir` with Park-Martin plus joint SE(3) LM/BA, validates the fixed
  candidate on new poses, and publishes a flange-primary schema-2 artifact only after
  the held-out gates pass.
- PySide6 raw D435i IR stereo-calibration workflow using the stored 14x9 ChArUco target:
  synchronized Y8 capture without factory IR calibration access, offline independent
  Zhang initialization, joint stereo bundle adjustment, epipolar metrics, selectable
  radial2/Brown5/Rational8 distortion models, held-out automatic model comparison, and
  user-calibration YAML export/load with resolution checks.
- Read-only Elite RTSI state acquisition and controller MDH export.
- D435i synchronized infrared/native-depth capture with calibration snapshots.
- Atomic schema-v2 session writer and validated reader.
- Native-depth point-cloud initialization, conservative thin-blade proxy, bilateral
  partitioning, candidate generation/filtering/scoring, and non-executable plan export.
- Offline ES68 KDL endpoint IK validation with captured seed joints.
- Calibrated D435i stereo rectification with explicit frame-chain transforms.
- Official FoundationStereo source pinned as a Git submodule.
- Lazy FoundationStereo adapter with no implicit EdgeNeXt or DINOv2 network download;
  inference scale is converted back to full-resolution disparity pixel units.
- Rectified left/right valid-region filtering and disparity-to-metric-depth conversion.
- Atomic, checksummed stereo inference artifacts and `bbf stereo infer-session`.
- Offline Park-Martin/Tsai/Horaud/Andreff/Daniilidis initial solving, motion
  observability gates, fixed-target closure RMSE, atomic artifacts, and CLI integration.
- Identified ChArUco detection from raw stored left-IR frames, positive-depth IPPE pose
  selection, planar-ambiguity/reprojection gates, automatic sample extraction, and
  durable rejection reasons.
- FoundationStereo-depth proxy initialization with source-identity checks, correct
  `base_T_left_rectified` geometry, and an end-to-end raw-session-to-plan integration
  path. Initialization schema 5 records depth source/projection frame and reads schema 4.
- Bilateral per-patch coverage grids, independent front/back evidence, incomplete versus
  blocked replanning state, and immutable checksummed seed-coverage artifacts.
- Native and FoundationStereo pose-registered view artifacts, with source identity,
  checksummed clouds/masks, hand-eye provenance, duplicate-frame prevention, and
  immutable coverage-ledger append support.
- Coverage-driven next-view artifacts that cryptographically bind the source plan and
  ledger, re-derive their contents on read, distinguish completed/remaining/blocked
  patches, and explicitly forbid motion.
- Calibrated paired native/stereo depth comparison in `left_rectified`, including
  z-buffered native reprojection, shared-valid-pixel metrics, checksummed arrays, source
  verification, and explicit non-ground-truth interpretation.
- Manifest-driven depth aggregation with duplicate-frame rejection, view-balanced and
  pixel-pooled metrics, plus retained front/back and incidence-angle strata.
- Initialization schema 7 adds source-pose authority and TCP-validation evidence on top
  of the SHA-256, dtype, and shape manifests for base clouds, pixel provenance, and masks;
  the reader retains schema 4/5/6 compatibility.
- Achieved-pose experiment labeling composes robot, hand-eye, and rectification
  transforms to derive proxy side and incidence; ambiguous mid-plane/away-facing views
  are rejected, and generated manifests bind the fixed initialization metadata.
- Correct Elite KDL IK orientation encoding: the vendor plugin consumes roll/pitch/yaw,
  which is intentionally distinct from the controller TCP rotation-vector encoding.
- Exact vendor-convention MDH link origins, fail-closed capsule/workcell geometry,
  joint-limit checks, bounded-step discrete joint-space sampling, explicit ordered view-sequence
  validation, and immutable reports that always forbid motion.
- `bbf doctor` collision-readiness diagnostics enumerate missing radii, tool geometry,
  joint limits, and required workcell obstacles before path validation is attempted.
- View-plan schema 3 cryptographically binds endpoint-feasible IK solutions to their
  controller-specific MDH artifact and six joint-zero offsets; safety validation rejects
  legacy or mismatched endpoint provenance while retaining older geometry-only plans for
  read-only compatibility.
- Adapted the pinned HoloRobot structural resources (whose upstream package paths retain
  `cs68` identifiers) to the calibrated ES68 chain and D435i wrist geometry. Development
  fixtures exercise YAML/Pinocchio FK and FCL sampling, while production resolution now
  requires the separately accepted final ES68+D435i manifest and never falls back to the
  upstream-labelled meshes.
- HoloRobot-aligned Elite Dashboard/RTSI/EliteDriver lifecycle, RPY TCP convention,
  point trajectories, SpeedJ, ServoJ prewarm/hold/streaming, stop, and safety faults.
- Conservative linear-joint motion preflight using copied velocity limits, plus exact
  preflight-hash confirmation, expiring one-shot execution permits, live-start checks,
  and immediate collision revalidation. No motion command is exposed through the CLI.
- Added a library-level, FoundationStereo-only receding-horizon coordinator for the
  explicit stop/settle, single-view capture, inference, fresh-map rebuild, one-short-leg
  preflight, per-leg operator approval, execution, explicit stop, settle and recapture
  sequence. The coordinator starts with operator-guided bootstrap, reads each segment
  start from the live settled joints, freezes one fully attested occupancy generation
  during authorization/execution, and rejects concurrent perception/motion operations.
  It remains disabled by default, has no motion CLI, and correctly reaches
  `MOTION_BLOCKED` with the current production checkers because continuous mesh and
  occupancy sweeps are still unavailable.
- Added the concrete stop-scan FoundationStereo perception transaction. Every accepted
  view is a separately closed one-view raw session; inference source hashes are verified,
  robot stationarity is sampled throughout inference, and every occupancy generation is
  rebuilt from scratch from the still-fresh sliding source window before full semantic
  replay. Native RealSense depth is forbidden in this coordinator. The current concrete
  engine produces stereo and safety-occupancy assets; online reconstructed-view and
  coverage outputs remain integration work.
- Added inference-window stationarity evidence and an append-only stop-scan event store.
  Stationarity checks arbitrary sample-pair joint/TCP drift, goal error and independent
  clock duration rather than relying on sleep alone. The event API publishes each JSON
  path once, while a forward SHA-256 chain makes later filesystem tampering detectable;
  `run.json` is explicitly navigation-only and the reader replays event files without
  trusting that index.
- The exact coordinator protocol, asset boundaries and fail-closed states are documented
  in `docs/stop-and-capture-coordinator.md`; that document explicitly does not authorize
  hardware motion.
- Immutable ordered view-sequence motion-preflight schema-5 artifacts bind plan,
  initialization, occupancy, and motion-model hashes and re-derive the fail-closed report
  on read. The production path currently stops at missing continuous swept-mesh evidence
  before ServoJ generation; diagnostic-only library overrides do not create approval
  evidence. `bbf safety preflight-path` always writes `motion_authorized: false`.
- Mesh motion preflight persists calibrated `base_T_tcp` goals and bounded-step sequence
  cost evidence;
  configured workcell AABBs are clearance-expanded hpp-fcl geometry checked against
  the resolved ES68+D435i model. Missing required production geometry is blocking.
- The 2026-08-27 software baseline passed 206 tests, Ruff, an offline wheel build, the
  packaged-resource audit, and Elite/Pinocchio/hpp-fcl/trimesh import checks. Curved
  reconstruction remains software-verified on deterministic data and pending physical
  hardware validation; later commits add their own regression evidence.

## Software pre-acceptance status

- The ES68+D435i software path is complete to the pre-acceptance boundary: exact robot
  geometry, guarded one-segment execution, both conservative continuous sweep proofs,
  FoundationStereo-only stop/capture transactions, physical-source occupancy identity,
  bilateral coarse science, schema-5 promotion, reference-guided fine reconstruction,
  deterministic coverage selection, terminal reconstruction replay, append-only global
  sealing, exact-chain resume, and the public supervised CLI are integrated and covered by
  deterministic regression tests.
- Recovery and completion are evidence operations, not motion shortcuts. Resume restores
  no permit, approval, prepared trajectory, freshness or driver authority; a completed
  chain opens no hardware. Every motion-capable segment still requires a fresh accepted
  map, both continuous proofs and the operator's exact one-shot confirmation.
- Earlier entries below/above that describe missing continuous proofs, missing public
  composition, library-only fine staging, or an `INIT -> PREPARED -> FINE_STARTED` chain
  are retained as historical increment records and are superseded by the 2026-08-29
  closure entry. They must not be read as current status.
- No statement here claims physical accuracy, collision-envelope or GPU timing acceptance,
  unattended/autonomous robot operation, or completed thermal imaging.
- Final software verification on 2026-08-29 passed 986 tests, repository-wide Ruff,
  Python bytecode compilation, both bootstrap-script syntax checks, strict JSON-template
  parsing, local Markdown-link validation and `uv lock --check`.

## Remaining physical acceptance and future scope

1. On the target GPU machine, validate the pinned FoundationStereo source, checkpoint and
   model configuration; record worst-case inference and schema-5 handoff durations rather
   than using guessed budgets.
2. With an attended, low-speed, emergency-stop-ready ES68 setup, accept the final
   ES68+D435i STL/manifest, static-free regions, tracking/stop envelope, continuous-proof
   behavior and workcell bounds using known-safe and known-collision cases.
3. Acquire real front/back/fins datasets with traceable dimensional references. Measure
   the required distance/incidence envelope, mask precision/recall, depth errors, final
   surface/thickness/normal/hole metrics and selector behavior; archive the declaration
   through `bbf safety record-science-acceptance` and configure its exact path/ID.
4. Run `scan doctor --mode unknown` against those immutable acceptances, then execute only
   the attended per-segment approval protocol. Record bootstrap, map replay, preflight,
   operator response, segment execution, stop and handoff timing evidence.
5. Implement and separately calibrate/validate the thermal-camera acquisition,
   radiometric correction, hand-eye relationship and geometry-temperature fusion after
   the final sensor/SDK is selected. Thermal reconstruction remains future work.
