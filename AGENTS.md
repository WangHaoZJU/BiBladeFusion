# BiBladeFusion project instructions

## Required context

Before planning, editing code, interpreting an experiment, or giving the operator a
hardware command, read these files in order:

1. `docs/PROJECT_MEMORY.md`
2. `docs/CURRENT_STATE.md`
3. `docs/DECISION_LOG.md`
4. `docs/EXPERIMENT_RUNBOOK.md` when hardware, deployment, capture, or acceptance is involved

Treat `docs/PROJECT_MEMORY.md` as the stable research contract and
`docs/CURRENT_STATE.md` as the latest operational checkpoint. If either conflicts with
the current code, Git state, immutable experiment evidence, or a new operator report,
do not silently choose one: identify the conflict, establish the newer evidence, and
update the documents in the same change.

## Project objective

The central objective is a paper-ready active measurement system for a thin-walled,
double-sided, finned blade using an ES68 robot, a D435i stereo camera, and
FoundationStereo. The main scientific contribution is robot viewpoint planning:
starting from one operator-selected initial view and one blade polygon, actively seek
views that add useful blade/fin information, have a valid IK solution, and have a safe
continuous path.

Do not let downstream reconstruction, thermal mapping, generic software architecture,
or production-release ceremony displace this objective.

## Implementation boundaries

- Keep robot geometry, environment safety geometry, and blade science geometry separate.
- Robot self-collision and robot geometry use the active URDF and original collision STL
  meshes through Pinocchio/HPP-FCL. Do not replace a link with one large bounding sphere
  for a final collision decision.
- The full-scene occupancy map is safety evidence. Blade ROI/support data is scientific
  evidence. Never crop the safety occupancy input to the blade ROI, and never feed the
  full scene directly into the blade reconstruction.
- Candidate distance and incidence angle are adaptive search variables. Do not restore a
  fixed standoff band or a hard `+-15 degree` fin-view rule as the method definition.
- Collision, IK, workspace, and path validity are vetoes. They do not create positive
  scientific information gain.
- Preserve real collision and controller-stop checks, but do not add arbitrary wall-clock
  expiry gates, duplicated validation, or acceptance ceremony merely for conservatism.
  Every blocker must protect a stated physical or data-integrity invariant.

## HoloRobot reference-first rule

The operator also owns and developed the reference project at
`~/Documents/HoloRobot` (normally `/home/vale/Documents/HoloRobot` on vale). Before
designing or changing robot control, robot-state sampling, visual feedback, ServoJ
streaming, stop/recovery behavior, kinematics integration, or occupancy-map computation:

1. Locate and read the corresponding HoloRobot implementation and tests.
2. Compare its thread/process ownership, SDK connection lifetime, state transitions,
   timing, coordinate frames, and failure handling with BiBladeFusion.
3. Prefer transplanting the proven HoloRobot implementation or the smallest compatible
   adaptation when that capability already exists.
4. Design new behavior only for requirements genuinely absent from HoloRobot, especially
   blade-specific proxy, information gain, ROI propagation, and NBV policy.
5. Record any deliberate divergence and its reason in `docs/DECISION_LOG.md`.

Do not continue symptom-by-symptom patching before this comparison. Reuse is authorized
by the operator; still adapt interfaces and dependency versions explicitly rather than
copying code without checking its contracts.

## Hardware-test discipline

Before asking the operator to repeat a physical run:

1. Check whether HoloRobot already implements the affected control or perception path.
2. Reproduce with saved data or a focused unit/integration test when possible.
3. Trace the complete failing call path, including recovery, revalidation, permit
   ownership, controller state, and cleanup when relevant.
4. Run the smallest relevant test set and record the result.
5. State the exact expected transition and the exact evidence to collect on failure.
6. Use a new `run_id` and output directory; reuse the `placement_id` only if the blade and
   fixture have not moved.

Never ask for another long hardware run simply to discover an error that can be found by
code inspection or replay.

## Branch and data ownership

- Robot planning and unknown-blade fixes belong on `main` unless the operator says
  otherwise.
- Do not mix changes from the thermal-camera feature worktree or branch into `main`.
- Do not overwrite or delete experiment data, acceptance assets, operator configuration,
  or unrelated dirty-worktree changes.
- `configs/local.yaml` is deployment-specific and normally remains outside committed
  defaults. Record important deployed values and immutable acceptance IDs in
  `docs/CURRENT_STATE.md` without pretending they are repository defaults.

## Continuity updates

After a material design decision, accepted configuration change, new hardware result,
fix, or commit:

- update `docs/CURRENT_STATE.md` with the newest checkpoint and next action;
- append the rationale to `docs/DECISION_LOG.md` when the decision changes or clarifies
  the method;
- update `docs/EXPERIMENT_RUNBOOK.md` when an operator command or recovery procedure
  changes;
- keep `docs/PROJECT_MEMORY.md` stable unless the research contract itself changes.

Do not mark an unresolved hardware failure as fixed. Distinguish code verification,
offline replay, and physical verification explicitly.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **BiBladeFusion** (24798 symbols, 34922 relationships, 259 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/BiBladeFusion/context` | Codebase overview, check index freshness |
| `gitnexus://repo/BiBladeFusion/clusters` | All functional areas |
| `gitnexus://repo/BiBladeFusion/processes` | All execution flows |
| `gitnexus://repo/BiBladeFusion/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
