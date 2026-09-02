# CUDA occupancy ray integration — 2026-09-02

## Outcome and scope

The coarse occupancy mapper now has an explicit `cpu`/`cuda` ray-integration backend.
The CUDA path accelerates only the per-pixel Amanatides-Woo voxel traversal that dominated
the measured attempt-11 CPU profile. FoundationStereo remains its existing CUDA workload;
the two stages are sequential and are not CPU/GPU comparison branches during acquisition.

The CUDA backend does not change workspace bounds, robot self masking, valid-depth gates,
minimum independent-view votes, occupied-wins behavior, map state transitions, collision
checking, IK, or motion authority. CUDA unavailability, allocation/kernel failure, or a ray
that does not reach its endpoint raises `DepthIntegrationError`; there is no silent CPU
fallback.

## Exact algorithm boundary

For every source frame, CPU code retains the existing float64 endpoint and traversal-state
construction. It produces each ray's `current`, `target`, `step`, `t_max`, `t_delta`, hit
voxel and maximum-step bound. PyTorch transfers these arrays to CUDA and advances all rays
in parallel. Each visited in-bounds free voxel is written idempotently into one dense
per-source boolean bitmap. This preserves the existing rule that one physical source can
cast at most one free vote into a voxel. Hit voxels are collected separately and the
unchanged CPU merge applies occupied-wins and the fixed source order.

The tie rule remains float64
`1e-12 * max(1, abs(next_t))`, including simultaneous grid-face/edge/corner crossings.
The bitmap is returned in C-order linear voxel order before conversion to the existing set
representation. No approximate ray march, octree, probabilistic occupancy, or unordered
vote accumulation was introduced.

## Explicit selection

`configs/default.yaml` remains `cpu` so non-CUDA development and historical replay remain
portable. The eiai run must select CUDA explicitly; `configs/local.yaml` is intentionally
Git-ignored, so relying on a local uncommitted edit is not acceptable evidence.

```bash
/usr/bin/env -u PYTHONPATH .venv/bin/bbf scan doctor \
  --mode unknown \
  --experimental \
  --config configs/local.yaml \
  --ray-integration-backend cuda
```

The table must contain both `supervised_scan_elite_sdk PASS` and
`scan_occupancy_ray_backend PASS`, with the latter reporting deterministic CUDA DDA and at
least one CUDA device. The same flag must then be present on `scan run-unknown`. The
selected backend is serialized in the occupancy mapping context and therefore hash-bound
to the generated artifacts.

## Mandatory eiai equivalence gate

Before a motion-capable CUDA run, replay the immutable attempt-11 first source on eiai. The
script never connects to the robot or camera and never modifies the input experiment. It
reintegrates every stored source with both CPU and CUDA, checks the CPU snapshot against the
stored snapshot, then requires exact CPU/CUDA snapshot equality.

```bash
/usr/bin/env -u PYTHONPATH .venv/bin/python -B \
  scripts/validate_cuda_ray_integration.py \
  data/experiments/blade-placement-20260901-01-attempt-11 \
  /tmp/bbf-attempt11-cuda-ray-validation.json
```

Acceptance requires command exit code zero and `"all_exact": true`. The no-clobber report
records Git revision, source metadata SHA-256, Torch/CUDA/device identity, per-source CPU
and CUDA wall time, CUDA Event device time, peak CUDA allocation, speedup and exact snapshot
equality. A nonzero CUDA device time distinguishes actual GPU execution from a process that
merely holds model memory. Re-run with a new output filename; the script refuses to replace
prior evidence.

This gate proves the tested source window only. It does not authorize motion or replace the
existing workcell, IK, continuous-collision, runtime-timing or operator approval gates.

## Environment rule

A Git pull does not require `uv sync`. If the environment genuinely must be rebuilt, use
`scripts/bootstrap-gpu.sh` with the measured Elite SDK wheel path. It installs all locked
extras and then installs the proprietary wheel last, because an exact `uv sync` can remove
packages not declared in `pyproject.toml`.

## Acquisition evidence

Production timing now distinguishes `occupancy.cuda_ray_integration` from
`occupancy.cpu_ray_integration`. For the new three-frame run, retain every cycle's
`performance_timing.json` and `coarse_generation_timing.json`. Acceptance requires:

- all three frames use the CUDA span and never the CPU span;
- all three masks, occupancy artifacts and coarse generations complete normally;
- the three-frame cumulative times are compared with the prior attempt, without attributing
  FoundationStereo or operator annotation wait to DDA;
- CUDA errors, RTSI gaps, IK/workspace rejection, or any artifact mismatch remain blocking
  failures and are not bypassed as performance issues.
