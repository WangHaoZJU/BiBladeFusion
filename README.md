# BiBladeFusion

**BiBladeFusion** is a robot-guided bilateral 3D geometry and thermal reconstruction
system for thin-walled blades.

The current development stage provides a Python 3.12 application, validated
configuration, safe read-only integration with an Elite CS68 robot, synchronized raw
stereo acquisition from an Intel RealSense D435i, reproducible session storage, a
FoundationStereo integration boundary, and a conservative single-view blade proxy.
Robot motion is disabled by default.

## Bootstrap

```bash
./scripts/bootstrap.sh
```

The bootstrap script creates the project virtual environment with `uv`, synchronizes
the locked dependencies, and installs the local Elite CS SDK wheel.

## Verify

```bash
uv run bbf version
uv run bbf doctor
uv run bbf stereo doctor
uv run pytest
uv run ruff check .
```

`bbf stereo doctor` intentionally fails until the official FoundationStereo source,
checkpoint, inference dependencies, and requested CUDA device are present. The main
project remains on Python 3.12; the upstream Python 3.11 environment is treated as a
tested baseline, not as a hard-coded interpreter restriction.

## Read-only acquisition

Set `robot.robot_ip` in a Git-ignored `configs/local.yaml`, then run:

```bash
uv run bbf robot status --config configs/local.yaml
uv run bbf camera list
uv run bbf acquire snapshot --config configs/local.yaml --view-id seed
```

The synchronized snapshot brackets the D435i capture with two RTSI robot states and
rejects it when timing or stationary-state tolerances are exceeded. It does not issue
robot motion commands.

## Bilateral initialization

The initial visible-face point cloud is reduced to a density-balanced voxel cloud and
used to estimate the blade's two in-plane principal axes. The unseen side is explicitly
extruded away from the initial camera by `estimated_thickness_m`, with separate visible,
hidden, and tangential safety margins. Proxy construction refuses to continue when the
thickness prior is unset, the cloud is degenerate, or the initial view is too grazing.

`estimated_planar_extents_m` is optional and ordered as `(major, minor)`. When supplied,
the proxy uses the larger of the observed dimensions and these conservative prior
dimensions. The resulting proxy center is a planning-volume center, not a claim about
the blade's physical center of mass.
