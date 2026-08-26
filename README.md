# BiBladeFusion

**BiBladeFusion** is a robot-guided bilateral 3D geometry and thermal reconstruction
system for thin-walled blades.

The first development stage provides a Python 3.12 application skeleton, environment
diagnostics, hardware abstractions, and safe read-only integration with an Elite CS68
robot and an Intel RealSense D435i. Robot motion is disabled by default.

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
uv run pytest
```

