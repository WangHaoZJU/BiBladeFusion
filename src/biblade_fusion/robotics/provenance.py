"""Provenance for robot-stack code and resources imported from HoloRobot."""

from __future__ import annotations

from typing import Final

HOLOROBOT_SOURCE_REPOSITORY: Final = "git@github.com:wxyice/HoloRobot.git"
HOLOROBOT_SOURCE_COMMIT: Final = "93216a428cb8004382e9e39e5da7cd7bc6cbfffd"
HOLOROBOT_SOURCE_VERSION: Final = "0.2.0-rc.1"
HOLOROBOT_MODEL_LICENSE: Final = "Apache-2.0"
HOLOROBOT_TCP_ORIENTATION_CONVENTION: Final = "rpy_xyz_rad"


def robot_stack_provenance() -> dict[str, str]:
    """Return stable provenance suitable for persisted diagnostics and artifacts."""

    return {
        "source_repository": HOLOROBOT_SOURCE_REPOSITORY,
        "source_commit": HOLOROBOT_SOURCE_COMMIT,
        "source_version": HOLOROBOT_SOURCE_VERSION,
        "model_license": HOLOROBOT_MODEL_LICENSE,
        "elite_tcp_orientation_convention": HOLOROBOT_TCP_ORIENTATION_CONVENTION,
    }
