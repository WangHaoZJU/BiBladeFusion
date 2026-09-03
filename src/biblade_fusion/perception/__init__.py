"""Perception algorithms and inference backends.

Concrete algorithms are imported from their submodules to keep perception independent
from the planning package's compatibility exports.
"""

from biblade_fusion.perception.blade_foreground import (
    REFERENCE_PROJECTED_ALGORITHM,
    BladeForegroundDiagnostics,
    BladeForegroundMaskError,
    BladeForegroundMaskResult,
    reference_guided_blade_mask,
)
from biblade_fusion.perception.bootstrap_foreground import (
    BOOTSTRAP_FOREGROUND_ALGORITHM,
    BootstrapForegroundConfig,
    BootstrapForegroundDiagnostics,
    BootstrapForegroundError,
    BootstrapForegroundResult,
    BootstrapSeed,
    bootstrap_blade_foreground,
)
from biblade_fusion.perception.coarse_foreground import (
    PROJECTED_COARSE_FOREGROUND_ALGORITHM,
    ProjectedCoarseForegroundDiagnostics,
    ProjectedCoarseForegroundGuide,
    ProjectedCoarseForegroundResult,
    projected_coarse_blade_foreground,
    projected_coarse_foreground_policy_sha256,
)

__all__ = [
    "REFERENCE_PROJECTED_ALGORITHM",
    "BladeForegroundDiagnostics",
    "BladeForegroundMaskError",
    "BladeForegroundMaskResult",
    "reference_guided_blade_mask",
    "BOOTSTRAP_FOREGROUND_ALGORITHM",
    "BootstrapForegroundConfig",
    "BootstrapForegroundDiagnostics",
    "BootstrapForegroundError",
    "BootstrapForegroundResult",
    "BootstrapSeed",
    "bootstrap_blade_foreground",
    "PROJECTED_COARSE_FOREGROUND_ALGORITHM",
    "ProjectedCoarseForegroundDiagnostics",
    "ProjectedCoarseForegroundGuide",
    "ProjectedCoarseForegroundResult",
    "projected_coarse_blade_foreground",
    "projected_coarse_foreground_policy_sha256",
]
