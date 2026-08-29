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
]
