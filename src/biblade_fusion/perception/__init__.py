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

__all__ = [
    "REFERENCE_PROJECTED_ALGORITHM",
    "BladeForegroundDiagnostics",
    "BladeForegroundMaskError",
    "BladeForegroundMaskResult",
    "reference_guided_blade_mask",
]
