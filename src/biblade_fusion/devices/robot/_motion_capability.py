"""Private capability shared only by the guarded execution boundary."""

from __future__ import annotations

# Identity, rather than equality, is the contract.  The symbol and module are
# deliberately private: normal application code receives only the read-only arm.
_GUARDED_MOTION_CAPABILITY = object()


def require_guarded_motion_capability(value: object) -> None:
    """Fail closed unless the caller is the guarded executor."""

    if value is not _GUARDED_MOTION_CAPABILITY:
        raise PermissionError(
            "Elite motion primitive requires the guarded-executor capability"
        )
