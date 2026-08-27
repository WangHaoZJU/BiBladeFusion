"""Acquisition errors."""


class AcquisitionError(RuntimeError):
    """Base exception for synchronized acquisition failures."""


class AcquisitionRejectedError(AcquisitionError):
    """The captured bundle violated timing or stationary-state constraints."""
