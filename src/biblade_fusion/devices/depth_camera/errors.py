"""Depth-camera adapter errors."""


class DepthCameraError(RuntimeError):
    """Base exception for stereo camera failures."""


class DepthCameraConnectionError(DepthCameraError):
    """The configured camera could not be opened."""


class DepthCameraNotOpenError(DepthCameraError):
    """Capture was requested before opening the camera."""


class DepthCameraFrameError(DepthCameraError):
    """A synchronized camera frame was incomplete or invalid."""

