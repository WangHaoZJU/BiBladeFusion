"""Robot adapter errors."""


class RobotError(RuntimeError):
    """Base exception for robot integration failures."""


class RobotConfigurationError(RobotError):
    """Robot configuration is missing or invalid."""


class RobotConnectionError(RobotError):
    """The robot state channel could not be connected."""


class RobotNotConnectedError(RobotError):
    """A state read was requested before connecting."""

