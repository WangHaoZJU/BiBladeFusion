"""Robot adapter errors."""


class RobotError(RuntimeError):
    """Base exception for robot integration failures."""


class RobotConfigurationError(RobotError):
    """Robot configuration is missing or invalid."""


class RobotConnectionError(RobotError):
    """The robot state channel could not be connected."""


class RobotNotConnectedError(RobotError):
    """A state read was requested before connecting."""


class RobotMotionDisabledError(RobotError):
    """Motion was requested while the configuration safety gate is disabled."""


class RobotNotEnabledError(RobotError):
    """A motion command was requested before the arm was enabled."""


class RobotReleasedError(RobotError):
    """An operation was requested after the arm resources were permanently released."""


class RobotCommandError(RobotError):
    """The controller rejected or could not complete a command."""


class RobotMotionTimeoutError(RobotCommandError):
    """A robot motion exceeded its configured timeout."""


class RobotMotionInterruptedError(RobotCommandError):
    """A robot motion was interrupted or completed unsuccessfully."""


class RobotHardwareFaultError(RobotCommandError):
    """The controller reported a safety state that forbids motion."""
