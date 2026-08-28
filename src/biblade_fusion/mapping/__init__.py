"""Conservative online occupancy mapping for supervised robot motion."""

from biblade_fusion.mapping.integrator import (
    DepthIntegrationConfig,
    DepthIntegrationError,
    DepthRayIntegrator,
)
from biblade_fusion.mapping.occupancy import (
    OccupancyGridSpec,
    OccupancyMapState,
    OccupancySnapshot,
    OccupancyState,
    SphereQueryResult,
)
from biblade_fusion.mapping.robot_depth_renderer import Es68D435iRobotDepthRenderer
from biblade_fusion.mapping.self_mask import (
    RobotSelfMaskConfig,
    RobotSelfMaskReport,
    RobotSelfMaskResult,
    depth_consistent_robot_self_mask,
)
from biblade_fusion.mapping.serialization import (
    OccupancySnapshotFormatError,
    load_occupancy_snapshot,
    save_occupancy_snapshot,
)

__all__ = [
    "DepthIntegrationConfig",
    "DepthIntegrationError",
    "DepthRayIntegrator",
    "Es68D435iRobotDepthRenderer",
    "OccupancyGridSpec",
    "OccupancyMapState",
    "OccupancySnapshot",
    "OccupancySnapshotFormatError",
    "OccupancyState",
    "RobotSelfMaskConfig",
    "RobotSelfMaskReport",
    "RobotSelfMaskResult",
    "SphereQueryResult",
    "depth_consistent_robot_self_mask",
    "load_occupancy_snapshot",
    "save_occupancy_snapshot",
]
