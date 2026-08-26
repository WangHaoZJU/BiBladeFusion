import numpy as np
import pytest

from biblade_fusion.core.pose import PoseSE3


def test_pose_inverse_and_compose() -> None:
    base_t_camera = PoseSE3.from_rotation_translation(
        "base",
        "camera",
        np.eye(3),
        [0.1, -0.2, 0.3],
    )

    identity = base_t_camera.compose(base_t_camera.inverse())

    assert identity.parent_frame == "base"
    assert identity.child_frame == "base"
    np.testing.assert_allclose(identity.matrix, np.eye(4), atol=1e-12)


def test_pose_transforms_single_and_batched_points() -> None:
    pose = PoseSE3.from_rotation_translation("base", "camera", np.eye(3), [1, 2, 3])

    np.testing.assert_allclose(pose.transform_points([0, 0, 0]), [1, 2, 3])
    np.testing.assert_allclose(
        pose.transform_points([[0, 0, 0], [1, 0, 0]]),
        [[1, 2, 3], [2, 2, 3]],
    )


def test_pose_rejects_disconnected_composition() -> None:
    base_t_camera = PoseSE3.identity("base", "camera")
    tool_t_sensor = PoseSE3.identity("tool", "sensor")

    with pytest.raises(ValueError, match="disconnected frames"):
        base_t_camera.compose(tool_t_sensor)


def test_pose_rejects_non_rigid_matrix() -> None:
    invalid = np.eye(4)
    invalid[0, 0] = 2.0

    with pytest.raises(ValueError, match="orthonormal"):
        PoseSE3("base", "camera", invalid)

