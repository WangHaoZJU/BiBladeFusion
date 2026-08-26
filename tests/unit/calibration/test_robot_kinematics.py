from pathlib import Path
from types import SimpleNamespace

import numpy as np

from biblade_fusion.calibration import (
    Cs68KinematicsModel,
    fetch_cs68_kinematics,
    load_cs68_kinematics,
    write_cs68_kinematics,
)


class FakePrimary:
    def __init__(self) -> None:
        self.disconnected = False

    def connect(self, robot_ip: str) -> bool:
        assert robot_ip == "192.0.2.10"
        return True

    def getPackage(self, package, timeout_ms: int) -> bool:
        assert timeout_ms == 1234
        package.dh_alpha_ = np.arange(6) * 0.1
        package.dh_a_ = np.arange(6) * 0.01
        package.dh_d_ = np.arange(6) * 0.02
        return True

    def disconnect(self) -> None:
        self.disconnected = True


def test_fetch_and_round_trip_controller_kinematics(tmp_path: Path) -> None:
    primary = FakePrimary()
    sdk = SimpleNamespace(
        PrimaryClientInterface=lambda: primary,
        KinematicsInfo=SimpleNamespace,
    )

    model = fetch_cs68_kinematics(
        "192.0.2.10",
        timeout_ms=1234,
        sdk_module=sdk,
    )
    path = write_cs68_kinematics(tmp_path / "cs68.yaml", model)
    loaded = load_cs68_kinematics(path)

    assert primary.disconnected is True
    np.testing.assert_allclose(loaded.dh_alpha_rad, np.arange(6) * 0.1)
    np.testing.assert_allclose(loaded.dh_a_m, np.arange(6) * 0.01)
    np.testing.assert_allclose(loaded.dh_d_m, np.arange(6) * 0.02)


def test_kinematics_model_copies_arrays() -> None:
    values = np.arange(6, dtype=float)
    model = Cs68KinematicsModel(values, values, values, "test")
    values[0] = 99

    assert model.dh_a_m[0] == 0
    assert model.dh_a_m.flags.writeable is False
