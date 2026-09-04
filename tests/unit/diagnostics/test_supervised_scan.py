from __future__ import annotations

from types import SimpleNamespace

from biblade_fusion.core.settings import AxisAlignedBoxConfig, load_settings
from biblade_fusion.diagnostics import supervised_scan
from biblade_fusion.diagnostics.types import CheckLevel


def test_elite_sdk_readiness_fails_before_hardware_when_module_is_missing(
    monkeypatch,
) -> None:
    settings = load_settings("configs/default.yaml")

    def missing(_name: str):
        raise ModuleNotFoundError("missing wheel")

    monkeypatch.setattr(supervised_scan, "import_module", missing)
    result = supervised_scan._elite_sdk_check(settings)

    assert result.level is CheckLevel.FAIL
    assert "missing wheel" in result.message
    assert result.details["hardware_connection_attempted"] is False


def test_cuda_ray_backend_readiness_fails_when_cuda_is_unavailable(monkeypatch) -> None:
    settings = load_settings("configs/default.yaml")
    settings.occupancy.ray_integration_backend = "cuda"
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: False,
            device_count=lambda: 0,
        )
    )
    monkeypatch.setattr(supervised_scan, "import_module", lambda _name: torch)

    result = supervised_scan._ray_integration_backend_check(settings)

    assert result.level is CheckLevel.FAIL
    assert "is_available" in result.message
    assert result.details["backend"] == "cuda"


def test_bootstrap_policy_requires_explicit_static_free_acceptance() -> None:
    settings = load_settings("configs/default.yaml")
    settings.robot.motion_enabled = True
    settings.stop_and_capture.enabled = True
    settings.stop_and_capture.maximum_segment_joint_delta_rad = 0.02
    settings.occupancy.enabled = True
    settings.occupancy.workspace_bounds_min_m = (-1.0, -1.0, -1.0)
    settings.occupancy.workspace_bounds_max_m = (1.0, 1.0, 1.0)

    result = supervised_scan._policy_check(settings)

    assert result.level is CheckLevel.FAIL
    assert "occupancy.accepted_static_free_aabbs" in result.details["missing"]
    assert "occupancy.accepted_static_free_acceptance_id" in result.details["missing"]
    assert "occupancy.accepted_static_free_acceptance_path" in result.details["missing"]

    settings.occupancy.accepted_static_free_aabbs = (
        AxisAlignedBoxConfig(
            name="accepted_robot_staging_envelope",
            minimum_m=(-0.5, -0.5, -0.5),
            maximum_m=(0.5, 0.5, 0.5),
        ),
    )
    settings.occupancy.accepted_static_free_acceptance_id = "a" * 64
    settings.occupancy.accepted_static_free_acceptance_path = (
        settings.project.data_root / "static-free-acceptance"
    )

    assert supervised_scan._policy_check(settings).level is CheckLevel.PASS


def test_collision_backend_audit_checks_both_real_backend_contracts(monkeypatch) -> None:
    class FakeMeshChecker:
        continuous_swept_volume_supported = True
        robot_geometry_hash = "a" * 64
        motion_model_contract_hash = "b" * 64

        @classmethod
        def from_es68_resources(cls, *args, **kwargs):
            del args, kwargs
            return cls()

    class FakeOccupancyChecker:
        continuous_swept_volume_supported = True

        def __init__(self, checker, provider):
            assert isinstance(checker, FakeMeshChecker)
            assert provider() is None

    resources = SimpleNamespace(
        load_active=lambda: SimpleNamespace(model_id="accepted-es68-d435i")
    )
    monkeypatch.setattr(
        supervised_scan.Es68D435iCollisionResources,
        "packaged_template",
        lambda: resources,
    )
    monkeypatch.setattr(
        supervised_scan,
        "Es68PinocchioCollisionChecker",
        FakeMeshChecker,
    )
    monkeypatch.setattr(
        supervised_scan,
        "OccupancyRobotCollisionChecker",
        FakeOccupancyChecker,
    )
    monkeypatch.setattr(supervised_scan, "ompl_available", lambda: True)

    result = supervised_scan._collision_backend_check(
        load_settings("configs/default.yaml")
    )

    assert result.level is CheckLevel.PASS
    assert result.name == "supervised_scan_holorobot_single_arm"
    assert result.details["online_path_validation_mode"] == (
        "holorobot_sampled_joint_v2"
    )
    assert result.details["holorobot_native_segment_samples"] == 5
    assert result.details["online_effective_maximum_sample_step_rad"] == 0.025
    assert result.details["offline_continuous_swept_mesh_supported"] is True
    assert result.details["offline_continuous_swept_occupancy_supported"] is True
    assert result.details["online_composite_planner_enabled"] is True
    assert result.details["ompl_fallback_available"] is True
    assert result.details["ompl_fallback_timeout_s"] == 1.0
    assert result.details["motion_authorized"] is False


def test_bootstrap_reference_check_never_requires_a_reference() -> None:
    result = supervised_scan._reference_check(
        load_settings("configs/default.yaml"),
        None,
        "bootstrap",
    )

    assert result.level is CheckLevel.PASS
    assert result.details["reference_required"] is False
    assert result.details["motion_authorized"] is False
