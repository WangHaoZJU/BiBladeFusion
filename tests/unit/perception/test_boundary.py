from __future__ import annotations

import numpy as np
import pytest

from biblade_fusion.core.settings import SurfacePartitionConfig
from biblade_fusion.perception.boundary import (
    BoundaryModelError,
    BoundaryName,
    boundary_driven_coordinates,
    fit_blade_boundary,
)
from biblade_fusion.perception.fusion import FusedBladeCloud
from biblade_fusion.perception.surface import SurfacePartitionError, partition_curved_blade
from biblade_fusion.planning.views import BladeSide


def _irregular_surface() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    major = np.linspace(-0.13, 0.11, 65)
    fraction = (major - major.min()) / np.ptp(major)
    half_width = 0.047 - 0.017 * fraction + 0.004 * np.sin(2.0 * np.pi * fraction)
    centerline = 0.006 * np.sin(np.pi * fraction)
    points = []
    boundary = []
    for index, (x, width, offset) in enumerate(zip(major, half_width, centerline, strict=True)):
        transverse = np.linspace(-1.0, 1.0, 29)
        y = offset + width * transverse
        z = 0.012 * fraction[index] ** 2 + 0.002 * transverse**2
        points.extend(np.column_stack((np.full(len(y), x), y, z)))
        boundary.extend(
            (index in {0, len(major) - 1}) or item in {0, len(transverse) - 1}
            for item in range(len(transverse))
        )
    cloud = np.asarray(points, dtype=np.float64)
    planar = cloud[:, :2]
    return cloud, planar, np.asarray(boundary, dtype=np.bool_)


def _config(**updates: object) -> SurfacePartitionConfig:
    values = {
        "minimum_points_per_side": 100,
        "minimum_patch_points": 10,
        "derive_footprint_from_intrinsics": False,
        "usable_footprint_m": (0.08, 0.06),
        "boundary_min_points_per_curve": 8,
        "boundary_control_points": 10,
        "boundary_max_fit_rmse_m": 0.004,
        "boundary_huber_delta_m": 0.002,
        "fin_mode": "disabled",
    }
    values.update(updates)
    return SurfacePartitionConfig(**values)


def test_four_robust_splines_parameterize_an_irregular_outline() -> None:
    points, planar, boundary = _irregular_surface()
    # Isolated AC false positives must not become topological corners.
    outliers = np.array([[0.25, 0.18, 0.0], [-0.28, -0.16, 0.0]])
    points = np.vstack((points, outliers))
    planar = np.vstack((planar, outliers[:, :2]))
    boundary = np.concatenate((boundary, np.ones(len(outliers), dtype=np.bool_)))

    model = fit_blade_boundary(points, planar, boundary, BladeSide.FRONT, _config())

    assert tuple(curve.name for curve in model.curves) == (
        BoundaryName.ROOT,
        BoundaryName.TRAILING_EDGE,
        BoundaryName.TIP,
        BoundaryName.LEADING_EDGE,
    )
    assert model.corners_m[:, 0].min() < -0.11
    assert model.corners_m[:, 0].max() > 0.09
    assert model.fit_rmse_m < 0.004
    for index, curve in enumerate(model.curves):
        assert np.allclose(curve.evaluate([0.0])[0], model.corners_m[index], atol=1e-12)
        assert np.allclose(curve.evaluate([1.0])[0], model.corners_m[(index + 1) % 4], atol=1e-12)
        samples = curve.sample_by_arc_length(12)
        lengths = np.linalg.norm(np.diff(samples, axis=0), axis=1)
        assert np.std(lengths) / np.mean(lengths) < 0.12

    progress = np.linspace(0.0, 1.0, 21)
    expected = {
        BoundaryName.ROOT: np.column_stack((np.zeros(21), progress)),
        BoundaryName.TRAILING_EDGE: np.column_stack((progress, np.ones(21))),
        BoundaryName.TIP: np.column_stack((np.ones(21), 1.0 - progress)),
        BoundaryName.LEADING_EDGE: np.column_stack((1.0 - progress, np.zeros(21))),
    }
    for curve in model.curves:
        boundary_coordinates = boundary_driven_coordinates(
            model,
            curve.sample_by_arc_length(21),
            np.zeros(3),
            np.eye(3)[:, :2],
        )
        assert np.allclose(boundary_coordinates, expected[curve.name], atol=0.035)

    coordinates = boundary_driven_coordinates(
        model,
        points[: -len(outliers)],
        np.zeros(3),
        np.eye(3)[:, :2],
    )
    assert np.all((coordinates >= 0.0) & (coordinates <= 1.0))
    assert np.percentile(coordinates[:, 0], 2.0) < 0.03
    assert np.percentile(coordinates[:, 0], 98.0) > 0.97
    assert np.percentile(coordinates[:, 1], 2.0) < 0.03
    assert np.percentile(coordinates[:, 1], 98.0) > 0.97


def test_boundary_failure_is_explicit_and_configurable() -> None:
    points, _, _ = _irregular_surface()
    front = points + np.array([0.0, 0.0, 0.004])
    back = points - np.array([0.0, 0.0, 0.004])
    normals = np.tile([0.0, 0.0, 1.0], (len(points), 1))
    fused = FusedBladeCloud(
        np.vstack((front, back)),
        np.vstack((normals, -normals)),
        np.concatenate((np.ones(len(points), dtype=np.int8), -np.ones(len(points), dtype=np.int8))),
        np.zeros(3),
        np.eye(3),
        0.008,
        (),
    )
    fallback = partition_curved_blade(
        fused,
        _config(
            voxel_size_m=0.003,
            maximum_points_per_side=2500,
            boundary_min_points_per_curve=1000,
            boundary_allow_fallback=True,
        ),
    )
    assert fallback.parameterization_methods == ("section_fallback", "section_fallback")
    assert all(fallback.boundary_fallback_reasons)

    with pytest.raises(SurfacePartitionError, match="boundary-curve parameterization failed"):
        partition_curved_blade(
            fused,
            _config(
                voxel_size_m=0.003,
                maximum_points_per_side=2500,
                boundary_min_points_per_curve=1000,
                boundary_allow_fallback=False,
            ),
        )


def test_boundary_model_rejects_insufficient_ac_evidence() -> None:
    points, planar, _ = _irregular_surface()
    with pytest.raises(BoundaryModelError, match="too few boundary candidates"):
        fit_blade_boundary(
            points,
            planar,
            np.zeros(len(points), dtype=np.bool_),
            BladeSide.FRONT,
            _config(),
        )
