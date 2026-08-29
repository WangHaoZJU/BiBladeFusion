from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from biblade_fusion.core.settings import FineFinalizationConfig
from biblade_fusion.perception.fusion import FusedBladeCloud, RegisteredCloudView
from biblade_fusion.perception.surface import SurfaceRegion
from biblade_fusion.perception.tsdf import (
    BilateralTSDFResult,
    SparseTSDFVolume,
    TriangleMesh,
)
from biblade_fusion.planning.surface_coverage import (
    SurfacePatchQuality,
    SurfaceQualityReport,
)
from biblade_fusion.planning.views import BladeSide
from biblade_fusion.workflows.fine_finalization import _gate_report


def _fused() -> FusedBladeCloud:
    front = np.array(
        [[0.0, 0.0, 0.01], [0.01, 0.0, 0.01], [0.0, 0.01, 0.01]]
    )
    back = front.copy()
    back[:, 2] = -0.01
    return FusedBladeCloud(
        np.vstack((front, back)),
        np.vstack((np.tile([0.0, 0.0, 1.0], (3, 1)), np.tile([0.0, 0.0, -1.0], (3, 1)))),
        np.array([1, 1, 1, -1, -1, -1], dtype=np.int8),
        np.zeros(3),
        np.eye(3),
        0.02,
        (),
    )


def _mesh(*, hole: bool = False, omit_back: bool = False) -> TriangleMesh:
    tetra = np.array(
        [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.0, 0.0, 0.01]]
    )
    triangles = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=np.int32)
    if hole:
        triangles = triangles[:-1]
    vertices = tetra
    sides = np.ones(len(triangles), dtype=np.int8)
    if not omit_back:
        vertices = np.vstack((vertices, tetra + [0.03, 0.0, 0.0]))
        back_triangles = np.array(
            [[4, 5, 6], [4, 7, 5], [5, 7, 6], [6, 7, 4]], dtype=np.int32
        )
        if hole:
            back_triangles = back_triangles[:-1]
        triangles = np.vstack((triangles, back_triangles))
        sides = np.concatenate(
            (np.ones(len(sides), dtype=np.int8), -np.ones(len(back_triangles), dtype=np.int8))
        )
    return TriangleMesh(vertices, triangles, sides)


def _tsdf(mesh: TriangleMesh) -> BilateralTSDFResult:
    indices = np.array([[0, 0, 0]], dtype=np.int32)
    front = SparseTSDFVolume(1, np.zeros(3), 0.001, 0.002, indices, np.zeros(1), np.ones(1))
    back = SparseTSDFVolume(-1, np.zeros(3), 0.001, 0.002, indices, np.zeros(1), np.ones(1))
    return BilateralTSDFResult(front, back, mesh, 0.002)


def _terminal_and_quality(mesh: TriangleMesh, *, two_faces: bool = True):
    patches = []
    qualities = []
    for side in (BladeSide.FRONT, BladeSide.BACK):
        for region in (
            SurfaceRegion.SURFACE,
            SurfaceRegion.FIN_FACE,
            SurfaceRegion.FIN_ROOT,
            SurfaceRegion.FIN_FREE_EDGE,
        ):
            patch_id = f"{side.value}_{region.value}"
            patches.append(patch_id)
            qualities.append(
                SurfacePatchQuality(
                    patch_id,
                    side,
                    region,
                    10,
                    10,
                    1.0,
                    0.0,
                    1.0,
                    0.0,
                    True,
                    (),
                )
            )
    surface = SimpleNamespace(
        fin_components=tuple(
            SimpleNamespace(side=side, two_faces_observed=two_faces)
            for side in (BladeSide.FRONT, BladeSide.BACK)
        )
    )
    terminal = SimpleNamespace(required_patch_ids=tuple(patches), surface=surface)
    quality = SurfaceQualityReport(
        tuple(qualities),
        1.0,
        {},
        len(mesh.triangles),
        mesh.boundary_edge_count,
        0 if mesh.boundary_edge_count == 0 else 2,
        bool(len(mesh.triangles)) and mesh.boundary_edge_count == 0,
    )
    return terminal, quality


def _views(*, only_front: bool = False) -> tuple[RegisteredCloudView, ...]:
    points = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]])
    values = [RegisteredCloudView("front", points, np.array([0.0, 0.0, 0.3]))]
    if not only_front:
        values.append(RegisteredCloudView("back", points, np.array([0.0, 0.0, -0.3])))
    return tuple(values)


def _report(
    *,
    hole: bool = False,
    omit_back: bool = False,
    only_front: bool = False,
    two_faces: bool = True,
):
    mesh = _mesh(hole=hole, omit_back=omit_back)
    terminal, quality = _terminal_and_quality(mesh, two_faces=two_faces)
    return _gate_report(
        terminal,
        _views(only_front=only_front),
        _fused(),
        _tsdf(mesh),
        quality,
        FineFinalizationConfig(),
    )


def test_terminal_gates_accept_bilateral_watertight_fin_reconstruction() -> None:
    report = _report()

    assert report.passed
    assert report.front_source_view_count == report.back_source_view_count == 1
    assert report.front_mesh_triangle_count == report.back_mesh_triangle_count == 4


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"only_front": True}, "back-side source"),
        ({"omit_back": True}, "back TSDF surface"),
        ({"two_faces": False}, "two independently observed faces"),
        ({"hole": True}, "hole gate"),
    ],
)
def test_terminal_gates_reject_missing_bilateral_fin_or_hole_evidence(
    kwargs: dict[str, bool],
    reason: str,
) -> None:
    report = _report(**kwargs)

    assert not report.passed
    assert any(reason in violation for violation in report.violations)
