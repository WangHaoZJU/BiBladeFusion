from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

import pytest
import yaml

from biblade_fusion.robotics.collision_template import (
    Es68D435iCollisionResources,
    Es68D435iCollisionTemplate,
    build_es68_d435i_collision_urdf,
)


def test_packaged_collision_template_is_explicitly_inactive() -> None:
    resources = Es68D435iCollisionResources.packaged_template()
    template = resources.load_template()

    assert template.ready is False
    assert template.mesh_units == "m"
    assert len(template.links) == 7
    assert template.attachment.parent_link == "flange"
    assert resources.collision_mesh_dir.joinpath("README.md").is_file()
    assert resources.manifest_path.name == "manifest.yaml"
    with pytest.raises(ValueError, match="not ready"):
        template.validate_assets()


def _ready_template(tmp_path: Path, *, units: str = "m") -> Es68D435iCollisionTemplate:
    packaged = Es68D435iCollisionResources.packaged_template()
    payload = yaml.safe_load(packaged.manifest_template_path.read_text(encoding="utf-8"))
    payload["ready"] = True
    payload["mesh_units"] = units
    manifest = tmp_path / "collision_models" / "es68_d435i" / "manifest.yaml"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    for spec in (*payload["links"].values(), payload["attachment"]):
        mesh = tmp_path / spec["mesh"]
        mesh.parent.mkdir(parents=True, exist_ok=True)
        mesh.write_bytes(b"solid placeholder\nendsolid placeholder\n")
    return Es68D435iCollisionTemplate.load(manifest, model_root=tmp_path)


def test_materialized_template_uses_es68_chain_and_articulated_meshes(tmp_path: Path) -> None:
    template = _ready_template(tmp_path, units="mm")

    root = ElementTree.fromstring(build_es68_d435i_collision_urdf(template))

    shoulder = root.find(".//joint[@name='shoulder_pan_joint']/origin")
    assert shoulder is not None
    assert float(shoulder.attrib["xyz"].split()[2]) == pytest.approx(0.161444008581)
    base_mesh = root.find(".//link[@name='base_link_inertia']/collision/geometry/mesh")
    assert base_mesh is not None
    assert base_mesh.attrib["filename"] == "meshes/es68_d435i/collision/base.stl"
    assert base_mesh.attrib["scale"] == "0.001 0.001 0.001"
    attachment_joint = root.find(".//joint[@name='flange-d435i_collision']")
    assert attachment_joint is not None
    assert attachment_joint.find("parent").attrib["link"] == "flange"


def test_ready_template_fails_closed_when_a_mesh_is_missing(tmp_path: Path) -> None:
    template = _ready_template(tmp_path)
    template.attachment.mesh_path.unlink()

    with pytest.raises(FileNotFoundError, match="d435i_assembly.stl"):
        build_es68_d435i_collision_urdf(template)
