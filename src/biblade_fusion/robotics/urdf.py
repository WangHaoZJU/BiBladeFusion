"""Materialize the HoloRobot CS68 URDF with its wrist depth-camera attachment."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree

from biblade_fusion.robotics.cs68_model import Cs68ModelResources

D435I_MOUNT_LINK = "depth_camera_mount_link"
D435I_MOUNT_JOINT = "wrist_3-depth_camera_mount"
D435I_MOUNT_MESH = "meshes/cs68/collision/depth_camera_mount.stl"
D435I_MOUNT_COLLISION_ORIGIN_XYZ_M = (-0.0505, -0.031815, 0.0)


def _origin(parent: ElementTree.Element, xyz: tuple[float, float, float]) -> None:
    ElementTree.SubElement(
        parent,
        "origin",
        {
            "xyz": " ".join(f"{value:.12g}" for value in xyz),
            "rpy": "0 0 0",
        },
    )


def build_cs68_urdf(
    resources: Cs68ModelResources | None = None,
    *,
    include_d435i_mount: bool = True,
) -> str:
    """Return the copied HoloRobot flat URDF, optionally with its D435i mount."""

    resolved = resources or Cs68ModelResources.packaged()
    resolved.validate()
    root = ElementTree.parse(resolved.urdf_path).getroot()
    if not include_d435i_mount:
        return ElementTree.tostring(root, encoding="unicode")
    if root.find(f".//link[@name='{D435I_MOUNT_LINK}']") is not None:
        return ElementTree.tostring(root, encoding="unicode")

    link = ElementTree.SubElement(root, "link", {"name": D435I_MOUNT_LINK})
    collision = ElementTree.SubElement(link, "collision")
    _origin(collision, D435I_MOUNT_COLLISION_ORIGIN_XYZ_M)
    geometry = ElementTree.SubElement(collision, "geometry")
    ElementTree.SubElement(geometry, "mesh", {"filename": D435I_MOUNT_MESH})

    joint = ElementTree.SubElement(
        root, "joint", {"name": D435I_MOUNT_JOINT, "type": "fixed"}
    )
    _origin(joint, (0.0, 0.0, 0.0))
    ElementTree.SubElement(joint, "parent", {"link": "wrist_3_link"})
    ElementTree.SubElement(joint, "child", {"link": D435I_MOUNT_LINK})
    return ElementTree.tostring(root, encoding="unicode")


def write_cs68_urdf(
    output_path: Path,
    resources: Cs68ModelResources | None = None,
    *,
    include_d435i_mount: bool = True,
) -> Path:
    """Write a deterministic flat URDF for Pinocchio/FCL loading."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        build_cs68_urdf(resources, include_d435i_mount=include_d435i_mount),
        encoding="utf-8",
    )
    return output_path
