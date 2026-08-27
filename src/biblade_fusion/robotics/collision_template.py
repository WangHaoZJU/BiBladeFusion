"""Manifest-driven ES68 + D435i articulated collision-model template.

The supplied meshes are collision geometry, not a single assembled visual model:
every moving robot link needs an independent STL and the camera/bracket assembly is
attached to ``flange`` through a fixed joint.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree

import yaml

from biblade_fusion.robotics.cs68_model import (
    CS68_COLLISION_LINK_NAMES,
    CS68_JOINT_NAMES,
    Cs68ModelResources,
)
from biblade_fusion.robotics.es68_model import Es68KinematicModel

ES68_D435I_COLLISION_SCHEMA = "biblade_fusion.es68_d435i_collision.v1"


def _vector3(value: object, *, field: str) -> tuple[float, float, float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{field} must contain exactly three values")
    parsed = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in parsed):
        raise ValueError(f"{field} must contain finite values")
    return parsed  # type: ignore[return-value]


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _relative_asset(model_root: Path, value: object, *, field: str) -> tuple[str, Path]:
    raw = Path(str(value))
    if raw.is_absolute():
        raise ValueError(f"{field} must be relative to the collision model root")
    resolved_root = model_root.resolve()
    resolved = (resolved_root / raw).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"{field} escapes the collision model root")
    return raw.as_posix(), resolved


@dataclass(frozen=True, slots=True)
class CollisionMeshSpec:
    link_name: str
    mesh_uri: str
    mesh_path: Path
    origin_xyz_m: tuple[float, float, float]
    origin_rpy_rad: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class CollisionAttachmentSpec(CollisionMeshSpec):
    joint_name: str
    parent_link: str
    joint_xyz_m: tuple[float, float, float]
    joint_rpy_rad: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class Es68D435iCollisionTemplate:
    model_root: Path
    manifest_path: Path
    model_id: str
    ready: bool
    mesh_units: str
    mesh_scale: float
    links: tuple[CollisionMeshSpec, ...]
    attachment: CollisionAttachmentSpec
    max_parent_joint_hop: int
    minimum_clearance_m: float

    @classmethod
    def load(
        cls,
        manifest_path: Path,
        *,
        model_root: Path,
    ) -> Es68D435iCollisionTemplate:
        path = Path(manifest_path)
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        root = _mapping(payload, field="manifest")
        if str(root.get("schema")) != ES68_D435I_COLLISION_SCHEMA:
            raise ValueError(f"Unsupported collision-template schema in {path}")

        model_id = str(root.get("model_id") or "").strip()
        if not model_id:
            raise ValueError("model_id must be non-empty")
        units = str(root.get("mesh_units") or "").strip().lower()
        scales = {"m": 1.0, "mm": 0.001}
        if units not in scales:
            raise ValueError("mesh_units must be 'm' or 'mm'")

        links_raw = _mapping(root.get("links"), field="links")
        expected_links = set(CS68_COLLISION_LINK_NAMES)
        actual_links = {str(name) for name in links_raw}
        if actual_links != expected_links:
            missing = sorted(expected_links - actual_links)
            extra = sorted(actual_links - expected_links)
            raise ValueError(
                f"links must match ES68 collision links; missing={missing}, extra={extra}"
            )
        links = tuple(
            _parse_mesh_spec(
                link_name,
                _mapping(links_raw[link_name], field=f"links.{link_name}"),
                model_root=Path(model_root),
            )
            for link_name in CS68_COLLISION_LINK_NAMES
        )

        attachment_raw = _mapping(root.get("attachment"), field="attachment")
        attachment_link = str(attachment_raw.get("link_name") or "").strip()
        attachment_mesh = _parse_mesh_spec(
            attachment_link,
            attachment_raw,
            model_root=Path(model_root),
        )
        joint_origin = _mapping(attachment_raw.get("joint_origin"), field="attachment.joint_origin")
        attachment = CollisionAttachmentSpec(
            link_name=attachment_mesh.link_name,
            mesh_uri=attachment_mesh.mesh_uri,
            mesh_path=attachment_mesh.mesh_path,
            origin_xyz_m=attachment_mesh.origin_xyz_m,
            origin_rpy_rad=attachment_mesh.origin_rpy_rad,
            joint_name=str(attachment_raw.get("joint_name") or "").strip(),
            parent_link=str(attachment_raw.get("parent_link") or "").strip(),
            joint_xyz_m=_vector3(joint_origin.get("xyz_m"), field="attachment.joint_origin.xyz_m"),
            joint_rpy_rad=_vector3(
                joint_origin.get("rpy_rad"), field="attachment.joint_origin.rpy_rad"
            ),
        )
        if not attachment.link_name or not attachment.joint_name:
            raise ValueError("attachment link_name and joint_name must be non-empty")
        if attachment.parent_link != "flange":
            raise ValueError("D435i collision attachment must use parent_link='flange'")

        pair_filter = _mapping(root.get("pair_filter", {}), field="pair_filter")
        max_hop = int(pair_filter.get("max_parent_joint_hop", 1))
        if max_hop < 1:
            raise ValueError("pair_filter.max_parent_joint_hop must be at least 1")
        safety = _mapping(root.get("safety", {}), field="safety")
        clearance = float(safety.get("minimum_clearance_m", 0.0))
        if not math.isfinite(clearance) or clearance < 0.0:
            raise ValueError("safety.minimum_clearance_m must be finite and non-negative")

        return cls(
            model_root=Path(model_root).resolve(),
            manifest_path=path.resolve(),
            model_id=model_id,
            ready=bool(root.get("ready", False)),
            mesh_units=units,
            mesh_scale=scales[units],
            links=links,
            attachment=attachment,
            max_parent_joint_hop=max_hop,
            minimum_clearance_m=clearance,
        )

    def validate_assets(self) -> None:
        if not self.ready:
            raise ValueError(
                f"Collision template {self.manifest_path} is not ready; place and inspect all "
                "STLs, configure origins/units, then set ready: true"
            )
        missing = [
            str(spec.mesh_path)
            for spec in (*self.links, self.attachment)
            if not spec.mesh_path.is_file()
        ]
        if missing:
            raise FileNotFoundError("Missing ES68+D435i collision STL files: " + ", ".join(missing))


@dataclass(frozen=True, slots=True)
class Es68D435iCollisionResources:
    root: Path

    @classmethod
    def packaged_template(cls) -> Es68D435iCollisionResources:
        return cls(Path(__file__).resolve().parent / "resources" / "elite_cs")

    @property
    def manifest_template_path(self) -> Path:
        return self.root / "collision_models" / "es68_d435i" / "manifest.template.yaml"

    @property
    def manifest_path(self) -> Path:
        return self.root / "collision_models" / "es68_d435i" / "manifest.yaml"

    @property
    def collision_mesh_dir(self) -> Path:
        return self.root / "meshes" / "es68_d435i" / "collision"

    def load_template(self) -> Es68D435iCollisionTemplate:
        return Es68D435iCollisionTemplate.load(
            self.manifest_template_path,
            model_root=self.root,
        )

    def load_active(self) -> Es68D435iCollisionTemplate:
        template = Es68D435iCollisionTemplate.load(
            self.manifest_path,
            model_root=self.root,
        )
        template.validate_assets()
        return template


def _parse_mesh_spec(
    link_name: str,
    payload: Mapping[str, object],
    *,
    model_root: Path,
) -> CollisionMeshSpec:
    mesh_uri, mesh_path = _relative_asset(
        model_root,
        payload.get("mesh"),
        field=f"{link_name}.mesh",
    )
    origin = _mapping(payload.get("origin"), field=f"{link_name}.origin")
    return CollisionMeshSpec(
        link_name=str(link_name),
        mesh_uri=mesh_uri,
        mesh_path=mesh_path,
        origin_xyz_m=_vector3(origin.get("xyz_m"), field=f"{link_name}.origin.xyz_m"),
        origin_rpy_rad=_vector3(origin.get("rpy_rad"), field=f"{link_name}.origin.rpy_rad"),
    )


def _format_vector(values: Sequence[float]) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def _set_origin(element: ElementTree.Element, xyz: Sequence[float], rpy: Sequence[float]) -> None:
    origin = element.find("origin")
    if origin is None:
        origin = ElementTree.SubElement(element, "origin")
    origin.set("xyz", _format_vector(xyz))
    origin.set("rpy", _format_vector(rpy))


def _replace_link_collision(
    link: ElementTree.Element,
    spec: CollisionMeshSpec,
    *,
    mesh_scale: float,
) -> None:
    for collision in tuple(link.findall("collision")):
        link.remove(collision)
    collision = ElementTree.SubElement(link, "collision", {"name": f"{spec.link_name}_mesh"})
    _set_origin(collision, spec.origin_xyz_m, spec.origin_rpy_rad)
    geometry = ElementTree.SubElement(collision, "geometry")
    ElementTree.SubElement(
        geometry,
        "mesh",
        {
            "filename": spec.mesh_uri,
            "scale": _format_vector((mesh_scale, mesh_scale, mesh_scale)),
        },
    )


def build_es68_d435i_collision_urdf(
    template: Es68D435iCollisionTemplate,
    *,
    base_urdf_path: Path | None = None,
) -> str:
    """Build an ES68-calibrated articulated collision URDF after strict asset checks."""

    template.validate_assets()
    source = Path(base_urdf_path) if base_urdf_path else Cs68ModelResources.packaged().urdf_path
    root = ElementTree.parse(source).getroot()
    root.set("name", template.model_id)

    es68 = Es68KinematicModel.from_resources()
    for joint_name, segment in zip(CS68_JOINT_NAMES, es68.segments, strict=True):
        joint = root.find(f".//joint[@name='{joint_name}']")
        if joint is None:
            raise ValueError(f"Base URDF is missing ES68 joint {joint_name}")
        _set_origin(
            joint,
            (segment["x"], segment["y"], segment["z"]),
            (segment["roll"], segment["pitch"], segment["yaw"]),
        )

    for spec in template.links:
        link = root.find(f".//link[@name='{spec.link_name}']")
        if link is None:
            raise ValueError(f"Base URDF is missing collision link {spec.link_name}")
        _replace_link_collision(link, spec, mesh_scale=template.mesh_scale)

    attachment = template.attachment
    if root.find(f".//link[@name='{attachment.link_name}']") is not None:
        raise ValueError(f"Base URDF already contains attachment link {attachment.link_name}")
    link = ElementTree.SubElement(root, "link", {"name": attachment.link_name})
    _replace_link_collision(link, attachment, mesh_scale=template.mesh_scale)
    joint = ElementTree.SubElement(root, "joint", {"name": attachment.joint_name, "type": "fixed"})
    _set_origin(joint, attachment.joint_xyz_m, attachment.joint_rpy_rad)
    ElementTree.SubElement(joint, "parent", {"link": attachment.parent_link})
    ElementTree.SubElement(joint, "child", {"link": attachment.link_name})
    return ElementTree.tostring(root, encoding="unicode")


def write_es68_d435i_collision_urdf(
    output_path: Path,
    template: Es68D435iCollisionTemplate,
    *,
    base_urdf_path: Path | None = None,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        build_es68_d435i_collision_urdf(template, base_urdf_path=base_urdf_path),
        encoding="utf-8",
    )
    return output
