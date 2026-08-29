"""Fail-closed foreground bootstrap for an unknown blade.

This module is deliberately independent from the schema-5 coarse-surface guided
foreground extractor. It is used only before such a surface exists. Automatic
selection joins neighbouring valid depths without erosion, rejects components
touching the valid depth-domain boundary, and refuses ambiguous scenes. A human
polygon/rectangle may instead be recorded either as a hard, deterministic mask or
as a component-selection hint.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

BOOTSTRAP_FOREGROUND_ALGORITHM = "depth_connected_boundary_exclusion_v1"


class BootstrapForegroundError(ValueError):
    """The observation cannot support a unique, trustworthy bootstrap mask."""


@dataclass(frozen=True, slots=True)
class BootstrapForegroundConfig:
    """Numerical policy for unknown-object foreground extraction."""

    minimum_depth_m: float = 0.15
    maximum_depth_m: float = 2.0
    maximum_neighbour_depth_jump_m: float = 0.030
    maximum_neighbour_relative_depth_jump: float = 0.035
    connectivity: Literal[4, 8] = 8
    boundary_margin_px: int = 2
    minimum_valid_pixels: int = 1_000
    minimum_component_pixels: int = 100
    minimum_mask_pixels: int = 500
    minimum_mask_fraction: float = 0.001
    maximum_mask_fraction: float = 0.70
    maximum_unseeded_ambiguity_ratio: float = 0.35
    minimum_seed_valid_pixels: int = 25
    minimum_seed_valid_fraction: float = 0.10
    minimum_component_hint_selection_fraction: float = 0.10

    def __post_init__(self) -> None:
        finite_nonnegative = (
            self.minimum_depth_m,
            self.maximum_depth_m,
            self.maximum_neighbour_depth_jump_m,
            self.maximum_neighbour_relative_depth_jump,
        )
        if not all(np.isfinite(value) and value >= 0.0 for value in finite_nonnegative):
            raise ValueError("Bootstrap depth thresholds must be finite and non-negative")
        if self.minimum_depth_m <= 0.0 or self.maximum_depth_m <= self.minimum_depth_m:
            raise ValueError("Bootstrap depth interval must be positive and ordered")
        if self.connectivity not in {4, 8}:
            raise ValueError("Bootstrap connectivity must be four or eight")
        if self.boundary_margin_px < 1:
            raise ValueError("Bootstrap boundary margin must be at least one pixel")
        integer_limits = (
            self.minimum_valid_pixels,
            self.minimum_component_pixels,
            self.minimum_mask_pixels,
            self.minimum_seed_valid_pixels,
        )
        if any(value < 1 for value in integer_limits):
            raise ValueError("Bootstrap pixel-count gates must be positive")
        fractions = (
            self.minimum_mask_fraction,
            self.maximum_mask_fraction,
            self.maximum_unseeded_ambiguity_ratio,
            self.minimum_seed_valid_fraction,
            self.minimum_component_hint_selection_fraction,
        )
        if any(not np.isfinite(value) or not 0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("Bootstrap fraction gates must lie in [0, 1]")
        if self.minimum_mask_fraction > self.maximum_mask_fraction:
            raise ValueError("Bootstrap mask-fraction interval is reversed")


@dataclass(frozen=True, slots=True)
class BootstrapSeed:
    """An immutable user annotation in rectified-left pixel coordinates.

    hard_roi declares the rasterised annotation to be the human foreground
    decision; every valid depth inside it is retained, including disconnected thin
    fin pixels. component_hint selects one depth-connected component and never
    clips that component to the annotation.
    """

    kind: Literal["polygon", "rectangle"]
    mode: Literal["hard_roi", "component_hint"]
    vertices_uv: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        vertices = tuple((float(u), float(v)) for u, v in self.vertices_uv)
        required = 2 if self.kind == "rectangle" else 3
        if self.kind not in {"polygon", "rectangle"} or len(vertices) < required:
            raise ValueError("Bootstrap seed kind or vertex count is invalid")
        if self.kind == "rectangle" and len(vertices) != 2:
            raise ValueError("A rectangle seed requires exactly two opposite corners")
        if self.mode not in {"hard_roi", "component_hint"}:
            raise ValueError("Bootstrap seed mode is unsupported")
        if not np.isfinite(np.asarray(vertices, dtype=np.float64)).all():
            raise ValueError("Bootstrap seed vertices must be finite")
        if self.kind == "rectangle" and (
            vertices[0][0] == vertices[1][0] or vertices[0][1] == vertices[1][1]
        ):
            raise ValueError("Bootstrap rectangle must have non-zero area")
        object.__setattr__(self, "vertices_uv", vertices)

    @classmethod
    def rectangle(
        cls,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        *,
        mode: Literal["hard_roi", "component_hint"] = "component_hint",
    ) -> BootstrapSeed:
        return cls("rectangle", mode, ((x0, y0), (x1, y1)))

    @classmethod
    def polygon(
        cls,
        vertices_uv: ArrayLike,
        *,
        mode: Literal["hard_roi", "component_hint"] = "hard_roi",
    ) -> BootstrapSeed:
        vertices = np.asarray(vertices_uv, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1:] != (2,):
            raise ValueError("Bootstrap polygon vertices must have shape (N, 2)")
        return cls("polygon", mode, tuple(tuple(item) for item in vertices.tolist()))


@dataclass(frozen=True, slots=True)
class BootstrapForegroundDiagnostics:
    image_pixel_count: int
    supplied_valid_pixel_count: int
    depth_valid_pixel_count: int
    component_count: int
    boundary_component_count: int
    boundary_component_pixel_count: int
    interior_component_count: int
    selected_component_count: int
    largest_interior_component_pixels: int
    second_interior_component_pixels: int
    ambiguity_ratio: float
    seed_pixel_count: int
    valid_seed_pixel_count: int
    seed_valid_fraction: float
    selected_seed_pixel_count: int
    selected_seed_fraction: float
    mask_pixel_count: int
    mask_fraction: float
    minimum_mask_depth_m: float
    median_mask_depth_m: float
    maximum_mask_depth_m: float
    left_intensity_mean: float
    left_intensity_std: float


@dataclass(frozen=True, slots=True)
class BootstrapForegroundResult:
    """Immutable mask, annotation, diagnostics and source-array identities."""

    mask: NDArray[np.bool_]
    seed_mask: NDArray[np.bool_]
    diagnostics: BootstrapForegroundDiagnostics
    config: BootstrapForegroundConfig
    seed: BootstrapSeed | None
    algorithm: str
    policy_sha256: str
    left_image_content_sha256: str
    depth_content_sha256: str
    valid_mask_content_sha256: str

    def __post_init__(self) -> None:
        mask = np.array(self.mask, dtype=np.bool_, copy=True)
        seed_mask = np.array(self.seed_mask, dtype=np.bool_, copy=True)
        if mask.ndim != 2 or seed_mask.shape != mask.shape:
            raise ValueError("Bootstrap mask and seed mask must have the same 2-D shape")
        if self.algorithm != BOOTSTRAP_FOREGROUND_ALGORITHM:
            raise ValueError("Unsupported bootstrap-foreground algorithm")
        if self.policy_sha256 != bootstrap_policy_sha256(self.config, self.seed):
            raise ValueError("Bootstrap foreground policy hash does not reproduce")
        for digest in (
            self.left_image_content_sha256,
            self.depth_content_sha256,
            self.valid_mask_content_sha256,
        ):
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError("Bootstrap input content hash is malformed")
        if int(np.count_nonzero(mask)) != self.diagnostics.mask_pixel_count:
            raise ValueError("Bootstrap diagnostics do not match mask pixels")
        if int(np.count_nonzero(seed_mask)) != self.diagnostics.seed_pixel_count:
            raise ValueError("Bootstrap diagnostics do not match seed pixels")
        mask.setflags(write=False)
        seed_mask.setflags(write=False)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "seed_mask", seed_mask)


@dataclass(frozen=True, slots=True)
class _Component:
    label: int
    pixel_count: int
    boundary_pixel_count: int
    seed_pixel_count: int

    @property
    def touches_boundary(self) -> bool:
        return self.boundary_pixel_count > 0


def array_content_sha256(value: ArrayLike) -> str:
    """Hash numeric array content together with shape and dtype."""

    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def bootstrap_seed_payload(seed: BootstrapSeed | None) -> dict[str, object] | None:
    if seed is None:
        return None
    return {
        "kind": seed.kind,
        "mode": seed.mode,
        "vertices_uv": [list(vertex) for vertex in seed.vertices_uv],
    }


def bootstrap_policy_sha256(
    config: BootstrapForegroundConfig,
    seed: BootstrapSeed | None,
) -> str:
    payload = {
        "algorithm": BOOTSTRAP_FOREGROUND_ALGORITHM,
        "config": asdict(config),
        "seed": bootstrap_seed_payload(seed),
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _rasterize_seed(seed: BootstrapSeed, shape: tuple[int, int]) -> NDArray[np.bool_]:
    height, width = shape
    vertices = np.asarray(seed.vertices_uv, dtype=np.float64)
    if (
        np.any(vertices[:, 0] < 0.0)
        or np.any(vertices[:, 0] > width - 1)
        or np.any(vertices[:, 1] < 0.0)
        or np.any(vertices[:, 1] > height - 1)
    ):
        raise BootstrapForegroundError("Bootstrap seed lies outside the rectified image")
    rows, columns = np.indices(shape, dtype=np.float64)
    if seed.kind == "rectangle":
        lower = np.minimum(vertices[0], vertices[1])
        upper = np.maximum(vertices[0], vertices[1])
        return (
            (columns >= lower[0]) & (columns <= upper[0]) & (rows >= lower[1]) & (rows <= upper[1])
        )

    # Even-odd rasterisation at integer pixel centres, with segment pixels included.
    inside = np.zeros(shape, dtype=np.bool_)
    on_boundary = np.zeros(shape, dtype=np.bool_)
    previous = vertices[-1]
    for current in vertices:
        x0, y0 = previous
        x1, y1 = current
        dx = x1 - x0
        dy = y1 - y0
        cross = (columns - x0) * dy - (rows - y0) * dx
        within = (
            (columns >= min(x0, x1))
            & (columns <= max(x0, x1))
            & (rows >= min(y0, y1))
            & (rows <= max(y0, y1))
        )
        on_boundary |= within & np.isclose(cross, 0.0, rtol=0.0, atol=1e-9)
        crossing = (y0 > rows) != (y1 > rows)
        denominator = dy if dy != 0.0 else np.finfo(np.float64).eps
        intersection_x = x0 + (rows - y0) * dx / denominator
        inside ^= crossing & (columns < intersection_x)
        previous = current
    return inside | on_boundary


def _valid_domain_boundary(
    valid: NDArray[np.bool_],
    margin: int,
) -> NDArray[np.bool_]:
    coordinates = np.argwhere(valid)
    if len(coordinates) == 0:
        return np.zeros_like(valid)
    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0)
    rows, columns = np.indices(valid.shape)
    return valid & (
        (rows < minimum[0] + margin)
        | (rows > maximum[0] - margin)
        | (columns < minimum[1] + margin)
        | (columns > maximum[1] - margin)
    )


def _depth_components(
    candidate: NDArray[np.bool_],
    depth_m: NDArray[np.float64],
    boundary: NDArray[np.bool_],
    seed_mask: NDArray[np.bool_],
    config: BootstrapForegroundConfig,
) -> tuple[NDArray[np.int32], tuple[_Component, ...]]:
    height, width = candidate.shape
    labels = np.full(candidate.shape, -1, dtype=np.int32)
    offsets = ((-1, 0), (0, -1), (0, 1), (1, 0))
    if config.connectivity == 8:
        offsets += ((-1, -1), (-1, 1), (1, -1), (1, 1))
    components: list[_Component] = []
    for start_flat in np.flatnonzero(candidate):
        start_v, start_u = divmod(int(start_flat), width)
        if labels[start_v, start_u] >= 0:
            continue
        label = len(components)
        labels[start_v, start_u] = label
        queue: deque[tuple[int, int]] = deque(((start_v, start_u),))
        count = boundary_count = seed_count = 0
        while queue:
            v, u = queue.popleft()
            count += 1
            boundary_count += int(boundary[v, u])
            seed_count += int(seed_mask[v, u])
            current_depth = depth_m[v, u]
            for dv, du in offsets:
                neighbour_v, neighbour_u = v + dv, u + du
                if (
                    neighbour_v < 0
                    or neighbour_v >= height
                    or neighbour_u < 0
                    or neighbour_u >= width
                    or not candidate[neighbour_v, neighbour_u]
                    or labels[neighbour_v, neighbour_u] >= 0
                ):
                    continue
                neighbour_depth = depth_m[neighbour_v, neighbour_u]
                threshold = max(
                    config.maximum_neighbour_depth_jump_m,
                    config.maximum_neighbour_relative_depth_jump
                    * min(current_depth, neighbour_depth),
                )
                if abs(current_depth - neighbour_depth) <= threshold:
                    labels[neighbour_v, neighbour_u] = label
                    queue.append((neighbour_v, neighbour_u))
        components.append(_Component(label, count, boundary_count, seed_count))
    return labels, tuple(components)


def bootstrap_blade_foreground(
    left_rectified: ArrayLike,
    depth_m: ArrayLike,
    valid_mask: ArrayLike,
    config: BootstrapForegroundConfig,
    seed: BootstrapSeed | None = None,
) -> BootstrapForegroundResult:
    """Extract one auditable unknown-blade mask without a background model."""

    left = np.asarray(left_rectified)
    original_depth = np.asarray(depth_m)
    depth = np.asarray(depth_m, dtype=np.float64)
    supplied_valid = np.asarray(valid_mask, dtype=np.bool_)
    if left.ndim != 2 or depth.ndim != 2 or supplied_valid.shape != depth.shape:
        raise BootstrapForegroundError("Bootstrap inputs must be shape-matched 2-D arrays")
    if left.shape != depth.shape or not np.issubdtype(left.dtype, np.number):
        raise BootstrapForegroundError("Rectified-left image must match depth and be numeric")
    if not np.isfinite(left).all():
        raise BootstrapForegroundError("Rectified-left image contains non-finite pixels")

    depth_valid = (
        supplied_valid
        & np.isfinite(depth)
        & (depth >= config.minimum_depth_m)
        & (depth <= config.maximum_depth_m)
    )
    valid_count = int(np.count_nonzero(depth_valid))
    if valid_count < config.minimum_valid_pixels:
        raise BootstrapForegroundError(
            f"Depth-valid support {valid_count} is below minimum_valid_pixels "
            f"{config.minimum_valid_pixels}"
        )
    seed_mask = np.zeros_like(depth_valid)
    if seed is not None:
        seed_mask = _rasterize_seed(seed, depth.shape)
        seed_count = int(np.count_nonzero(seed_mask))
        valid_seed_count = int(np.count_nonzero(seed_mask & depth_valid))
        if valid_seed_count < config.minimum_seed_valid_pixels:
            raise BootstrapForegroundError("Seed contains too few valid depth pixels")
        if valid_seed_count / seed_count < config.minimum_seed_valid_fraction:
            raise BootstrapForegroundError("Seed valid-depth fraction is below policy")

    candidate = (
        depth_valid if seed is None or seed.mode == "component_hint" else depth_valid & seed_mask
    )
    boundary = _valid_domain_boundary(depth_valid, config.boundary_margin_px)
    labels, components = _depth_components(candidate, depth, boundary, seed_mask, config)
    interior = tuple(
        component
        for component in components
        if not component.touches_boundary
        and component.pixel_count >= config.minimum_component_pixels
    )
    by_size = sorted(interior, key=lambda item: (-item.pixel_count, item.label))
    largest = by_size[0].pixel_count if by_size else 0
    second = by_size[1].pixel_count if len(by_size) > 1 else 0
    ambiguity = second / largest if largest else 0.0

    selected: tuple[_Component, ...]
    if seed is not None and seed.mode == "hard_roi":
        if any(component.touches_boundary for component in components):
            raise BootstrapForegroundError("Hard ROI reaches the valid depth-domain boundary")
        # The annotation is the human decision: do not erase small or disconnected
        # support that may be a narrow fin. The aggregate mask gates still apply.
        selected = components
        mask = candidate.copy()
    elif seed is not None:
        ranked = sorted(
            (component for component in interior if component.seed_pixel_count > 0),
            key=lambda item: (-item.seed_pixel_count, -item.pixel_count, item.label),
        )
        if not ranked:
            raise BootstrapForegroundError("Component hint does not select an interior component")
        selected = (ranked[0],)
        valid_seed_count = int(np.count_nonzero(seed_mask & depth_valid))
        if ranked[0].seed_pixel_count / valid_seed_count < (
            config.minimum_component_hint_selection_fraction
        ):
            raise BootstrapForegroundError("Component hint selection fraction is below policy")
        if len(ranked) > 1 and ranked[1].seed_pixel_count / ranked[0].seed_pixel_count > (
            config.maximum_unseeded_ambiguity_ratio
        ):
            raise BootstrapForegroundError("Component hint selects multiple ambiguous components")
        mask = labels == ranked[0].label
    else:
        if not by_size:
            raise BootstrapForegroundError(
                "No interior depth component remains after boundary exclusion"
            )
        if ambiguity > config.maximum_unseeded_ambiguity_ratio:
            raise BootstrapForegroundError(
                "Automatic bootstrap contains ambiguous interior components"
            )
        selected = (by_size[0],)
        mask = labels == by_size[0].label

    mask_count = int(np.count_nonzero(mask))
    mask_fraction = mask_count / mask.size
    if mask_count < config.minimum_mask_pixels:
        raise BootstrapForegroundError("Bootstrap mask is below minimum_mask_pixels")
    if not config.minimum_mask_fraction <= mask_fraction <= config.maximum_mask_fraction:
        raise BootstrapForegroundError("Bootstrap mask fraction is outside configured bounds")
    if np.any(mask & boundary):
        raise BootstrapForegroundError("Bootstrap mask reaches the valid depth-domain boundary")

    seed_count = int(np.count_nonzero(seed_mask))
    valid_seed_count = int(np.count_nonzero(seed_mask & depth_valid))
    selected_seed_count = int(np.count_nonzero(mask & seed_mask))
    mask_depth = depth[mask]
    diagnostics = BootstrapForegroundDiagnostics(
        image_pixel_count=int(mask.size),
        supplied_valid_pixel_count=int(np.count_nonzero(supplied_valid)),
        depth_valid_pixel_count=valid_count,
        component_count=len(components),
        boundary_component_count=sum(item.touches_boundary for item in components),
        boundary_component_pixel_count=sum(
            item.pixel_count for item in components if item.touches_boundary
        ),
        interior_component_count=len(interior),
        selected_component_count=len(selected),
        largest_interior_component_pixels=largest,
        second_interior_component_pixels=second,
        ambiguity_ratio=ambiguity,
        seed_pixel_count=seed_count,
        valid_seed_pixel_count=valid_seed_count,
        seed_valid_fraction=(valid_seed_count / seed_count if seed_count else 0.0),
        selected_seed_pixel_count=selected_seed_count,
        selected_seed_fraction=(
            selected_seed_count / valid_seed_count if valid_seed_count else 0.0
        ),
        mask_pixel_count=mask_count,
        mask_fraction=mask_fraction,
        minimum_mask_depth_m=float(np.min(mask_depth)),
        median_mask_depth_m=float(np.median(mask_depth)),
        maximum_mask_depth_m=float(np.max(mask_depth)),
        left_intensity_mean=float(np.mean(left, dtype=np.float64)),
        left_intensity_std=float(np.std(left, dtype=np.float64)),
    )
    return BootstrapForegroundResult(
        mask=mask,
        seed_mask=seed_mask,
        diagnostics=diagnostics,
        config=config,
        seed=seed,
        algorithm=BOOTSTRAP_FOREGROUND_ALGORITHM,
        policy_sha256=bootstrap_policy_sha256(config, seed),
        left_image_content_sha256=array_content_sha256(left),
        depth_content_sha256=array_content_sha256(original_depth),
        valid_mask_content_sha256=array_content_sha256(supplied_valid),
    )
