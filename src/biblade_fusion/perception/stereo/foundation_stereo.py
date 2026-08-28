"""Lazy adapter for the pinned official NVIDIA FoundationStereo implementation."""

from __future__ import annotations

import hashlib
import sys
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from biblade_fusion.core.settings import FoundationStereoConfig
from biblade_fusion.diagnostics.types import CheckLevel, CheckResult
from biblade_fusion.perception.stereo.base import StereoResult

_REQUIRED_MODULES = (
    ("torch", "PyTorch"),
    ("torchvision", "torchvision"),
    ("omegaconf", "OmegaConf"),
    ("timm", "timm"),
    ("huggingface_hub", "Hugging Face Hub"),
    ("einops", "einops"),
    ("scipy", "SciPy"),
    ("cv2", "OpenCV"),
)
_OPTIONAL_ACCELERATION_MODULES = (
    ("xformers", "xFormers"),
    ("flash_attn", "FlashAttention"),
)


def _path_check(name: str, path: Path, expected_child: str | None = None) -> CheckResult:
    complete = (path / expected_child).is_file() if expected_child else path.is_file()
    if complete:
        return CheckResult(name, CheckLevel.PASS, str(path))
    expectation = f" containing {expected_child}" if expected_child else ""
    return CheckResult(name, CheckLevel.FAIL, f"missing {path}{expectation}")


def _module_check(module_name: str, display_name: str, required: bool) -> CheckResult:
    available = find_spec(module_name) is not None
    if available:
        return CheckResult(f"stereo_dependency:{module_name}", CheckLevel.PASS, display_name)
    level = CheckLevel.FAIL if required else CheckLevel.WARN
    qualifier = "required" if required else "optional acceleration"
    return CheckResult(
        f"stereo_dependency:{module_name}",
        level,
        f"{display_name} is not installed ({qualifier})",
    )


def _cuda_check(config: FoundationStereoConfig) -> CheckResult:
    if config.device == "cpu":
        return CheckResult(
            "foundation_stereo_device",
            CheckLevel.WARN,
            "CPU selected; official demo is CUDA-oriented and inference will be slow",
        )
    if find_spec("torch") is None:
        return CheckResult(
            "foundation_stereo_device",
            CheckLevel.FAIL,
            "CUDA requested but PyTorch is not installed",
        )
    try:
        torch = import_module("torch")
        available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count()) if available else 0
    except Exception as exc:
        return CheckResult(
            "foundation_stereo_device",
            CheckLevel.FAIL,
            f"PyTorch CUDA probe failed: {exc}",
        )
    if not available:
        return CheckResult(
            "foundation_stereo_device",
            CheckLevel.FAIL,
            "CUDA requested but torch.cuda.is_available() is false",
        )
    return CheckResult(
        "foundation_stereo_device",
        CheckLevel.PASS,
        f"CUDA available with {device_count} device(s)",
    )


def run_foundation_stereo_doctor(config: FoundationStereoConfig) -> list[CheckResult]:
    """Validate source, weights, Python dependencies, and requested compute device."""

    results = [
        _path_check(
            "foundation_stereo_repository",
            config.repository_path,
            "core/foundation_stereo.py",
        ),
        _path_check("foundation_stereo_checkpoint", config.checkpoint_path),
        _path_check("foundation_stereo_model_config", _model_config_path(config)),
    ]
    results.extend(
        _module_check(module_name, display_name, required=True)
        for module_name, display_name in _REQUIRED_MODULES
    )
    results.extend(
        _module_check(module_name, display_name, required=False)
        for module_name, display_name in _OPTIONAL_ACCELERATION_MODULES
    )
    results.append(_cuda_check(config))
    return results


def _model_config_path(config: FoundationStereoConfig) -> Path:
    return config.model_config_path or config.checkpoint_path.parent / "cfg.yaml"


class FoundationStereoError(RuntimeError):
    """FoundationStereo could not be loaded or returned an invalid result."""


class FoundationStereoRuntime(Protocol):
    """Small injectable boundary around the heavyweight upstream runtime."""

    @property
    def metadata(self) -> dict[str, Any]: ...

    def infer(
        self,
        left_rgb: NDArray[np.uint8],
        right_rgb: NDArray[np.uint8],
        *,
        valid_iterations: int,
        hierarchical: bool,
    ) -> NDArray[np.float32]: ...


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _plain_scalar(value: Any) -> bool | int | float | str | None:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    item = getattr(value, "item", None)
    if callable(item):
        return _plain_scalar(item())
    return str(value)


def _ensure_upstream_import_path(repository_path: Path) -> None:
    repository = repository_path.resolve()
    existing = sys.modules.get("core")
    if existing is not None:
        module_file = getattr(existing, "__file__", None)
        if module_file is None or not Path(module_file).resolve().is_relative_to(repository):
            raise FoundationStereoError(
                "top-level Python module 'core' is already imported from outside "
                f"FoundationStereo: {module_file}"
            )
    repository_text = str(repository)
    if repository_text not in sys.path:
        sys.path.insert(0, repository_text)


def _install_minimal_upstream_utils() -> bool:
    """Provide the two model helpers without importing upstream demo dependencies."""

    existing = sys.modules.get("Utils")
    if existing is not None:
        required = ("freeze_model", "get_resize_keep_aspect_ratio")
        if not all(hasattr(existing, name) for name in required):
            raise FoundationStereoError("top-level Python module 'Utils' is incompatible")
        return False

    module = ModuleType("Utils")

    def freeze_model(model: Any) -> Any:
        model = model.eval()
        for parameter in model.parameters():
            parameter.requires_grad = False
        for buffer in model.buffers():
            buffer.requires_grad = False
        return model

    def get_resize_keep_aspect_ratio(
        height: int,
        width: int,
        divider: int = 16,
        max_H: int = 1232,
        max_W: int = 1232,
    ) -> tuple[int, int]:
        if max_H % divider != 0 or max_W % divider != 0:
            raise ValueError("maximum DINO dimensions must be divisible by divider")

        def round_by_divider(value: float) -> int:
            return int(np.ceil(value / divider) * divider)

        resized_height = round_by_divider(height)
        resized_width = round_by_divider(width)
        if resized_height > max_H or resized_width > max_W:
            if resized_height > resized_width:
                resized_width = round_by_divider(resized_width * max_H / resized_height)
                resized_height = max_H
            else:
                resized_height = round_by_divider(resized_height * max_W / resized_width)
                resized_width = max_W
        return resized_height, resized_width

    module.freeze_model = freeze_model
    module.get_resize_keep_aspect_ratio = get_resize_keep_aspect_ratio
    sys.modules["Utils"] = module
    return True


class _OfficialFoundationStereoRuntime:
    """Load the exact official source and checkpoint only when inference is requested."""

    def __init__(self, config: FoundationStereoConfig) -> None:
        repository = config.repository_path.resolve()
        checkpoint = config.checkpoint_path.resolve()
        model_config = _model_config_path(config).resolve()
        for path, description in (
            (repository / "core/foundation_stereo.py", "official source"),
            (checkpoint, "checkpoint"),
            (model_config, "model configuration"),
        ):
            if not path.is_file():
                raise FoundationStereoError(f"FoundationStereo {description} is missing: {path}")

        try:
            torch = import_module("torch")
            omega_conf = import_module("omegaconf").OmegaConf
            _ensure_upstream_import_path(repository)
            installed_utils_shim = _install_minimal_upstream_utils()
            try:
                foundation_module = import_module("core.foundation_stereo")
                extractor_module = import_module("core.extractor")
                input_padder = import_module("core.utils.utils").InputPadder
            finally:
                if installed_utils_shim:
                    sys.modules.pop("Utils", None)
        except (ImportError, AttributeError) as exc:
            raise FoundationStereoError(
                "FoundationStereo runtime dependencies are incomplete; run "
                "`uv sync --extra foundation-stereo`"
            ) from exc

        if config.device == "cuda" and not torch.cuda.is_available():
            raise FoundationStereoError("CUDA was requested but torch.cuda.is_available() is false")

        arguments = omega_conf.load(str(model_config))
        if "vit_size" not in arguments:
            arguments["vit_size"] = "vitl"
        arguments["valid_iters"] = config.valid_iterations
        arguments["hiera"] = int(config.hierarchical)
        arguments["remove_invisible"] = int(config.remove_invisible)
        if config.device == "cpu":
            arguments["mixed_precision"] = False

        # Upstream constructs EdgeNeXt with pretrained=True, which otherwise triggers
        # an implicit network download. The full FoundationStereo state dict loaded
        # immediately below replaces those initialization weights.
        original_create_model = extractor_module.timm.create_model
        original_hub_load = torch.hub.load

        def create_model_without_download(*args: Any, **kwargs: Any) -> Any:
            kwargs["pretrained"] = False
            return original_create_model(*args, **kwargs)

        def load_bundled_dinov2(
            repository_name: str,
            model_name: str,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            if repository_name != "facebookresearch/dinov2":
                raise FoundationStereoError(
                    f"unexpected network-backed torch.hub repository: {repository_name}"
                )
            kwargs["source"] = "local"
            kwargs["pretrained"] = False
            return original_hub_load(
                str(repository / "dinov2"),
                model_name,
                *args,
                **kwargs,
            )

        extractor_module.timm.create_model = create_model_without_download
        torch.hub.load = load_bundled_dinov2
        try:
            model = foundation_module.FoundationStereo(arguments)
        finally:
            extractor_module.timm.create_model = original_create_model
            torch.hub.load = original_hub_load

        try:
            checkpoint_payload = torch.load(
                str(checkpoint),
                map_location=config.device,
                weights_only=True,
            )
            model.load_state_dict(checkpoint_payload["model"], strict=True)
        except Exception as exc:
            raise FoundationStereoError(
                f"FoundationStereo checkpoint could not be loaded safely: {checkpoint}"
            ) from exc

        self._torch = torch
        self._input_padder = input_padder
        self._model = model.to(config.device).eval()
        self._device = config.device
        self._metadata = {
            "runtime": "official_nvidia_foundation_stereo",
            "repository_path": str(repository),
            "checkpoint_path": str(checkpoint),
            "model_config_path": str(model_config),
            "source_sha256": _sha256(repository / "core/foundation_stereo.py"),
            "checkpoint_sha256": _sha256(checkpoint),
            "model_config_sha256": _sha256(model_config),
            "checkpoint_global_step": _plain_scalar(checkpoint_payload.get("global_step")),
            "checkpoint_epoch": _plain_scalar(checkpoint_payload.get("epoch")),
        }

    @property
    def metadata(self) -> dict[str, Any]:
        return dict(self._metadata)

    def infer(
        self,
        left_rgb: NDArray[np.uint8],
        right_rgb: NDArray[np.uint8],
        *,
        valid_iterations: int,
        hierarchical: bool,
    ) -> NDArray[np.float32]:
        torch = self._torch
        left = torch.as_tensor(left_rgb, device=self._device).float()[None].permute(0, 3, 1, 2)
        right = torch.as_tensor(right_rgb, device=self._device).float()[None].permute(0, 3, 1, 2)
        padder = self._input_padder(left.shape, divis_by=32, force_square=False)
        left, right = padder.pad(left, right)
        with torch.inference_mode():
            if hierarchical:
                disparity = self._model.run_hierachical(
                    left,
                    right,
                    iters=valid_iterations,
                    test_mode=True,
                    small_ratio=0.5,
                )
            else:
                disparity = self._model.forward(
                    left,
                    right,
                    iters=valid_iterations,
                    test_mode=True,
                )
        disparity = padder.unpad(disparity.float())
        array = disparity.detach().cpu().numpy()
        if array.shape != (1, 1, left_rgb.shape[0], left_rgb.shape[1]):
            raise FoundationStereoError(f"unexpected disparity tensor shape: {array.shape}")
        return np.asarray(array[0, 0], dtype=np.float32)


class FoundationStereoBackend:
    """Infer full-resolution rectified disparity without moving any hardware."""

    def __init__(
        self,
        config: FoundationStereoConfig,
        runtime: FoundationStereoRuntime | None = None,
    ) -> None:
        self._config = config
        self._runtime = runtime

    def _get_runtime(self) -> FoundationStereoRuntime:
        if self._runtime is None:
            self._runtime = _OfficialFoundationStereoRuntime(self._config)
        return self._runtime

    def infer(
        self,
        left_rectified: NDArray[np.uint8],
        right_rectified: NDArray[np.uint8],
    ) -> StereoResult:
        left = np.asarray(left_rectified)
        right = np.asarray(right_rectified)
        if left.dtype != np.uint8 or right.dtype != np.uint8:
            raise ValueError("FoundationStereo inputs must be uint8 images")
        if left.ndim != 2 or right.ndim != 2:
            raise ValueError("FoundationStereo inputs must be single-channel rectified images")
        if left.shape != right.shape:
            raise ValueError("FoundationStereo left and right image shapes must match")

        height, width = left.shape
        scaled_width = max(1, round(width * self._config.scale))
        scaled_height = max(1, round(height * self._config.scale))
        if (scaled_height, scaled_width) != left.shape:
            cv2 = import_module("cv2")
            left = cv2.resize(left, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
            right = cv2.resize(right, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
        left_rgb = np.ascontiguousarray(np.repeat(left[:, :, None], 3, axis=2))
        right_rgb = np.ascontiguousarray(np.repeat(right[:, :, None], 3, axis=2))

        runtime = self._get_runtime()
        horizontal_scale = scaled_width / width
        disparity = self._infer_full_resolution_disparity(
            runtime,
            left_rgb,
            right_rgb,
            inference_shape=(scaled_height, scaled_width),
            output_shape=(height, width),
            horizontal_scale=horizontal_scale,
        )

        valid = np.isfinite(disparity) & (disparity > 0.0)
        if self._config.remove_invisible:
            horizontal_pixels = np.arange(width, dtype=np.float32)[None, :]
            valid &= horizontal_pixels - disparity >= 0.0

        confidence = None
        threshold = self._config.left_right_consistency_threshold_px
        if threshold is not None:
            # Flip-and-swap preserves the positive-disparity convention while
            # estimating the correspondence field from the physical right view.
            flipped_right = np.ascontiguousarray(right_rgb[:, ::-1])
            flipped_left = np.ascontiguousarray(left_rgb[:, ::-1])
            flipped_right_disparity = self._infer_full_resolution_disparity(
                runtime,
                flipped_right,
                flipped_left,
                inference_shape=(scaled_height, scaled_width),
                output_shape=(height, width),
                horizontal_scale=horizontal_scale,
            )
            right_disparity = np.ascontiguousarray(
                flipped_right_disparity[:, ::-1]
            )
            row_indices = np.arange(height, dtype=np.int64)[:, None]
            horizontal_pixels = np.arange(width, dtype=np.float32)[None, :]
            finite_left = np.isfinite(disparity)
            correspondence_x = np.rint(
                horizontal_pixels - np.where(finite_left, disparity, 0.0)
            ).astype(np.int64)
            correspondence_in_bounds = (
                finite_left
                & (correspondence_x >= 0)
                & (correspondence_x < width)
            )
            clipped_x = np.clip(correspondence_x, 0, width - 1)
            corresponding_right = right_disparity[row_indices, clipped_x]
            error_px = np.abs(disparity - corresponding_right)
            comparable = (
                correspondence_in_bounds
                & np.isfinite(corresponding_right)
                & (corresponding_right > 0.0)
            )
            consistent = comparable & (error_px <= threshold)
            valid &= consistent
            confidence = np.zeros(disparity.shape, dtype=np.float32)
            confidence[comparable] = np.exp(
                -error_px[comparable] / float(threshold)
            ).astype(np.float32)

        metadata = {
            "backend": "foundation_stereo",
            "requested_scale": self._config.scale,
            "effective_horizontal_scale": horizontal_scale,
            "inference_shape": [scaled_height, scaled_width],
            "output_disparity_units": "full_resolution_left_pixels",
            "valid_iterations": self._config.valid_iterations,
            "hierarchical": self._config.hierarchical,
            "remove_invisible": self._config.remove_invisible,
            "left_right_consistency_applied": threshold is not None,
            "left_right_consistency_threshold_px": threshold,
            "confidence_semantic": (
                "exp_negative_left_right_disparity_error_not_calibrated_probability"
                if threshold is not None
                else None
            ),
            **runtime.metadata,
        }
        return StereoResult(disparity, valid, confidence, metadata)

    def _infer_full_resolution_disparity(
        self,
        runtime: FoundationStereoRuntime,
        left_rgb: NDArray[np.uint8],
        right_rgb: NDArray[np.uint8],
        *,
        inference_shape: tuple[int, int],
        output_shape: tuple[int, int],
        horizontal_scale: float,
    ) -> NDArray[np.float32]:
        scaled_disparity = np.asarray(
            runtime.infer(
                left_rgb,
                right_rgb,
                valid_iterations=self._config.valid_iterations,
                hierarchical=self._config.hierarchical,
            ),
            dtype=np.float32,
        )
        if scaled_disparity.shape != inference_shape:
            raise FoundationStereoError(
                "runtime disparity shape does not match scaled input: "
                f"{scaled_disparity.shape} != {inference_shape}"
            )
        if scaled_disparity.shape == output_shape:
            return np.asarray(scaled_disparity, dtype=np.float32).copy()
        cv2 = import_module("cv2")
        width, height = output_shape[1], output_shape[0]
        return np.asarray(
            cv2.resize(
                scaled_disparity,
                (width, height),
                interpolation=cv2.INTER_LINEAR,
            )
            / horizontal_scale,
            dtype=np.float32,
        )
