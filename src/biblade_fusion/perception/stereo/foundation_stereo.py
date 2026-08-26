"""FoundationStereo environment validation and integration boundary."""

from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from pathlib import Path

from biblade_fusion.core.settings import FoundationStereoConfig
from biblade_fusion.diagnostics import CheckLevel, CheckResult

_REQUIRED_MODULES = (
    ("torch", "PyTorch"),
    ("torchvision", "torchvision"),
    ("omegaconf", "OmegaConf"),
    ("timm", "timm"),
    ("cv2", "OpenCV"),
    ("imageio", "imageio"),
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
