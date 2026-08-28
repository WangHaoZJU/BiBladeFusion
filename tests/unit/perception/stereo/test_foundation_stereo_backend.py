from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest

from biblade_fusion.core.settings import FoundationStereoConfig
from biblade_fusion.perception.stereo import (
    FoundationStereoBackend,
    FoundationStereoError,
)


@dataclass
class FakeRuntime:
    disparity: np.ndarray
    calls: list[tuple[np.ndarray, np.ndarray, int, bool]] = field(default_factory=list)

    @property
    def metadata(self) -> dict[str, Any]:
        return {"runtime": "fake"}

    def infer(self, left_rgb, right_rgb, *, valid_iterations, hierarchical):
        self.calls.append((left_rgb, right_rgb, valid_iterations, hierarchical))
        return self.disparity


def test_backend_restores_full_resolution_disparity_units_after_scaling() -> None:
    runtime = FakeRuntime(np.ones((4, 6), dtype=np.float32))
    backend = FoundationStereoBackend(
        FoundationStereoConfig(
            scale=0.5,
            valid_iterations=17,
            hierarchical=True,
            remove_invisible=True,
        ),
        runtime,
    )
    left = np.arange(8 * 12, dtype=np.uint8).reshape(8, 12)

    result = backend.infer(left, left)

    np.testing.assert_allclose(result.disparity_px, 2.0)
    assert not result.valid_mask[:, :2].any()
    assert result.valid_mask[:, 2:].all()
    assert result.confidence is not None
    np.testing.assert_allclose(result.confidence[:, 2:], 1.0)
    assert len(runtime.calls) == 2
    assert result.metadata["left_right_consistency_applied"] is True
    assert result.metadata["output_disparity_units"] == "full_resolution_left_pixels"
    call_left, call_right, iterations, hierarchical = runtime.calls[0]
    assert call_left.shape == (4, 6, 3)
    np.testing.assert_array_equal(call_left[:, :, 0], call_left[:, :, 1])
    np.testing.assert_array_equal(call_left, call_right)
    assert iterations == 17
    assert hierarchical is True


def test_backend_rejects_invalid_inputs_and_runtime_shape() -> None:
    backend = FoundationStereoBackend(
        FoundationStereoConfig(device="cpu"),
        FakeRuntime(np.ones((3, 3), dtype=np.float32)),
    )

    with pytest.raises(ValueError, match="uint8"):
        backend.infer(np.zeros((4, 4), dtype=np.float32), np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="shapes"):
        backend.infer(np.zeros((4, 4), dtype=np.uint8), np.zeros((4, 5), dtype=np.uint8))
    with pytest.raises(FoundationStereoError, match="shape"):
        backend.infer(np.zeros((4, 4), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8))


def test_backend_masks_nonpositive_and_nonfinite_disparity() -> None:
    disparity = np.array([[1.0, 0.0], [np.nan, -1.0]], dtype=np.float32)
    backend = FoundationStereoBackend(
        FoundationStereoConfig(
            device="cpu",
            remove_invisible=False,
            left_right_consistency_threshold_px=None,
        ),
        FakeRuntime(disparity),
    )

    result = backend.infer(np.zeros((2, 2), dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8))

    np.testing.assert_array_equal(result.valid_mask, [[True, False], [False, False]])
    assert result.confidence is None


@dataclass
class SequentialRuntime:
    disparities: list[np.ndarray]

    @property
    def metadata(self) -> dict[str, Any]:
        return {"runtime": "sequential-fake"}

    def infer(self, left_rgb, right_rgb, *, valid_iterations, hierarchical):
        del left_rgb, right_rgb, valid_iterations, hierarchical
        return self.disparities.pop(0)


def test_backend_rejects_left_right_inconsistent_matches() -> None:
    left_disparity = np.full((2, 4), 1.0, dtype=np.float32)
    # The backend flips this result back into right-image coordinates.  Only
    # the upper row agrees with the left disparity after that operation.
    flipped_right_disparity = np.vstack(
        (
            np.full(4, 1.0, dtype=np.float32),
            np.full(4, 3.0, dtype=np.float32),
        )
    )
    runtime = SequentialRuntime([left_disparity, flipped_right_disparity])
    backend = FoundationStereoBackend(
        FoundationStereoConfig(
            device="cpu",
            remove_invisible=False,
            left_right_consistency_threshold_px=0.5,
        ),
        runtime,
    )

    result = backend.infer(
        np.zeros((2, 4), dtype=np.uint8),
        np.zeros((2, 4), dtype=np.uint8),
    )

    np.testing.assert_array_equal(
        result.valid_mask,
        [[False, True, True, True], [False, False, False, False]],
    )
    assert result.confidence is not None
    np.testing.assert_allclose(result.confidence[0, 1:], 1.0)
