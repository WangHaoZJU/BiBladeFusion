from pathlib import Path

from biblade_fusion.core.settings import FoundationStereoConfig
from biblade_fusion.diagnostics import CheckLevel
from biblade_fusion.perception.stereo.foundation_stereo import (
    run_foundation_stereo_doctor,
)


def test_foundation_stereo_doctor_reports_missing_source_and_weights(tmp_path: Path) -> None:
    config = FoundationStereoConfig(
        repository_path=tmp_path / "missing-repository",
        checkpoint_path=tmp_path / "missing-model.pth",
        device="cpu",
    )

    results = run_foundation_stereo_doctor(config)
    levels = {result.name: result.level for result in results}

    assert levels["foundation_stereo_repository"] is CheckLevel.FAIL
    assert levels["foundation_stereo_checkpoint"] is CheckLevel.FAIL
    assert levels["foundation_stereo_device"] is CheckLevel.WARN
