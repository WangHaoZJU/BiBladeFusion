"""Idempotent terminal-asset service used by fine next-view completion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from biblade_fusion.core.settings import (
    FineFinalizationConfig,
    MultiViewFusionConfig,
    SurfaceQualityConfig,
    TSDFConfig,
)
from biblade_fusion.storage.fine_reconstruction import (
    read_final_fine_reconstruction,
    replay_final_fine_reconstruction,
    write_final_fine_reconstruction,
    write_unaccepted_legacy_fine_reconstruction,
)
from biblade_fusion.storage.science_authority import ScienceAcceptanceAuthority
from biblade_fusion.storage.surface_coverage import StoredSurfaceCoverageGeneration
from biblade_fusion.workflows.fine_finalization import build_final_fine_reconstruction


@dataclass(frozen=True, slots=True)
class FinalFineCompletionEvidence:
    root: Path
    artifact_id: str
    metadata_sha256: str

    def __post_init__(self) -> None:
        root = Path(self.root).resolve()
        if not root.is_dir():
            raise ValueError("Final fine completion evidence root does not exist")
        for value in (self.artifact_id, self.metadata_sha256):
            if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
                raise ValueError("Final fine completion evidence requires SHA-256 identities")
        object.__setattr__(self, "root", root)


def finalize_fine_science(
    state: StoredSurfaceCoverageGeneration,
    *,
    fusion_config: MultiViewFusionConfig,
    tsdf_config: TSDFConfig,
    surface_quality_config: SurfaceQualityConfig,
    finalization_config: FineFinalizationConfig,
    science_authority: ScienceAcceptanceAuthority,
    output_dir: str | Path | None = None,
) -> FinalFineCompletionEvidence:
    """Create or replay-verify the one terminal asset for a coverage generation."""

    if science_authority is None:
        raise ValueError(
            "Production fine completion requires a science acceptance authority"
        )

    output = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (state.root.parent / "final_reconstruction").resolve()
    )
    science_authority.assert_acceptance_asset_current()
    if output.exists():
        stored = replay_final_fine_reconstruction(
            output,
            expected_science_authority=science_authority,
        )
    else:
        result = build_final_fine_reconstruction(
            state.root,
            fusion_config=fusion_config,
            tsdf_config=tsdf_config,
            surface_quality_config=surface_quality_config,
            finalization_config=finalization_config,
        )
        write_final_fine_reconstruction(
            output,
            result,
            fusion_config=fusion_config,
            tsdf_config=tsdf_config,
            surface_quality_config=surface_quality_config,
            finalization_config=finalization_config,
            science_authority=science_authority,
        )
        stored = replay_final_fine_reconstruction(
            output,
            expected_science_authority=science_authority,
        )
    if (
        stored.result.coverage.root != state.root
        or stored.result.coverage.generation_id != state.generation_id
        or stored.result.coverage.metadata_sha256 != state.metadata_sha256
    ):
        raise ValueError(
            "Existing terminal reconstruction is bound to a different fine generation"
        )
    # A second strict read makes the returned evidence independent of a stale
    # in-memory object after replay.
    verified = read_final_fine_reconstruction(output)
    if verified.science_authority != science_authority:
        raise ValueError("Terminal reconstruction science authority changed")
    return FinalFineCompletionEvidence(
        verified.root,
        verified.artifact_id,
        verified.metadata_sha256,
    )


def finalize_unaccepted_fine_science(
    state: StoredSurfaceCoverageGeneration,
    *,
    fusion_config: MultiViewFusionConfig,
    tsdf_config: TSDFConfig,
    surface_quality_config: SurfaceQualityConfig,
    finalization_config: FineFinalizationConfig,
    output_dir: str | Path | None = None,
) -> FinalFineCompletionEvidence:
    """Create a replayable experiment result without a science acceptance claim."""

    output = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (state.root.parent / "final_reconstruction").resolve()
    )
    if output.exists():
        stored = replay_final_fine_reconstruction(output)
    else:
        result = build_final_fine_reconstruction(
            state.root,
            fusion_config=fusion_config,
            tsdf_config=tsdf_config,
            surface_quality_config=surface_quality_config,
            finalization_config=finalization_config,
        )
        write_unaccepted_legacy_fine_reconstruction(
            output,
            result,
            fusion_config=fusion_config,
            tsdf_config=tsdf_config,
            surface_quality_config=surface_quality_config,
            finalization_config=finalization_config,
        )
        stored = replay_final_fine_reconstruction(output)
    if (
        stored.result.coverage.root != state.root
        or stored.result.coverage.generation_id != state.generation_id
        or stored.result.coverage.metadata_sha256 != state.metadata_sha256
        or stored.science_authority is not None
    ):
        raise ValueError("Unaccepted reconstruction changed its source or claimed authority")
    verified = read_final_fine_reconstruction(output)
    if verified.science_authority is not None:
        raise ValueError("Unaccepted reconstruction unexpectedly claims science authority")
    return FinalFineCompletionEvidence(
        verified.root,
        verified.artifact_id,
        verified.metadata_sha256,
    )
