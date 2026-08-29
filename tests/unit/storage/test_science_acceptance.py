from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import biblade_fusion.storage as storage_api
from biblade_fusion.core.settings import load_settings
from biblade_fusion.storage.science_acceptance import (
    ScienceTestEnvelope,
    canonicalize_science_evidence,
    load_science_acceptance_declaration,
    read_science_acceptance,
    science_runtime_contract_for_settings,
    science_runtime_contract_sha256,
    write_science_acceptance,
)

_CHECKLIST = {
    "traceable_depth_reference_verified": True,
    "distance_and_incidence_envelope_verified": True,
    "front_back_and_both_fins_annotated": True,
    "bootstrap_and_reference_masks_reviewed": True,
    "final_mesh_holes_and_thickness_reviewed": True,
    "raw_acceptance_assets_archived": True,
    "independent_result_review_completed": True,
}


def _metrics(*, passing: bool = True) -> tuple[dict[str, float], dict[str, float]]:
    limits = {
        "depth_rmse_m": 0.003,
        "depth_p95_m": 0.006,
        "depth_absolute_bias_m": 0.002,
        "bootstrap_mask_precision": 0.95,
        "bootstrap_mask_recall": 0.90,
        "fine_mask_precision": 0.97,
        "fine_mask_recall": 0.93,
        "final_surface_rmse_m": 0.003,
        "final_surface_p95_m": 0.006,
        "final_hole_fraction": 0.02,
        "thickness_absolute_error_m": 0.002,
    }
    measured = {
        "depth_rmse_m": 0.002,
        "depth_p95_m": 0.005,
        "depth_absolute_bias_m": 0.001,
        "bootstrap_mask_precision": 0.97,
        "bootstrap_mask_recall": 0.94,
        "fine_mask_precision": 0.98,
        "fine_mask_recall": 0.95,
        "final_surface_rmse_m": 0.002,
        "final_surface_p95_m": 0.005,
        "final_hole_fraction": 0.01,
        "thickness_absolute_error_m": 0.001,
    }
    if not passing:
        measured["fine_mask_recall"] = 0.5
    return limits, measured


def _runtime_contract() -> dict[str, Any]:
    digest = "a" * 64
    distributions = [
        {"name": name, "version": "1.0"}
        for name in (
            "einops",
            "numpy",
            "omegaconf",
            "scipy",
            "timm",
            "torch",
            "torchvision",
        )
    ]
    return {
        "schema": "biblade_fusion.geometry_science_runtime_contract.v3",
        "sources": [{"label": "foundation_stereo", "sha256": digest, "size_bytes": 10}],
        "source_trees": [
            {
                "label": "biblade_fusion_python",
                "files": [
                    {
                        "relative_path": "science.py",
                        "sha256": digest,
                        "size_bytes": 10,
                    }
                ],
            }
        ],
        "project_runtime_files": [
            {"relative_path": "pyproject.toml", "sha256": digest, "size_bytes": 10},
            {"relative_path": "uv.lock", "sha256": digest, "size_bytes": 10},
        ],
        "policies": {"science": {"enabled": True}},
        "runtime_environment": {
            "python": {
                "implementation": "CPython",
                "version": "3.12.0",
                "compiler": "GCC",
                "cache_tag": "cpython-312",
            },
            "platform": {
                "os_name": "posix",
                "system": "Linux",
                "release": "6.0",
                "kernel_version": "test-kernel",
                "machine": "x86_64",
                "architecture": "64bit",
                "libc": {"name": "glibc", "version": "2.39"},
            },
            "python_distributions": distributions,
            "torch_runtime": {
                "importable": True,
                "torch_version": "1.0",
                "cuda_available": False,
                "cuda_version": None,
                "cudnn_version": None,
                "devices": [],
                "probe_error_type": None,
            },
            "nvidia_driver": {
                "readable": False,
                "source": None,
                "version": None,
                "content_sha256": None,
            },
            "visibility": {
                "cuda_visible_devices": None,
                "nvidia_visible_devices": None,
            },
        },
    }


def _canonical_write(path: Path, payload: dict[str, Any]) -> None:
    path.write_bytes(
        json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _values(tmp_path: Path) -> dict[str, Any]:
    limits, measured = _metrics()
    runtime = _runtime_contract()
    runtime_sha = science_runtime_contract_sha256(runtime)
    raw = tmp_path / "raw.json"
    _canonical_write(
        raw,
        {
            "schema_version": 1,
            "report_type": "biblade_fusion.raw_science_acceptance_asset_manifest",
            "generator": {"name": "acceptance-asset-indexer", "version": "1.0"},
            "created_at_utc": "2026-08-29T00:00:00+00:00",
            "assets": [
                {
                    "asset_id": "annotation-001",
                    "role": "annotation",
                    "archive_path": "annotations/blade-001.json",
                    "sha256": "1" * 64,
                    "size_bytes": 101,
                },
                {
                    "asset_id": "depth-001",
                    "role": "depth_reference",
                    "archive_path": "depth/depth-001.npz",
                    "sha256": "2" * 64,
                    "size_bytes": 102,
                },
                {
                    "asset_id": "specimen-001",
                    "role": "specimen",
                    "archive_path": "specimens/blade-001.json",
                    "sha256": "3" * 64,
                    "size_bytes": 103,
                },
            ],
        },
    )
    evaluation = tmp_path / "evaluation.json"
    _canonical_write(
        evaluation,
        {
            "schema_version": 1,
            "report_type": "biblade_fusion.geometry_science_evaluation_report",
            "generator": {"name": "bbf-science-evaluator", "version": "1.0"},
            "created_at_utc": "2026-08-29T00:01:00+00:00",
            "science_runtime_contract_sha256": runtime_sha,
            "raw_acceptance_asset_manifest_sha256": _sha256(raw),
            "measurements": measured,
            "test_envelope": {
                "minimum_distance_m": 0.25,
                "maximum_distance_m": 0.75,
                "minimum_incidence_deg": 0.0,
                "maximum_incidence_deg": 75.0,
            },
            "sample_counts": {
                "depth_reference": 1000,
                "annotated_frames": 40,
                "reconstructed_specimens": 5,
            },
        },
    )
    review = tmp_path / "review.json"
    _canonical_write(
        review,
        {
            "schema_version": 1,
            "report_type": "biblade_fusion.geometry_science_independent_review_report",
            "reviewed_at_utc": "2026-08-29T00:02:00+00:00",
            "reviewer_id": "reviewer-2",
            "operator_id": "operator-1",
            "decision": "accepted",
            "notes": "Independent review reproduced the declared measurements.",
            "science_runtime_contract_sha256": runtime_sha,
            "geometry_evaluation_report_sha256": _sha256(evaluation),
            "raw_acceptance_asset_manifest_sha256": _sha256(raw),
            "checklist": _CHECKLIST,
        },
    )
    return {
        "workcell_id": "cell-1",
        "operator_id": "operator-1",
        "accepted_at_utc": datetime(2026, 8, 29, tzinfo=UTC),
        "science_runtime_contract": runtime,
        "geometry_evaluation_report_path": evaluation,
        "raw_acceptance_asset_manifest_path": raw,
        "independent_review_report_path": review,
        "limits": limits,
        "measurements": measured,
        "minimum_test_distance_m": 0.25,
        "maximum_test_distance_m": 0.75,
        "minimum_test_incidence_deg": 0.0,
        "maximum_test_incidence_deg": 75.0,
        "depth_reference_sample_count": 1000,
        "annotated_frame_count": 40,
        "reconstructed_specimen_count": 5,
        "checklist": _CHECKLIST,
    }


def test_science_acceptance_seals_evidence_and_runtime(tmp_path: Path) -> None:
    values = _values(tmp_path)
    stored = write_science_acceptance(tmp_path / "acceptance", **values)
    for key in (
        "geometry_evaluation_report_path",
        "raw_acceptance_asset_manifest_path",
        "independent_review_report_path",
    ):
        Path(values[key]).unlink()
    reread = read_science_acceptance(stored.path)
    runtime_sha = science_runtime_contract_sha256(values["science_runtime_contract"])

    assert reread.acceptance_id == stored.acceptance_id
    assert reread.science_runtime_contract == values["science_runtime_contract"]
    assert reread.test_envelope == ScienceTestEnvelope(0.25, 0.75, 0.0, 75.0)
    assert (stored.path / "evidence/geometry_evaluation_report.json").is_file()
    assert (stored.path / "evidence/raw_acceptance_asset_manifest.json").is_file()
    assert (stored.path / "evidence/independent_review_report.json").is_file()
    metadata = json.loads((stored.path / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["motion_authorized"] is False
    reread.assert_matches(
        acceptance_id=stored.acceptance_id,
        runtime_contract_sha256=runtime_sha,
        required_test_envelope=ScienceTestEnvelope(0.25, 0.75, 0.0, 70.0),
    )


def test_science_evidence_canonicalizer_validates_all_kinds_without_overwrite(
    tmp_path: Path,
) -> None:
    values = _values(tmp_path)
    sources = {
        "raw-manifest": Path(values["raw_acceptance_asset_manifest_path"]),
        "evaluation": Path(values["geometry_evaluation_report_path"]),
        "review": Path(values["independent_review_report_path"]),
    }
    for kind, source in sources.items():
        pretty = tmp_path / f"{kind}.pretty.json"
        pretty.write_text(
            json.dumps(json.loads(source.read_text(encoding="utf-8")), indent=2),
            encoding="utf-8",
        )
        output = tmp_path / "canonical" / f"{kind}.json"
        stored = canonicalize_science_evidence(
            kind=kind,
            input_path=pretty,
            output_path=output,
        )
        payload = json.loads(output.read_text(encoding="utf-8"))
        expected = (
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        )
        assert output.read_bytes() == expected
        assert stored.sha256 == hashlib.sha256(expected).hexdigest()
        assert stored.size_bytes == len(expected)
        with pytest.raises(FileExistsError):
            canonicalize_science_evidence(
                kind=kind,
                input_path=pretty,
                output_path=output,
            )


@pytest.mark.parametrize("invalid_constant", ["NaN", "Infinity", "-Infinity"])
def test_science_evidence_canonicalizer_rejects_nonfinite_and_duplicate_json(
    tmp_path: Path,
    invalid_constant: str,
) -> None:
    values = _values(tmp_path)
    source = Path(values["geometry_evaluation_report_path"])
    content = source.read_text(encoding="utf-8")
    invalid = tmp_path / f"invalid-{invalid_constant}.json"
    invalid.write_text(content.replace("0.002", invalid_constant, 1), encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON"):
        canonicalize_science_evidence(
            kind="evaluation",
            input_path=invalid,
            output_path=tmp_path / f"invalid-{invalid_constant}.canonical.json",
        )

    duplicate = tmp_path / f"duplicate-{invalid_constant}.json"
    duplicate.write_text(
        content.replace('{"created_at_utc"', '{"schema_version":1,"created_at_utc"'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        canonicalize_science_evidence(
            kind="evaluation",
            input_path=duplicate,
            output_path=tmp_path / f"duplicate-{invalid_constant}.canonical.json",
        )


def test_science_acceptance_rejects_missing_and_mismatched_evidence(tmp_path: Path) -> None:
    values = _values(tmp_path)
    Path(values["geometry_evaluation_report_path"]).unlink()
    with pytest.raises(FileNotFoundError):
        write_science_acceptance(tmp_path / "missing", **values)

    values = _values(tmp_path)
    evaluation = Path(values["geometry_evaluation_report_path"])
    payload = json.loads(evaluation.read_text(encoding="utf-8"))
    payload["measurements"]["depth_rmse_m"] = 0.0025
    _canonical_write(evaluation, payload)
    with pytest.raises(ValueError, match="measurements differs"):
        write_science_acceptance(tmp_path / "mismatch", **values)

    values = _values(tmp_path)
    evaluation = Path(values["geometry_evaluation_report_path"])
    payload = json.loads(evaluation.read_text(encoding="utf-8"))
    payload["sample_counts"]["annotated_frames"] = 41
    _canonical_write(evaluation, payload)
    with pytest.raises(ValueError, match="sample_counts differs"):
        write_science_acceptance(tmp_path / "count-mismatch", **values)


def test_science_acceptance_rejects_forged_duplicate_raw_asset(tmp_path: Path) -> None:
    values = _values(tmp_path)
    raw = Path(values["raw_acceptance_asset_manifest_path"])
    payload = json.loads(raw.read_text(encoding="utf-8"))
    payload["assets"][1]["sha256"] = payload["assets"][0]["sha256"]
    payload["assets"][1]["size_bytes"] = payload["assets"][0]["size_bytes"]
    _canonical_write(raw, payload)
    with pytest.raises(ValueError, match="must be unique"):
        write_science_acceptance(tmp_path / "duplicate-asset", **values)


def test_science_acceptance_rejects_forged_independent_review(tmp_path: Path) -> None:
    values = _values(tmp_path)
    review = Path(values["independent_review_report_path"])
    payload = json.loads(review.read_text(encoding="utf-8"))
    payload["reviewer_id"] = payload["operator_id"]
    _canonical_write(review, payload)
    with pytest.raises(ValueError, match="reviewer must differ"):
        write_science_acceptance(tmp_path / "self-review", **values)

    values = _values(tmp_path)
    review = Path(values["independent_review_report_path"])
    payload = json.loads(review.read_text(encoding="utf-8"))
    payload["geometry_evaluation_report_sha256"] = "f" * 64
    _canonical_write(review, payload)
    with pytest.raises(ValueError, match="bindings are invalid"):
        write_science_acceptance(tmp_path / "forged-binding", **values)


def test_science_acceptance_rejects_noncanonical_duplicate_and_nan_json(
    tmp_path: Path,
) -> None:
    values = _values(tmp_path)
    raw = Path(values["raw_acceptance_asset_manifest_path"])
    payload = json.loads(raw.read_text(encoding="utf-8"))
    raw.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="not canonical"):
        write_science_acceptance(tmp_path / "pretty", **values)

    values = _values(tmp_path)
    evaluation = Path(values["geometry_evaluation_report_path"])
    content = evaluation.read_text(encoding="utf-8")
    evaluation.write_text(
        content.replace('{"created_at_utc"', '{"schema_version":1,"created_at_utc"'),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        write_science_acceptance(tmp_path / "duplicate-key", **values)

    values = _values(tmp_path)
    evaluation = Path(values["geometry_evaluation_report_path"])
    content = evaluation.read_text(encoding="utf-8")
    evaluation.write_text(content.replace("0.002", "NaN", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON"):
        write_science_acceptance(tmp_path / "nan", **values)


def test_science_acceptance_reader_rejects_evidence_tamper(tmp_path: Path) -> None:
    stored = write_science_acceptance(tmp_path / "acceptance", **_values(tmp_path))
    copied = stored.path / "evidence/geometry_evaluation_report.json"
    payload = json.loads(copied.read_text(encoding="utf-8"))
    payload["measurements"]["depth_rmse_m"] = 0.0025
    _canonical_write(copied, payload)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        read_science_acceptance(stored.path)


def test_science_acceptance_rejects_noncanonical_runtime_distributions(
    tmp_path: Path,
) -> None:
    values = _values(tmp_path)
    runtime = values["science_runtime_contract"]
    runtime["runtime_environment"]["python_distributions"][0]["name"] = "Einops"
    with pytest.raises(ValueError, match="not canonical"):
        write_science_acceptance(tmp_path / "bad-runtime", **values)

    values = _values(tmp_path)
    runtime = values["science_runtime_contract"]
    runtime["runtime_environment"]["python_distributions"].append(
        {"name": "torch", "version": "2.0"}
    )
    with pytest.raises(ValueError, match="uniquely sorted"):
        write_science_acceptance(tmp_path / "ambiguous-runtime", **values)

    values = _values(tmp_path)
    runtime = values["science_runtime_contract"]
    runtime["runtime_environment"]["torch_runtime"]["probe_error_type"] = "/tmp/leak"
    with pytest.raises(ValueError, match="absolute navigation path"):
        write_science_acceptance(tmp_path / "path-bearing-runtime", **values)


def test_declaration_loader_rejects_duplicate_and_nan(tmp_path: Path) -> None:
    template = Path("configs/science_acceptance.template.json").read_text(encoding="utf-8")
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(template.replace("{", '{"schema_version":2,', 1), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_science_acceptance_declaration(duplicate)

    non_finite = tmp_path / "nan.json"
    non_finite.write_text(template.replace("null", "NaN", 1), encoding="utf-8")
    with pytest.raises(ValueError, match="non-finite JSON"):
        load_science_acceptance_declaration(non_finite)


def test_science_runtime_contract_binds_sources_and_all_distributions(tmp_path: Path) -> None:
    repository = tmp_path / "FoundationStereo"
    source = repository / "core" / "foundation_stereo.py"
    source.parent.mkdir(parents=True)
    source.write_text("source-v1", encoding="utf-8")
    imported_utility = repository / "models" / "geometry.py"
    imported_utility.parent.mkdir(parents=True)
    imported_utility.write_text("utility-v1", encoding="utf-8")
    checkpoint = tmp_path / "model.pth"
    model_config = tmp_path / "cfg.yaml"
    calibration = tmp_path / "stereo.yaml"
    hand_eye = tmp_path / "hand_eye.yaml"
    for path, content in (
        (checkpoint, "weights"),
        (model_config, "model"),
        (calibration, "calibration"),
        (hand_eye, "hand-eye"),
    ):
        path.write_text(content, encoding="utf-8")
    settings = load_settings("configs/default.yaml")
    settings = settings.model_copy(
        update={
            "foundation_stereo": settings.foundation_stereo.model_copy(
                update={
                    "repository_path": repository,
                    "checkpoint_path": checkpoint,
                    "model_config_path": model_config,
                }
            ),
            "realsense": settings.realsense.model_copy(
                update={"stereo_calibration_path": calibration}
            ),
            "hand_eye": settings.hand_eye.model_copy(update={"calibration_path": hand_eye}),
        }
    )

    first = science_runtime_contract_for_settings(settings)
    imported_utility.write_text("utility-v2", encoding="utf-8")
    second = science_runtime_contract_for_settings(settings)
    payload = storage_api.science_runtime_contract_payload(settings)

    assert first != second
    assert all("path" not in item for item in payload["sources"])
    distributions = payload["runtime_environment"]["python_distributions"]
    identities = [(item["name"], item["version"]) for item in distributions]
    assert identities == sorted(identities)
    assert len({name for name, _version in identities}) == len(identities)
    assert len(distributions) > 6
    assert payload["runtime_environment"]["platform"]["kernel_version"]
    assert "libc" in payload["runtime_environment"]["platform"]
    assert "nvidia_driver" in payload["runtime_environment"]
    trees = {item["label"]: item["files"] for item in payload["source_trees"]}
    assert {item["relative_path"] for item in trees["foundation_stereo_python"]} == {
        "core/foundation_stereo.py",
        "models/geometry.py",
    }
    assert [item["relative_path"] for item in payload["project_runtime_files"]] == [
        "pyproject.toml",
        "uv.lock",
    ]
    assert storage_api.science_runtime_contract_for_settings(settings) == second
    assert storage_api.read_science_acceptance is read_science_acceptance
    assert storage_api.write_science_acceptance is write_science_acceptance
