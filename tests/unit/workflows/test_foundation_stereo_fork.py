from __future__ import annotations

import threading
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import biblade_fusion.workflows.foundation_stereo_cycle as cycle_module
from biblade_fusion.core.settings import load_settings
from biblade_fusion.workflows.foundation_stereo_cycle import (
    FoundationStereoCycleError,
    FoundationStereoOccupancyCycleEngine,
)


def _settings_pair():
    fine = load_settings("configs/default.yaml")
    fine = fine.model_copy(
        update={"blade_foreground": fine.blade_foreground.model_copy(update={"enabled": True})}
    )
    coarse = fine.model_copy(
        update={"blade_foreground": fine.blade_foreground.model_copy(update={"enabled": False})}
    )
    return coarse, fine


def _empty_engine(coarse_settings):
    engine = object.__new__(FoundationStereoOccupancyCycleEngine)
    engine._settings = coarse_settings
    engine._pending_lock = threading.Lock()
    engine._pending_key = None
    engine._pending_attempt_root = None
    engine._pending_sampler = None
    engine._pending_commit = None
    engine._sources = []
    engine._utc_clock = lambda: datetime(2026, 8, 29, 1, 0, tzinfo=UTC)
    engine._acquirer = object()
    engine._state_source = object()
    engine._backend = object()
    engine._hand_eye = object()
    engine._renderer = object()
    # The default fixture deliberately exercises the unbound/library-only path.
    # Production composition supplies both values together and the constructor
    # verifies the immutable authority before any inference is possible.
    engine._science_authority = None
    engine._science_authority_settings = None
    return engine


def test_fine_fork_refuses_pending_coarse_transaction(tmp_path: Path) -> None:
    coarse, fine = _settings_pair()
    engine = _empty_engine(coarse)
    engine._pending_key = ("view", 1)

    with pytest.raises(FoundationStereoCycleError, match="pending transaction"):
        engine.fork_for_fine_science(
            settings=fine,
            reference_coarse_model=tmp_path / "reference",
            output_root=tmp_path / "fine",
        )


def test_fine_fork_refuses_any_non_foreground_policy_change(tmp_path: Path) -> None:
    coarse, fine = _settings_pair()
    changed = fine.model_copy(
        update={
            "acquisition": fine.acquisition.model_copy(
                update={"max_bracket_ms": fine.acquisition.max_bracket_ms + 1.0}
            )
        }
    )
    engine = _empty_engine(coarse)

    with pytest.raises(FoundationStereoCycleError, match="non-foreground"):
        engine.fork_for_fine_science(
            settings=changed,
            reference_coarse_model=tmp_path / "reference",
            output_root=tmp_path / "fine",
        )


def test_fine_fork_copies_only_reverified_committed_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coarse, fine = _settings_pair()
    engine = _empty_engine(coarse)
    session = tmp_path / "session"
    stereo = tmp_path / "stereo"
    session.mkdir()
    stereo.mkdir()
    source = SimpleNamespace(
        captured=SimpleNamespace(
            captured_at_utc=datetime(2026, 8, 29, 0, 59, 59, tzinfo=UTC),
            raw_session_path=session,
            bundle=object(),
        ),
        stereo_path=stereo,
        stereo_metadata_sha256="a" * 64,
        session_manifest_sha256="b" * 64,
        session_view_metadata_sha256="c" * 64,
    )
    engine._sources = [source]
    hashes = {
        (stereo / "metadata.json").resolve(): "a" * 64,
        (session / "manifest.json").resolve(): "b" * 64,
    }
    monkeypatch.setattr(
        cycle_module,
        "_sha256",
        lambda path: hashes[Path(path).resolve()],
    )
    monkeypatch.setattr(
        cycle_module,
        "_single_view_metadata_hash",
        lambda *_args: "c" * 64,
    )
    monkeypatch.setattr(
        cycle_module,
        "read_stereo_inference",
        lambda path: SimpleNamespace(root=Path(path).resolve()),
    )
    verified = []
    monkeypatch.setattr(
        cycle_module,
        "verify_stereo_inference_source",
        lambda stored, *, expected_session: verified.append(
            (stored.root, Path(expected_session).resolve())
        ),
    )
    constructed = {}

    def fake_init(forked, **kwargs) -> None:
        constructed.update(kwargs)
        forked._sources = []

    monkeypatch.setattr(FoundationStereoOccupancyCycleEngine, "__init__", fake_init)

    forked = engine.fork_for_fine_science(
        settings=fine,
        reference_coarse_model=tmp_path / "reference",
        output_root=tmp_path / "fine",
    )

    assert forked._sources == [source]
    assert forked._sources is not engine._sources
    assert constructed["reference_coarse_model"] == tmp_path / "reference"
    assert constructed["accepted_coverage_path"] is None
    assert verified == [(stereo.resolve(), session.resolve())]


def test_fine_fork_rejects_changed_committed_source_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coarse, fine = _settings_pair()
    engine = _empty_engine(coarse)
    engine._sources = [
        SimpleNamespace(
            captured=SimpleNamespace(
                captured_at_utc=datetime(2026, 8, 29, 0, 59, 59, tzinfo=UTC),
                raw_session_path=tmp_path / "session",
                bundle=object(),
            ),
            stereo_path=tmp_path / "stereo",
            stereo_metadata_sha256="a" * 64,
            session_manifest_sha256="b" * 64,
            session_view_metadata_sha256="c" * 64,
        )
    ]
    monkeypatch.setattr(cycle_module, "_sha256", lambda _path: "f" * 64)

    with pytest.raises(FoundationStereoCycleError, match="evidence changed"):
        engine.fork_for_fine_science(
            settings=fine,
            reference_coarse_model=tmp_path / "reference",
            output_root=tmp_path / "fine",
        )


def test_fine_fork_reverifies_but_drops_expired_sources_for_fresh_rebootstrap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coarse, fine = _settings_pair()
    engine = _empty_engine(coarse)
    session = tmp_path / "session"
    stereo = tmp_path / "stereo"
    session.mkdir()
    stereo.mkdir()
    source = SimpleNamespace(
        captured=SimpleNamespace(
            captured_at_utc=datetime(2026, 8, 28, 23, 0, tzinfo=UTC),
            raw_session_path=session,
            bundle=object(),
        ),
        stereo_path=stereo,
        stereo_metadata_sha256="a" * 64,
        session_manifest_sha256="b" * 64,
        session_view_metadata_sha256="c" * 64,
    )
    engine._sources = [source]
    hashes = {
        (stereo / "metadata.json").resolve(): "a" * 64,
        (session / "manifest.json").resolve(): "b" * 64,
    }
    monkeypatch.setattr(cycle_module, "_sha256", lambda path: hashes[Path(path).resolve()])
    monkeypatch.setattr(cycle_module, "_single_view_metadata_hash", lambda *_args: "c" * 64)
    monkeypatch.setattr(
        cycle_module,
        "read_stereo_inference",
        lambda path: SimpleNamespace(root=Path(path).resolve()),
    )
    verified: list[tuple[Path, Path]] = []
    monkeypatch.setattr(
        cycle_module,
        "verify_stereo_inference_source",
        lambda stored, *, expected_session: verified.append(
            (stored.root, Path(expected_session).resolve())
        ),
    )

    def fake_init(forked, **_kwargs) -> None:
        forked._sources = []

    monkeypatch.setattr(FoundationStereoOccupancyCycleEngine, "__init__", fake_init)

    forked = engine.fork_for_fine_science(
        settings=fine,
        reference_coarse_model=tmp_path / "reference",
        output_root=tmp_path / "fine",
    )

    assert forked._sources == []
    assert verified == [(stereo.resolve(), session.resolve())]
