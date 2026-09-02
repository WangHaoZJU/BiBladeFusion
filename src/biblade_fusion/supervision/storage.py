"""Atomic persistence for immutable supervisory-console snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from types import TracebackType
from uuid import uuid4

import numpy as np
from numpy.typing import ArrayLike

from biblade_fusion.diagnostics.performance_timing import performance_timed
from biblade_fusion.supervision.snapshot import (
    ArrayReference,
    AssetRecord,
    StoredSupervisorySnapshot,
    SupervisorySnapshot,
    read_supervisory_snapshot,
    snapshot_array_references,
)

_SAFE_ARRAY_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class AtomicSupervisorySnapshotWriter:
    """Write one self-contained snapshot without exposing partial state.

    Array references returned by :meth:`write_array` are the only references
    accepted by :meth:`commit`.  The destination is installed with one atomic
    rename and is never overwritten.
    """

    def __init__(self, output_dir: str | Path) -> None:
        self.output = Path(output_dir).resolve()
        if self.output.exists():
            raise FileExistsError(f"Supervisory snapshot already exists: {self.output}")
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = self.output.with_name(
            f".{self.output.name}.{uuid4().hex}.partial"
        )
        self.temporary.mkdir()
        (self.temporary / "arrays").mkdir()
        (self.temporary / "assets").mkdir()
        self._references: dict[str, ArrayReference] = {}
        self._assets: dict[str, AssetRecord] = {}
        self._committed = False

    def __enter__(self) -> AtomicSupervisorySnapshotWriter:
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if not self._committed:
            shutil.rmtree(self.temporary, ignore_errors=True)

    @performance_timed("live.asset_array_write")
    def write_array(
        self,
        name: str,
        array: ArrayLike,
        *,
        semantic: str,
        allow_nonfinite: bool = False,
    ) -> ArrayReference:
        """Persist one non-object NumPy array and return its immutable reference."""

        if self._committed:
            raise RuntimeError("Cannot add arrays after committing a snapshot")
        if not _SAFE_ARRAY_NAME.fullmatch(name):
            raise ValueError(f"Unsafe supervisory array name: {name!r}")
        if name in self._references:
            raise ValueError(f"Duplicate supervisory array name: {name}")
        if not semantic.strip():
            raise ValueError("Supervisory array semantic must be non-empty")
        value = np.asarray(array)
        if value.dtype.hasobject:
            raise ValueError("Object arrays are forbidden in supervision snapshots")
        if (
            not allow_nonfinite
            and np.issubdtype(value.dtype, np.number)
            and not bool(np.all(np.isfinite(value)))
        ):
            raise ValueError(f"Supervisory array {name} contains non-finite values")

        relative = Path("arrays") / f"{name}.npy"
        path = self.temporary / relative
        with path.open("wb") as stream:
            np.save(stream, value, allow_pickle=False)
            stream.flush()
            os.fsync(stream.fileno())
        reference = ArrayReference(
            path=relative.as_posix(),
            sha256=_sha256(path),
            dtype=value.dtype.str,
            shape=tuple(int(dimension) for dimension in value.shape),
            semantic=semantic,
            allow_nonfinite=allow_nonfinite,
        )
        self._references[name] = reference
        return reference

    @performance_timed("live.asset_file_write")
    def write_asset(
        self,
        name: str,
        source_path: str | Path,
        *,
        logical_name: str,
        kind: str,
        version: str | None = None,
        expected_sha256: str | None = None,
    ) -> AssetRecord:
        """Copy one small provenance file into the immutable snapshot.

        The optional expected digest closes the gap between an already-verified
        source record and the bytes copied here.  A source that changes while it is
        being copied is rejected rather than silently rebinding the snapshot.
        """

        if self._committed:
            raise RuntimeError("Cannot add assets after committing a snapshot")
        if not _SAFE_ARRAY_NAME.fullmatch(name):
            raise ValueError(f"Unsafe supervisory asset name: {name!r}")
        if name in self._assets:
            raise ValueError(f"Duplicate supervisory asset name: {name}")
        if not logical_name.strip() or not kind.strip():
            raise ValueError("Supervisory asset identity must be non-empty")
        source = Path(source_path).resolve()
        if not source.is_file():
            raise ValueError(f"Supervisory source asset does not exist: {source}")
        source_sha256 = _sha256(source)
        if expected_sha256 is not None and source_sha256 != expected_sha256:
            raise ValueError(f"Supervisory source asset checksum changed: {source}")

        suffix = source.suffix.lower()
        if suffix not in {".json", ".yaml", ".yml", ".toml"}:
            suffix = ".bin"
        relative = Path("assets") / f"{name}{suffix}"
        destination = self.temporary / relative
        with source.open("rb") as input_stream, destination.open("wb") as output_stream:
            shutil.copyfileobj(input_stream, output_stream)
            output_stream.flush()
            os.fsync(output_stream.fileno())
        if _sha256(source) != source_sha256 or _sha256(destination) != source_sha256:
            raise ValueError(f"Supervisory source asset changed during copy: {source}")

        record = AssetRecord(
            logical_name=logical_name,
            kind=kind,
            path=relative.as_posix(),
            sha256=source_sha256,
            version=version,
        )
        self._assets[name] = record
        return record

    @performance_timed("live.snapshot_commit")
    def commit(self, snapshot: SupervisorySnapshot) -> StoredSupervisorySnapshot:
        """Validate and atomically publish a snapshot using every written array."""

        if self._committed:
            raise RuntimeError("Supervisory snapshot writer has already committed")
        referenced = {item.path: item for item in snapshot_array_references(snapshot)}
        written = {item.path: item for item in self._references.values()}
        if referenced != written:
            missing = sorted(set(referenced) - set(written))
            unused = sorted(set(written) - set(referenced))
            raise ValueError(
                "Snapshot and writer array references differ; "
                f"missing={missing}, unused={unused}"
            )
        asset_paths = [item.path for item in snapshot.assets]
        if len(asset_paths) != len(set(asset_paths)):
            raise ValueError("Snapshot contains duplicate asset paths")
        referenced_assets = {item.path: item for item in snapshot.assets}
        written_assets = {item.path: item for item in self._assets.values()}
        if referenced_assets != written_assets:
            missing = sorted(set(referenced_assets) - set(written_assets))
            unused = sorted(set(written_assets) - set(referenced_assets))
            raise ValueError(
                "Snapshot and writer asset references differ; "
                f"missing={missing}, unused={unused}"
            )

        snapshot_path = self.temporary / "snapshot.json"
        with snapshot_path.open("w", encoding="utf-8") as stream:
            json.dump(
                snapshot.model_dump(mode="json"),
                stream,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(self.temporary / "arrays")
        _fsync_directory(self.temporary / "assets")
        _fsync_directory(self.temporary)
        read_supervisory_snapshot(snapshot_path)
        if self.output.exists():
            raise FileExistsError(
                f"Supervisory snapshot appeared during build: {self.output}"
            )
        self.temporary.replace(self.output)
        _fsync_directory(self.output.parent)
        self._committed = True
        return read_supervisory_snapshot(self.output)
