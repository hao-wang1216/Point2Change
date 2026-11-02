"""Portable dataset annotations with an atomic, versioned snapshot commit."""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from pathlib import Path, PurePosixPath
from typing import Protocol

import numpy as np
from PIL import Image

from app.domain.annotation import REVIEW_LABELS, PairAnnotation, PointPrompt
from app.io.exporter import export_annotation


class _Pair(Protocol):
    key: str
    t1_path: Path
    t2_path: Path
    image_size: tuple[int, int]


_REVISION = re.compile(r"rev-[0-9a-f]{32}\Z")


def _relative_name(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        raise ValueError("expected a relative POSIX path")
    path = PurePosixPath(value)
    if not path.parts or path.is_absolute() or str(path) != value or any(part in (".", "..") for part in path.parts):
        raise ValueError(f"invalid relative path: {value}")
    return value


def _atomic_json(destination: Path, metadata: dict) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=".pending-", suffix=".json",
            dir=destination.parent, delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(metadata, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class DatasetStore:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.annotations_dir = self.root / "annotations"

    def metadata_path(self, pair: _Pair) -> Path:
        key = _relative_name(pair.key)
        destination = self.annotations_dir / key / "annotation.json"
        if not self.annotations_dir.resolve().is_relative_to(self.root) or \
                not destination.resolve().is_relative_to(self.annotations_dir.resolve()):
            raise ValueError("annotation directory escapes dataset")
        return destination

    def _source_name(self, path: Path) -> str:
        try:
            return Path(path).resolve().relative_to(self.root).as_posix()
        except ValueError as error:
            raise ValueError("image source is outside dataset") from error

    def _asset(self, sample_dir: Path, metadata: dict, name: object) -> Path:
        name = _relative_name(name)
        if PurePosixPath(name).parts[0] != metadata["revision"]:
            raise ValueError("mask does not belong to the committed revision")
        path = sample_dir / name
        revision_dir = sample_dir / metadata["revision"]
        if revision_dir.resolve().parent != sample_dir.resolve() or \
                not path.resolve().is_relative_to(revision_dir.resolve()):
            raise ValueError("mask path escapes annotation revision")
        if not path.is_file():
            raise ValueError(f"saved mask is missing: {name}")
        return path

    def _read_metadata(self, pair: _Pair) -> dict | None:
        path = self.metadata_path(pair)
        if not path.exists():
            return None
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if metadata.get("dataset_annotation_version") != 1:
                raise ValueError("unsupported dataset annotation version")
            revision = metadata["revision"]
            if not isinstance(revision, str) or not _REVISION.fullmatch(revision):
                raise ValueError("invalid annotation revision")
            if metadata["pair_key"] != pair.key:
                raise ValueError("saved annotation belongs to a different image pair")
            for name, source in (("t1_path", pair.t1_path), ("t2_path", pair.t2_path)):
                relative = _relative_name(metadata[name])
                if relative != self._source_name(source):
                    raise ValueError("saved image paths do not match the selected pair")
            width, height = pair.image_size
            if metadata["image_size"] != {"width": width, "height": height}:
                raise ValueError("saved dimensions do not match the selected pair")
            self._annotation_from_metadata(pair, metadata)
            if not isinstance(metadata["has_soft_mask"], bool):
                raise ValueError("has_soft_mask must be a boolean")
            for field in ("soft", "binary", "soft_array"):
                self._asset(path.parent, metadata, metadata["masks"][field])
            if metadata["masks"].get("manual_override"):
                self._asset(path.parent, metadata, metadata["masks"]["manual_override"])
            return metadata
        except (KeyError, TypeError, AttributeError, json.JSONDecodeError, OSError) as error:
            raise ValueError(f"invalid saved annotation: {path}: {error}") from error

    def _annotation_from_metadata(self, pair: _Pair, metadata: dict) -> PairAnnotation:
        annotation = PairAnnotation(
            pair.t1_path, pair.t2_path,
            points=[PointPrompt(**point) for point in metadata["points"]],
            threshold=metadata["threshold"],
            review_status=metadata["review_status"],
            prediction_pending=metadata["prediction_pending"],
        )
        self._validate_annotation(pair, annotation, validate_effective_mask=False)
        return annotation

    def _validate_annotation(
        self, pair: _Pair, annotation: PairAnnotation, *, validate_effective_mask: bool = True,
    ) -> None:
        width, height = pair.image_size
        if width <= 0 or height <= 0:
            raise ValueError("invalid image size")
        if annotation.review_status not in REVIEW_LABELS:
            raise ValueError("unknown review status")
        if not isinstance(annotation.prediction_pending, bool):
            raise ValueError("prediction_pending must be a boolean")
        if not isinstance(annotation.threshold, (int, float)) or not 0 <= annotation.threshold <= 1:
            raise ValueError("invalid mask threshold")
        if self._source_name(annotation.t1_path) != self._source_name(pair.t1_path) or \
                self._source_name(annotation.t2_path) != self._source_name(pair.t2_path):
            raise ValueError("annotation source does not match image pair")
        for point in annotation.points:
            if not isinstance(point.x, int) or isinstance(point.x, bool) or \
                    not isinstance(point.y, int) or isinstance(point.y, bool) or \
                    not 0 <= point.x < width or not 0 <= point.y < height or point.label not in (0, 1):
                raise ValueError("annotation point is outside image or invalid")
        if annotation.soft_mask is not None:
            soft = np.asarray(annotation.soft_mask)
            if soft.shape != (height, width) or soft.dtype.kind not in "fiu" or \
                    not np.isfinite(soft).all() or np.any(soft < 0) or np.any(soft > 1):
                raise ValueError("soft mask must match image dimensions and contain probabilities")
        if annotation.manual_override_mask is not None:
            manual = np.asarray(annotation.manual_override_mask)
            if manual.shape != (height, width) or not np.isin(manual, (-1, 0, 1)).all():
                raise ValueError("invalid manual override mask")
        if validate_effective_mask and annotation.review_status == "no_change":
            if self._effective_mask(annotation, (height, width)).any():
                raise ValueError("no-change annotation contains change pixels")

    @staticmethod
    def _effective_mask(annotation: PairAnnotation, shape: tuple[int, int]) -> np.ndarray:
        binary = np.zeros(shape, dtype=bool) if annotation.soft_mask is None else \
            np.asarray(annotation.soft_mask) >= annotation.threshold
        if annotation.manual_override_mask is not None:
            manual = np.asarray(annotation.manual_override_mask)
            binary[manual == 0] = False
            binary[manual == 1] = True
        return binary

    def load(self, pair: _Pair) -> PairAnnotation | None:
        metadata = self._read_metadata(pair)
        if metadata is None:
            return None
        sample_dir = self.metadata_path(pair).parent
        annotation = self._annotation_from_metadata(pair, metadata)
        try:
            with self._asset(sample_dir, metadata, metadata["masks"]["soft_array"]).open("rb") as handle:
                soft = np.load(handle, allow_pickle=False)
            if soft.shape != (pair.image_size[1], pair.image_size[0]) or \
                    soft.dtype.kind not in "fiu" or not np.isfinite(soft).all() or \
                    np.any(soft < 0) or np.any(soft > 1):
                raise ValueError("saved soft mask is invalid")
            annotation.soft_mask = soft if metadata["has_soft_mask"] else None
            if not metadata["has_soft_mask"] and np.any(soft):
                raise ValueError("absent soft mask has nonzero saved data")
            manual_name = metadata["masks"].get("manual_override")
            if manual_name:
                with Image.open(self._asset(sample_dir, metadata, manual_name)) as image:
                    manual_u8 = np.asarray(image).copy()
                if manual_u8.shape != soft.shape or not np.isin(manual_u8, (0, 127, 255)).all():
                    raise ValueError("saved manual mask is invalid")
                annotation.manual_override_mask = np.full(soft.shape, -1, dtype=np.int8)
                annotation.manual_override_mask[manual_u8 == 127] = 0
                annotation.manual_override_mask[manual_u8 == 255] = 1
            self._validate_annotation(pair, annotation)
            for field in ("soft", "binary"):
                with Image.open(self._asset(sample_dir, metadata, metadata["masks"][field])) as image:
                    preview = np.asarray(image).copy()
                if preview.shape != soft.shape:
                    raise ValueError(f"saved {field} mask dimensions are invalid")
                if field == "binary" and not np.array_equal(
                        preview, self._effective_mask(annotation, soft.shape).astype(np.uint8) * 255):
                    raise ValueError("saved binary mask disagrees with annotation")
            return annotation
        except (OSError, TypeError, KeyError, EOFError) as error:
            raise ValueError(f"invalid saved mask for {pair.key}: {error}") from error

    def save(self, pair: _Pair, annotation: PairAnnotation) -> Path:
        self._validate_annotation(pair, annotation)
        metadata_path = self.metadata_path(pair)
        # Reading the previous commit first also prevents overwriting a corrupt sample.
        previous = self._read_metadata(pair)
        if previous is not None:
            self.load(pair)
        revision = f"rev-{uuid.uuid4().hex}"
        revision_dir = metadata_path.parent / revision
        revision_dir.mkdir(parents=True, exist_ok=False)
        stem = Path(pair.t1_path).stem
        names = [f"{stem}_soft.png", f"{stem}_binary.png", f"{stem}.json", "soft.npy"]
        if annotation.manual_override_mask is not None:
            names.append(f"{stem}_manual.png")
        marker = {"version": 1, "pair_key": pair.key, "revision": revision, "files": names}
        try:
            _atomic_json(revision_dir / "revision.json", marker)
            paths = export_annotation(annotation, revision_dir, model_id="sam2_hiera_large")
            soft = np.zeros((pair.image_size[1], pair.image_size[0]), dtype=np.float32) \
                if annotation.soft_mask is None else np.asarray(annotation.soft_mask)
            with (revision_dir / "soft.npy").open("wb") as handle:
                np.save(handle, soft, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
            metadata.update({
                "dataset_annotation_version": 1, "pair_key": pair.key, "revision": revision,
                "path_base": "dataset_root", "t1_path": self._source_name(pair.t1_path),
                "t2_path": self._source_name(pair.t2_path),
                "has_soft_mask": annotation.soft_mask is not None,
            })
            metadata["masks"].update({
                "soft": f"{revision}/{paths.soft_mask.name}",
                "binary": f"{revision}/{paths.binary_mask.name}",
                "manual_override": f"{revision}/{paths.manual_mask.name}" if paths.manual_mask else None,
                "soft_array": f"{revision}/soft.npy",
            })
            # Keep every stored source path portable, including the revision's copy.
            _atomic_json(paths.metadata, metadata)
            _atomic_json(metadata_path, metadata)
        except Exception:
            self._remove_owned_revision(revision_dir, pair.key)
            raise
        retained = {revision}
        if previous:
            retained.add(previous["revision"])
        self._clean_old_revisions(metadata_path.parent, pair.key, retained)
        return metadata_path

    def read_status(self, pair: _Pair) -> str:
        metadata = self._read_metadata(pair)
        return "unstarted" if metadata is None else metadata["review_status"]

    def read_last_key(self) -> str | None:
        path = self.annotations_dir / "project.json"
        if not path.exists():
            return None
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if metadata.get("version") != 1:
                raise ValueError("unsupported project metadata version")
            return _relative_name(metadata["last_key"])
        except (KeyError, AttributeError, TypeError, OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid dataset project metadata: {error}") from error

    def write_last_key(self, key: str) -> None:
        _atomic_json(self.annotations_dir / "project.json", {"version": 1, "last_key": _relative_name(key)})

    def _clean_old_revisions(self, sample_dir: Path, pair_key: str, retained: set[str]) -> None:
        try:
            for child in sample_dir.iterdir():
                if child.name not in retained:
                    self._remove_owned_revision(child, pair_key)
        except OSError:
            pass  # A committed snapshot remains usable even if old files are locked.

    @staticmethod
    def _remove_owned_revision(directory: Path, pair_key: str) -> None:
        if not _REVISION.fullmatch(directory.name) or not directory.is_dir() or directory.is_symlink():
            return
        if directory.resolve().parent != directory.parent.resolve():
            return
        try:
            marker = json.loads((directory / "revision.json").read_text(encoding="utf-8"))
            if marker.get("version") != 1 or marker.get("pair_key") != pair_key or \
                    marker.get("revision") != directory.name:
                return
            names = marker["files"]
            if not isinstance(names, list) or any(
                    not isinstance(name, str) or Path(name).name != name or name in (".", "..")
                    for name in names):
                return
            allowed = set(names) | {"revision.json"}
            children = list(directory.iterdir())
            if any(child.name not in allowed or not child.is_file() or child.is_symlink() for child in children):
                return
            for child in children:
                child.unlink()
            directory.rmdir()
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            pass
