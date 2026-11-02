from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from app.domain.annotation import PairAnnotation, PointPrompt


@dataclass(frozen=True)
class ExportPaths:
    soft_mask: Path
    binary_mask: Path
    metadata: Path
    manual_mask: Path | None = None


def _mask_shape(pair: PairAnnotation) -> tuple[int, int]:
    for mask in (pair.soft_mask, pair.manual_override_mask):
        if mask is not None:
            array = np.asarray(mask)
            if array.ndim != 2:
                raise ValueError("annotation masks must be 2D")
            return array.shape
    with Image.open(pair.t1_path) as image:
        width, height = image.size
    return height, width


def _atomic_save_png(array: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{destination.stem}.", suffix=".png", dir=destination.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        Image.fromarray(array).save(temporary, format="PNG")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def export_annotation(
    pair: PairAnnotation, output_dir: str | Path, *, model_id: str
) -> ExportPaths:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = pair.t1_path.stem
    shape = _mask_shape(pair)
    soft = (
        np.zeros(shape, dtype=np.float32)
        if pair.soft_mask is None
        else np.asarray(pair.soft_mask)
    )
    if soft.shape != shape:
        raise ValueError(f"soft mask shape {soft.shape} does not match image shape {shape}")
    soft = np.nan_to_num(np.clip(soft, 0.0, 1.0), nan=0.0, posinf=1.0, neginf=0.0)
    soft_u8 = np.rint(soft * 255).astype(np.uint8)
    binary = np.zeros(shape, dtype=np.uint8) if pair.soft_mask is None else (soft >= pair.threshold).astype(np.uint8)

    manual_path: Path | None = None
    manual_u8: np.ndarray | None = None
    if pair.manual_override_mask is not None:
        override = np.asarray(pair.manual_override_mask, dtype=np.int8)
        if override.shape != shape or not np.isin(override, (-1, 0, 1)).all():
            raise ValueError("manual override mask must match image shape and contain only -1, 0, 1")
        binary[override == 0] = 0
        binary[override == 1] = 1
        manual_u8 = np.select(
            [override == -1, override == 0, override == 1], [0, 127, 255]
        ).astype(np.uint8)
        manual_path = output_dir / f"{stem}_manual.png"

    soft_path = output_dir / f"{stem}_soft.png"
    binary_path = output_dir / f"{stem}_binary.png"
    metadata_path = output_dir / f"{stem}.json"
    _atomic_save_png(soft_u8, soft_path)
    _atomic_save_png(binary * 255, binary_path)
    if manual_path is not None and manual_u8 is not None:
        _atomic_save_png(manual_u8, manual_path)

    height, width = shape
    metadata = {
        "version": 1,
        "t1_path": str(pair.t1_path.resolve()),
        "t2_path": str(pair.t2_path.resolve()),
        "image_size": {"width": width, "height": height},
        "points": [{"x": point.x, "y": point.y, "label": point.label} for point in pair.points],
        "threshold": pair.threshold,
        "review_status": pair.review_status,
        "prediction_pending": pair.prediction_pending,
        "has_soft_mask": pair.soft_mask is not None,
        "model_id": model_id,
        "masks": {
            "soft": soft_path.name,
            "binary": binary_path.name,
            "manual_override": manual_path.name if manual_path else None,
            "manual_encoding": {"unchanged": 0, "background": 127, "change": 255},
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return ExportPaths(soft_path, binary_path, metadata_path, manual_path)


def load_annotation(metadata_path: str | Path) -> PairAnnotation:
    metadata_path = Path(metadata_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mask_files = metadata["masks"]
    with Image.open(metadata_path.parent / mask_files["soft"]) as image:
        soft = np.asarray(image.convert("L"), dtype=np.float32).copy() / 255.0

    manual: np.ndarray | None = None
    manual_name = mask_files.get("manual_override")
    if manual_name:
        with Image.open(metadata_path.parent / manual_name) as image:
            encoded = np.asarray(image.convert("L"), dtype=np.uint8)
        manual = np.full(encoded.shape, -1, dtype=np.int8)
        manual[encoded == 127] = 0
        manual[encoded == 255] = 1

    return PairAnnotation(
        Path(metadata["t1_path"]),
        Path(metadata["t2_path"]),
        points=[PointPrompt(**point) for point in metadata["points"]],
        soft_mask=soft if metadata.get("has_soft_mask", True) else None,
        manual_override_mask=manual,
        threshold=float(metadata["threshold"]),
        review_status=metadata.get("review_status", "draft"),
        prediction_pending=metadata.get("prediction_pending", False),
    )
