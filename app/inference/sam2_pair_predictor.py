from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from app.domain.annotation import PointPrompt
from app.paths import model_checkpoint
from app.inference.change_response import build_augmented_change_map
from app.inference.change_selection import (
    InferenceConfig,
    fuse_point_masks,
    select_pair_candidate,
)


@dataclass(frozen=True)
class PairPrediction:
    soft_mask: np.ndarray
    binary_mask: np.ndarray
    source_time: str
    score: float
    candidate_scores: dict[str, float]


def _build_sam2_predictor(checkpoint_path: str | Path, device: str) -> Any:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint_path}")
    model = build_sam2(
        "sam2_hiera_l.yaml",
        str(checkpoint_path),
        device=device,
        mode="eval",
    )
    return SAM2ImagePredictor(model)


class PairSam2Predictor:
    """SAM2 image predictor that caches the two dates as a single image batch."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str = "cuda",
        *,
        predictor: Any | None = None,
        config: InferenceConfig | None = None,
    ) -> None:
        self.device = device
        self.config = config or InferenceConfig()
        self.predictor = predictor
        if self.predictor is None:
            self.predictor = _build_sam2_predictor(checkpoint_path or model_checkpoint(), device)
        self._images: tuple[np.ndarray, np.ndarray] | None = None
        self._change_map: np.ndarray | None = None

    def set_pair(self, t1_rgb: np.ndarray, t2_rgb: np.ndarray) -> None:
        t1 = np.asarray(t1_rgb)
        t2 = np.asarray(t2_rgb)
        if t1.ndim != 3 or t2.ndim != 3 or t1.shape[2] != 3 or t2.shape[2] != 3:
            raise ValueError("T1 and T2 must be HWC RGB arrays")
        if t1.shape != t2.shape:
            raise ValueError("T1 and T2 must have the same dimensions")
        if t1.dtype != np.uint8 or t2.dtype != np.uint8:
            raise ValueError("SAM2 inputs must be uint8 RGB arrays")
        self.predictor.set_image_batch([t1, t2])
        self._images = (t1, t2)

        self._change_map = build_augmented_change_map(t1, t2)

    @staticmethod
    def _as_candidate_arrays(logits: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logits = np.asarray(logits, dtype=np.float32)
        if logits.ndim == 4 and logits.shape[0] == 1:
            logits = logits[0]
        if logits.ndim == 2:
            logits = logits[None, ...]
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        if logits.ndim != 3 or len(logits) != len(scores):
            raise ValueError("SAM2 returned candidate logits/scores with incompatible shapes")
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
        return (logits > 0).astype(np.uint8), probabilities.astype(np.float32)

    def predict_change(self, points: Sequence[PointPrompt]) -> PairPrediction:
        if self._images is None or self._change_map is None:
            raise RuntimeError("call set_pair before requesting point predictions")
        positives = [point for point in points if point.label == 1]
        negatives = [(point.x, point.y) for point in points if point.label == 0]
        height, width = self._images[0].shape[:2]
        if not positives:
            return PairPrediction(
                np.zeros((height, width), dtype=np.float32),
                np.zeros((height, width), dtype=np.uint8),
                "none",
                0.0,
                {},
            )

        point_candidates = []
        for positive in positives:
            coords = np.asarray(
                [(positive.x, positive.y), *negatives], dtype=np.float32
            )
            labels = np.asarray([1, *([0] * len(negatives))], dtype=np.int32)
            logits_per_time, scores_per_time, _ = self.predictor.predict_batch(
                point_coords_batch=[coords, coords],
                point_labels_batch=[labels, labels],
                multimask_output=True,
                return_logits=True,
            )
            hard_t1, prob_t1 = self._as_candidate_arrays(logits_per_time[0], scores_per_time[0])
            hard_t2, prob_t2 = self._as_candidate_arrays(logits_per_time[1], scores_per_time[1])
            candidate = select_pair_candidate(
                hard_t1,
                scores_per_time[0],
                hard_t2,
                scores_per_time[1],
                self._change_map,
                (positive.x, positive.y),
                negative_points=negatives,
                prob_maps_t1=prob_t1,
                prob_maps_t2=prob_t2,
                config=self.config,
            )
            point_candidates.append(candidate)

        soft, binary = fuse_point_masks(
            [(candidate.soft_mask, candidate.score) for candidate in point_candidates],
            threshold=self.config.global_threshold,
            image_shape=(height, width),
        )
        source_times = {candidate.source_time for candidate in point_candidates}
        source_time = source_times.pop() if len(source_times) == 1 else "mixed"
        return PairPrediction(
            soft_mask=soft,
            binary_mask=binary,
            source_time=source_time,
            score=float(np.mean([candidate.score for candidate in point_candidates])),
            candidate_scores={
                "t1": float(np.mean([item.candidate_scores["t1"] for item in point_candidates])),
                "t2": float(np.mean([item.candidate_scores["t2"] for item in point_candidates])),
            },
        )
