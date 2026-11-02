from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from app.inference.mask_fusion import fuse_component_masks, fuse_ranked_masks, rank_masks


@dataclass(frozen=True)
class InferenceConfig:
    rank_w_sam: float = 0.50
    rank_w_change: float = 0.35
    rank_w_point: float = 0.15
    negative_point_penalty: float = 0.40
    time_select_margin: float = 0.06
    cross_time_same_iou: float = 0.85
    time_top_k: int = 3
    component_threshold: float = 0.50
    global_threshold: float = 0.50


@dataclass(frozen=True)
class PairCandidate:
    mask: np.ndarray
    soft_mask: np.ndarray
    source_time: str
    score: float
    candidate_scores: dict[str, float]


def _prepare_candidates(
    masks: np.ndarray,
    scores: np.ndarray,
    prob_maps: np.ndarray | None,
    diff_map: np.ndarray,
    positive_points: list[tuple[int, int]],
    negative_points: list[tuple[int, int]],
    config: InferenceConfig,
) -> tuple[np.ndarray, np.ndarray, float, list[dict]] | None:
    hard_masks = np.asarray(masks)
    if hard_masks.ndim == 2:
        hard_masks = hard_masks[None, ...]
    hard_masks = hard_masks > 0
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(hard_masks) != len(scores):
        raise ValueError("each candidate mask must have one SAM quality score")
    if not len(hard_masks):
        return None
    if prob_maps is None:
        probabilities = hard_masks.astype(np.float32)
    else:
        probabilities = np.asarray(prob_maps, dtype=np.float32)
        if probabilities.shape != hard_masks.shape:
            raise ValueError("probability maps must match candidate mask shapes")
        probabilities = np.clip(probabilities, 0.0, 1.0)

    ranked = rank_masks(
        masks=hard_masks.astype(np.uint8),
        scores=scores,
        diff_map=np.asarray(diff_map, dtype=np.uint8),
        pos_points=positive_points,
        prob_maps=probabilities,
        w_sam=config.rank_w_sam,
        w_change=config.rank_w_change,
        w_point=config.rank_w_point,
    )
    height, width = hard_masks.shape[1:]
    for item in ranked:
        negative_hits = sum(
            0 <= x < width and 0 <= y < height and item["mask"][y, x] > 0
            for x, y in negative_points
        )
        item["final_score"] = max(
            0.0,
            float(item["final_score"])
            - config.negative_point_penalty * negative_hits / max(1, len(negative_points)),
        )
    ranked.sort(key=lambda item: item["final_score"], reverse=True)

    fused_binary, fused_soft = fuse_ranked_masks(
        ranked,
        top_k=config.time_top_k,
        threshold=config.component_threshold,
    )
    if fused_binary is None:
        return None

    mask_area = float(fused_binary.sum())
    if mask_area == 0:
        quality = 0.0
    else:
        diff_norm = np.asarray(diff_map, dtype=np.float32) / 255.0
        quality = float((fused_binary * diff_norm).sum() / (mask_area + 1e-6))
    return fused_binary.astype(np.uint8), fused_soft.astype(np.float32), quality, ranked


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    union = np.logical_or(mask_a, mask_b).sum()
    return float(np.logical_and(mask_a, mask_b).sum() / union) if union else 1.0


def select_pair_candidate(
    masks_t1: np.ndarray,
    scores_t1: np.ndarray,
    masks_t2: np.ndarray,
    scores_t2: np.ndarray,
    diff_map: np.ndarray,
    point: tuple[int, int],
    *,
    negative_points: Sequence[tuple[int, int]] = (),
    prob_maps_t1: np.ndarray | None = None,
    prob_maps_t2: np.ndarray | None = None,
    config: InferenceConfig | None = None,
) -> PairCandidate:
    config = config or InferenceConfig()
    positive_points = [point]
    negatives = list(negative_points)
    t1 = _prepare_candidates(
        masks_t1, scores_t1, prob_maps_t1, diff_map,
        positive_points, negatives, config,
    )
    t2 = _prepare_candidates(
        masks_t2, scores_t2, prob_maps_t2, diff_map,
        positive_points, negatives, config,
    )
    if t1 is None and t2 is None:
        raise ValueError("SAM2 returned no candidate masks for either time point")

    if t1 is None:
        chosen, source_time = t2, "t2"
    elif t2 is None:
        chosen, source_time = t1, "t1"
    else:
        mask_t1, _, quality_t1, ranked_t1 = t1
        mask_t2, _, quality_t2, ranked_t2 = t2
        if _iou(mask_t1, mask_t2) >= config.cross_time_same_iou:
            use_t1 = quality_t1 > quality_t2
        else:
            use_t1 = quality_t1 > quality_t2 + config.time_select_margin
        chosen, source_time = (t1, "t1") if use_t1 else (t2, "t2")

    mask, soft, _, ranked = chosen
    mask = mask.copy()
    soft = soft.copy()
    height, width = mask.shape
    for x, y in negatives:
        if 0 <= x < width and 0 <= y < height:
            mask[y, x] = 0
            soft[y, x] = 0.0
    score = float(ranked[0]["final_score"]) if ranked else 0.0
    candidate_scores = {
        "t1": float(t1[2]) if t1 is not None else 0.0,
        "t2": float(t2[2]) if t2 is not None else 0.0,
    }
    return PairCandidate(mask, soft, source_time, score, candidate_scores)


def fuse_point_masks(
    point_masks: Sequence[tuple[np.ndarray, float]],
    *,
    threshold: float = 0.5,
    image_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if not point_masks:
        if image_shape is None:
            raise ValueError("image_shape is required when fusing zero point masks")
        height, width = image_shape
        return np.zeros((height, width), dtype=np.uint8), np.zeros((height, width), dtype=np.float32)
    arrays = [np.asarray(mask, dtype=np.float32) for mask, _ in point_masks]
    shape = arrays[0].shape
    if any(mask.shape != shape or mask.ndim != 2 for mask in arrays):
        raise ValueError("all point masks must be 2D arrays with the same shape")
    binary, soft = fuse_component_masks(
        [(mask, score) for mask, (_, score) in zip(arrays, point_masks)],
        image_shape=shape,
        threshold=threshold,
    )
    return soft.astype(np.float32), binary.astype(np.uint8)
