import numpy as np
from typing import Any, Dict, List, Tuple


def filter_masks_by_area(
    masks: np.ndarray,
    scores: np.ndarray,
    image_shape: Tuple[int, int],
    min_ratio: float = 0.001,
    max_ratio: float = 0.3,
):
    """
    Filter extreme candidates whose areas are too small or too large.
    """
    h, w = image_shape
    valid_masks = []
    valid_scores = []
    valid_indices = []

    for i in range(len(masks)):
        area = float(masks[i].sum())
        ratio = area / (h * w)
        if min_ratio < ratio < max_ratio:
            valid_masks.append(masks[i])
            valid_scores.append(scores[i])
            valid_indices.append(i)

    if len(valid_masks) == 0:
        return masks, scores, list(range(len(masks)))
    return np.array(valid_masks), np.array(valid_scores), valid_indices


def score_mask_with_change(mask: np.ndarray, diff_map: np.ndarray) -> float:
    mask_area = float(mask.sum())
    if mask_area < 1:
        return 0.0

    diff_norm = diff_map.astype(np.float32) / 255.0
    return float((mask.astype(np.float32) * diff_norm).sum() / (mask_area + 1e-6))


def point_coverage_score(mask: np.ndarray, pos_points: List[Tuple[int, int]]) -> float:
    if len(pos_points) == 0:
        return 0.0

    hit = 0
    h, w = mask.shape
    for x, y in pos_points:
        if 0 <= x < w and 0 <= y < h and mask[y, x] > 0:
            hit += 1
    return float(hit / len(pos_points))


def rank_masks(
    masks: np.ndarray,
    scores: np.ndarray,
    diff_map: np.ndarray,
    pos_points: List[Tuple[int, int]],
    prob_maps: np.ndarray = None,
    w_sam: float = 0.5,
    w_change: float = 0.35,
    w_point: float = 0.15,
) -> List[Dict[str, Any]]:
    """
    Rank candidates using SAM2 quality, change response, and point coverage.
    """
    ranked = []
    for i in range(len(masks)):
        m = masks[i].astype(np.uint8)
        sam_score = float(scores[i])
        change_score = score_mask_with_change(m, diff_map)
        point_score = point_coverage_score(m, pos_points)

        final_score = w_sam * sam_score + w_change * change_score + w_point * point_score
        ranked.append({
            "mask": m,
            "prob_map": prob_maps[i].astype(np.float32) if prob_maps is not None else m.astype(np.float32),
            "sam_score": sam_score,
            "change_score": change_score,
            "point_score": point_score,
            "final_score": final_score,
            "area": int(m.sum()),
            "index": i,
        })

    ranked.sort(key=lambda x: x["final_score"], reverse=True)
    return ranked


def fuse_ranked_masks(
    ranked_masks: List[Dict[str, Any]],
    top_k: int = 3,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Locally fuse multiple candidates for one point.
    """
    if len(ranked_masks) == 0:
        return None, None

    top_items = ranked_masks[:min(top_k, len(ranked_masks))]
    h, w = top_items[0]["mask"].shape
    soft_map = np.zeros((h, w), dtype=np.float32)

    for item in top_items:
        soft_map += item["prob_map"].astype(np.float32) * float(item["final_score"])

    max_val = float(soft_map.max())
    if max_val > 1e-6:
        soft_map = soft_map / max_val

    fused_mask = (soft_map >= threshold).astype(np.uint8)
    return fused_mask, soft_map


def fuse_component_masks(
    component_masks_with_scores: List[Tuple[np.ndarray, float]],
    image_shape: Tuple[int, int],
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Use pixelwise maximum for global fusion instead of summation.

    Summation can raise scores between dense change regions and create bridges.
    Maximum fusion avoids filling gaps by accumulating neighboring masks.
    """
    h, w = image_shape
    soft_map = np.zeros((h, w), dtype=np.float32)

    if len(component_masks_with_scores) == 0:
        return np.zeros((h, w), dtype=np.uint8), soft_map

    for prob_map, score in component_masks_with_scores:
        soft_map = np.maximum(soft_map, prob_map.astype(np.float32) * float(score))

    max_val = float(soft_map.max())
    if max_val > 1e-6:
        soft_map = soft_map / max_val

    pseudo_mask = (soft_map >= threshold).astype(np.uint8)
    return pseudo_mask, soft_map
