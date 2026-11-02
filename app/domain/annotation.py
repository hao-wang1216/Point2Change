from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


REVIEW_LABELS = {
    "unstarted": "Not started",
    "draft": "Draft",
    "reviewed": "Reviewed",
    "no_change": "Reviewed: No change",
    "needs_review": "Needs review",
}


@dataclass(frozen=True)
class PointPrompt:
    x: int
    y: int
    label: int = 1

    def __post_init__(self) -> None:
        if self.label not in (0, 1):
            raise ValueError("point label must be 1 (positive) or 0 (negative)")


@dataclass
class PairAnnotation:
    t1_path: Path
    t2_path: Path
    points: list[PointPrompt] = field(default_factory=list)
    soft_mask: np.ndarray | None = None
    manual_override_mask: np.ndarray | None = None
    threshold: float = 0.5
    review_status: str = "unstarted"
    prediction_pending: bool = False

    def __post_init__(self) -> None:
        self.t1_path = Path(self.t1_path)
        self.t2_path = Path(self.t2_path)
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if self.review_status not in REVIEW_LABELS:
            raise ValueError(f"unknown review status: {self.review_status}")
        if not isinstance(self.prediction_pending, bool):
            raise ValueError("prediction_pending must be a boolean")
